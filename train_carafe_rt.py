#!/usr/bin/env python3
"""Fine-tune DIA-NN's RT model directly on Carafe2 RT training data."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path

import pandas as pd

from diann_finetune import __version__
from diann_finetune.catalog import DEFAULT_CATALOG, resolve_catalog_build
from diann_finetune.data import (
    carafe_sequence_split,
    load_carafe_rt,
    nested_training_subset,
    oversample_heavy_dimethyl_rows,
)
from diann_finetune.diagnostics import generate_run_diagnostics
from diann_finetune.encoding import (
    TokenEncoder,
    augment_tokens_for_heavy_dimethyl,
    read_tokens,
    write_tokens,
)
from diann_finetune.training import (
    ORGANISM_SAMPLING_STRATEGIES,
    finalize_distributed,
    initialize_distributed,
    is_primary_process,
    resolve_device,
    train_rt_model,
    write_metadata,
)
from diann_finetune.tracking import WandbTracker


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(os.environ.get("DIANN_MODEL_DIR", ROOT / "models")).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DIA-NN rt.d0.pt using Carafe2 rt_train_data.tsv"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--in-dir", type=Path, help="Carafe output directory containing rt_train_data.tsv")
    source.add_argument("--dataset-id", help="Build ID from datasets.yaml; created automatically if absent")
    source.add_argument("--rt-data", type=Path, help="Direct path to a Carafe rt_train_data.tsv")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Fine-tuning datasets.yaml for --dataset-id")
    parser.add_argument("--train-split", type=Path, help="Prepared, peptidoform-disjoint training TSV")
    parser.add_argument("--validation-split", type=Path, help="Prepared, peptidoform-disjoint validation TSV")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="rt.d0.pt", help="DIA-NN RT model filename")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-learning-rate", type=float,
                        help="Backbone LR for --trainable discriminative (default: learning-rate / 10)")
    parser.add_argument(
        "--embedding-learning-rate", type=float,
        help="LR for custom UniMod 510 rows (default: learning-rate)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=("mae", "huber", "mse"), default="mae")
    parser.add_argument(
        "--huber-delta", type=float, default=0.03,
        help="Transition from quadratic to linear error for --loss huber",
    )
    parser.add_argument(
        "--normal-weight", type=float, default=1.0,
        help="Loss weight for rows without Heavy-Dimethyl; use a small positive value for target-focused training",
    )
    parser.add_argument("--heavy-nterm-weight", type=float, default=1.0)
    parser.add_argument("--heavy-lysine-weight", type=float, default=1.0)
    parser.add_argument(
        "--heavy-both-weight", type=float,
        help="Weight for N-term+lysine heavy rows (default: product of both site weights)",
    )
    parser.add_argument("--heavy-nterm-oversample", type=float, default=1.0)
    parser.add_argument("--heavy-lysine-oversample", type=float, default=1.0)
    parser.add_argument("--heavy-both-oversample", type=float, default=1.0)
    parser.add_argument(
        "--save-heavy-checkpoint-grid",
        action="store_true",
        help=(
            "Save validation-best checkpoints for 100/0, 75/25, 50/50, 25/75, and "
            "0/100 Overall/Heavy-Macro MAE priorities in addition to the standard 100/0 checkpoint"
        ),
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--organism-sampling",
        choices=ORGANISM_SAMPLING_STRATEGIES,
        help="Per-epoch source-organism sampling policy for balancing studies",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int,
        help="Hard optimizer-step budget; required with --organism-sampling",
    )
    parser.add_argument("--default-charge", type=int, default=2,
                        help="Placeholder charge required by DIA-NN input; its RT model ignores this column")
    parser.add_argument(
        "--trainable",
        choices=(
            "all", "output", "embeddings", "head", "rt_head", "cnn_head",
            "gru_head", "discriminative",
        ),
        default="all",
                        help="Layer policy; output is a backwards-compatible alias for head")
    parser.add_argument("--unknown-mods", choices=("error", "drop"), default="error")
    parser.add_argument("--max-rows", type=int, help="Development/smoke-test row limit")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Deterministic fraction of the fixed training split to use (0 < value <= 1)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG and HTML diagnostics")
    parser.add_argument("--wandb-config", type=Path, help="JSON configuration enabling W&B tracking")
    parser.add_argument("--wandb-run-name", help="W&B run name (default: output directory name)")
    parser.add_argument("--wandb-group", help="W&B campaign group")
    parser.add_argument("--wandb-run-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    if not 0.0 < args.train_fraction <= 1.0:
        raise SystemExit("--train-fraction must be greater than 0 and at most 1")
    if args.huber_delta <= 0:
        raise SystemExit("--huber-delta must be positive")
    if args.organism_sampling is not None and args.steps_per_epoch is None:
        raise SystemExit("--organism-sampling requires --steps-per-epoch")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise SystemExit("--steps-per-epoch must be positive")
    if min(args.normal_weight, args.heavy_nterm_weight, args.heavy_lysine_weight) <= 0 or (
        args.heavy_both_weight is not None and args.heavy_both_weight <= 0
    ):
        raise SystemExit("All heavy-dimethyl weights must be positive")
    if min(
        args.heavy_nterm_oversample,
        args.heavy_lysine_oversample,
        args.heavy_both_oversample,
    ) < 1.0:
        raise SystemExit("All heavy-dimethyl oversampling factors must be at least 1")
    split_mode = args.train_split is not None or args.validation_split is not None
    if split_mode and (args.train_split is None or args.validation_split is None):
        raise SystemExit("--train-split and --validation-split must be supplied together")
    if split_mode and (args.in_dir is not None or args.dataset_id is not None or args.rt_data is not None):
        raise SystemExit("Use either --in-dir/--rt-data or --train-split plus --validation-split")
    if not split_mode and args.in_dir is None and args.dataset_id is None and args.rt_data is None:
        raise SystemExit("Provide --in-dir, --rt-data, or both prepared split files")
    if split_mode and args.train_fraction != 1.0:
        raise SystemExit("--train-fraction cannot be used with prepared split files")
    device = resolve_device(args.device)
    initialize_distributed(device)
    atexit.register(finalize_distributed)
    primary = is_primary_process()
    # A fine-tune catalog build has dedicated child inputs for both model
    # families. Accepting its root here keeps the CLI concise while still
    # ensuring DIA-NN consumes the exact RT table documented by the build.
    if args.dataset_id is not None:
        args.in_dir = resolve_catalog_build(args.dataset_id, args.catalog)
    if args.rt_data is not None:
        rt_path = args.rt_data
    elif args.in_dir is not None:
        catalog_rt = args.in_dir / "diann_rt_input" / "rt_train_data.tsv"
        rt_path = catalog_rt if catalog_rt.is_file() else args.in_dir / "rt_train_data.tsv"
    else:
        rt_path = None
    model_path = args.model_dir / args.model
    dictionary_path = args.model_dir / "dict.txt"
    required_inputs = [model_path, dictionary_path]
    required_inputs.extend([args.train_split, args.validation_split] if split_mode else [rt_path])
    for path in required_inputs:
        if not path.is_file():
            raise SystemExit(f"Required input file not found: {path}")

    if split_mode:
        prepared_train = load_carafe_rt(args.train_split, unknown_mods=args.unknown_mods, max_rows=args.max_rows)
        prepared_validation = load_carafe_rt(args.validation_split, unknown_mods=args.unknown_mods)
        train_frame, validation_frame = prepared_train.frame, prepared_validation.frame
        # Prepared partitions separate peptidoforms; different modifications
        # of the same amino-acid sequence may occur in separate partitions.
        split_key = ["sequence", "mods", "mod_sites"]
        train_keys = set(map(tuple, train_frame.loc[:, split_key].itertuples(index=False, name=None)))
        validation_keys = set(map(tuple, validation_frame.loc[:, split_key].itertuples(index=False, name=None)))
        if not train_keys.isdisjoint(validation_keys):
            raise SystemExit("Prepared train and validation splits share exact peptidoforms")
        prepared = prepared_train
        split_metadata = {
            "split_source": "prepared_peptidoform_disjoint_files",
            "complete_training_rows": len(train_frame),
            "used_training_rows": len(train_frame),
            "train_fraction": 1.0,
            "used_training_unique_sequences": train_frame["sequence"].nunique(),
            "training_rows": len(train_frame),
            "training_unique_sequences": train_frame["sequence"].nunique(),
            "validation_rows": len(validation_frame),
            "validation_unique_sequences": validation_frame["sequence"].nunique(),
        }
    else:
        prepared = load_carafe_rt(rt_path, unknown_mods=args.unknown_mods, max_rows=args.max_rows)
        complete_train_frame, validation_frame, split_metadata = carafe_sequence_split(prepared.frame)
        train_frame = nested_training_subset(
            complete_train_frame, args.train_fraction, random_state=args.seed
        )
        split_metadata["complete_training_rows"] = len(complete_train_frame)
        split_metadata["used_training_rows"] = len(train_frame)
        split_metadata["train_fraction"] = args.train_fraction
        split_metadata["used_training_unique_sequences"] = train_frame["sequence"].nunique()
    train_frame, oversampling_metadata = oversample_heavy_dimethyl_rows(
        train_frame,
        nterm_factor=args.heavy_nterm_oversample,
        lysine_factor=args.heavy_lysine_oversample,
        both_factor=args.heavy_both_oversample,
        random_state=args.seed,
    )
    split_metadata["training_rows_before_oversampling"] = oversampling_metadata["base_rows"]
    split_metadata["training_rows_after_oversampling"] = oversampling_metadata["oversampled_rows"]
    tokens, embedding_initializations = augment_tokens_for_heavy_dimethyl(
        read_tokens(dictionary_path)
    )
    if primary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_frame.to_csv(args.output_dir / "train_split.tsv", sep="\t", index=False)
        validation_frame.to_csv(args.output_dir / "validation_split.tsv", sep="\t", index=False)
    output_dictionary = args.output_dir / "dict.txt"
    if primary:
        write_tokens(output_dictionary, tokens)
    encoder = TokenEncoder(tokens)
    if primary:
        print(
            f"Loaded {len(prepared.frame)} Carafe RT rows; training={len(train_frame)}, "
            f"validation={len(validation_frame)}, device={device}"
        )

    metadata = {
        "workflow_version": __version__,
        "task": "DIA-NN RT fine-tuning from Carafe2 training data",
        "input": str(rt_path.resolve()) if rt_path else str(args.train_split.resolve()),
        "model": str(model_path.resolve()),
        "dictionary": str(dictionary_path.resolve()),
        "output_dictionary": str(output_dictionary.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "parameters": {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "embedding_learning_rate": args.embedding_learning_rate,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "huber_delta": args.huber_delta,
            "normal_weight": args.normal_weight,
            "heavy_nterm_weight": args.heavy_nterm_weight,
            "heavy_lysine_weight": args.heavy_lysine_weight,
            "heavy_both_weight": args.heavy_both_weight,
            "heavy_nterm_oversample": args.heavy_nterm_oversample,
            "heavy_lysine_oversample": args.heavy_lysine_oversample,
            "heavy_both_oversample": args.heavy_both_oversample,
            "save_heavy_checkpoint_grid": args.save_heavy_checkpoint_grid,
            "seed": args.seed,
            "device": str(device),
            "trainable": args.trainable,
            "default_charge": args.default_charge,
            "train_fraction": args.train_fraction,
            "organism_sampling": args.organism_sampling,
            "steps_per_epoch": args.steps_per_epoch,
        },
        "split": split_metadata,
        "oversampling": oversampling_metadata,
        "dropped_rows": prepared.dropped_rows,
        "custom_token_usage": prepared.custom_token_counts,
        "embedding_initializations": {
            str(new_id): source_id for new_id, source_id in embedding_initializations.items()
        },
    }
    tracker = WandbTracker(
        args.wandb_config if primary else None,
        run_name=args.wandb_run_name or args.output_dir.name,
        group=args.wandb_group,
        run_config={**metadata["parameters"], **split_metadata},
        output_dir=args.output_dir,
        run_id=args.wandb_run_id,
    )
    try:
        metadata["result"] = train_rt_model(
            model_path,
            train_frame,
            validation_frame,
            encoder,
            args.output_dir,
            device=device,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            backbone_learning_rate=args.backbone_learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            default_charge=args.default_charge,
            trainable=args.trainable,
            embedding_initializations=embedding_initializations,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            normal_weight=args.normal_weight,
            heavy_nterm_weight=args.heavy_nterm_weight,
            heavy_lysine_weight=args.heavy_lysine_weight,
            heavy_both_weight=args.heavy_both_weight,
            embedding_learning_rate=args.embedding_learning_rate,
            organism_sampling=args.organism_sampling,
            steps_per_epoch=args.steps_per_epoch,
            save_heavy_checkpoint_grid=args.save_heavy_checkpoint_grid,
            epoch_callback=tracker.log_epoch,
        )
        if primary:
            write_metadata(args.output_dir / "run_metadata.json", metadata)
        if primary and not args.no_plots:
            metadata["diagnostics"] = generate_run_diagnostics(args.output_dir)
            write_metadata(args.output_dir / "run_metadata.json", metadata)
        tracker.finish(metadata, args.output_dir)
    except Exception:
        tracker.finish(metadata, args.output_dir, exit_code=1)
        raise
    finally:
        finalize_distributed()
    if primary:
        print(json.dumps(metadata["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
