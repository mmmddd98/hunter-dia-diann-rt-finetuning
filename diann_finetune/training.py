"""Training and evaluation utilities for DIA-NN RT TorchScript models."""

from __future__ import annotations

import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .encoding import TokenEncoder
from .data import heavy_dimethyl_row_strata


# These labels are deliberately ordered from an overall-focused to a
# Heavy-Dimethyl-focused checkpoint choice.  They are validation-only scores;
# training loss weights and oversampling never enter this calculation.
HEAVY_CHECKPOINT_SELECTIONS: Dict[str, Tuple[float, float]] = {
    "100_0": (1.0, 0.0),
    "75_25": (0.75, 0.25),
    "50_50": (0.50, 0.50),
    "25_75": (0.25, 0.75),
    "0_100": (0.0, 1.0),
}


def heavy_checkpoint_selection_scores(metrics: Dict[str, float]) -> Dict[str, float]:
    """Return the five validation-only overall/Heavy-Macro checkpoint scores.

    ``Heavy-Macro`` is the *unweighted* arithmetic mean of the three exact
    Heavy-Dimethyl strata.  Consequently a common Heavy-Lys class cannot
    dominate the rarer Heavy-N-term or Heavy-both classes merely by having
    more validation rows.  All inputs are model-scale MAEs, so every term in a
    score has the same unit.
    """
    required = ("mae", "heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae")
    missing = [name for name in required if name not in metrics or not math.isfinite(float(metrics[name]))]
    if missing:
        raise ValueError(f"Cannot calculate Heavy checkpoint scores; missing metrics: {missing}")
    heavy_macro = sum(float(metrics[name]) for name in required[1:]) / 3.0
    return {
        label: overall_weight * float(metrics["mae"]) + heavy_weight * heavy_macro
        for label, (overall_weight, heavy_weight) in HEAVY_CHECKPOINT_SELECTIONS.items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_world_size() -> int:
    """Return the active torchrun world size, or one outside distributed runs."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_rank() -> int:
    """Return the active global rank, or zero outside distributed runs."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_primary_process() -> bool:
    return distributed_rank() == 0


def initialize_distributed(device: torch.device) -> None:
    """Initialize NCCL when this process was launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    if device.type != "cuda":
        raise RuntimeError("Multi-process DIA-NN training requires CUDA")
    if not dist.is_available():
        raise RuntimeError("This PyTorch build does not provide torch.distributed")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def finalize_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    if requested == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(requested)


def verify_torchscript_model(path: Path) -> None:
    """Give a useful error when Git LFS left only a pointer file."""
    with path.open("rb") as handle:
        prefix = handle.read(80)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not model data. Install Git LFS and run "
            "`git lfs pull`, or pass --model-dir pointing to an installed DIA-NN models folder."
        )


def flatten_rnn_parameters(model: torch.jit.ScriptModule) -> None:
    """Ensure cuDNN sees DIA-NN's recurrent weights as one contiguous block.

    Loading a TorchScript module and switching between train/eval can invalidate
    cuDNN's cached RNN weight layout.  Refresh it at those boundaries instead
    of letting cuDNN compact the weights on every forward pass.
    """
    try:
        rnn = model.rnn
        flatten = getattr(rnn, "flatten_parameters", None)
        if flatten is None:
            # DIA-NN's bundled RNN is a RecursiveScriptModule and does not
            # expose PyTorch's flatten_parameters().  Calling the cuDNN path
            # therefore triggers a costly compaction on every forward pass.
            # Use the native CUDA RNN implementation for this model instead.
            # It is a little slower than cuDNN, but avoids repeated compaction
            # and the associated memory spikes.
            torch.backends.cudnn.enabled = False
            return
        flatten()
    except (AttributeError, RuntimeError):
        # Models without an exposed RNN, or CPU-only backends, do not need this.
        pass


def load_torchscript_model(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load a DIA-NN model and synchronize its serialized device attribute.

    DIA-NN's forward pass creates positional tensors from ``self.device``.
    PyTorch's ``module.to(device)`` moves parameters but does not update this
    scripted string attribute, so both operations are required.
    """
    model = torch.jit.load(str(path), map_location=device).to(device)
    if hasattr(model, "device"):
        model.device = str(device)
    flatten_rnn_parameters(model)
    return model


def save_portable_torchscript(model: torch.jit.ScriptModule, path: Path) -> None:
    """Save a checkpoint whose tensors and scripted device are both on CPU.

    DIA-NN's CPU-only binary loads TorchScript tensor storages at their saved
    device.  Merely changing ``model.device`` is insufficient: a model trained
    on CUDA would still serialize its parameters as CUDA tensors and fail
    during DIA-NN loading.  Round-tripping through an in-memory CPU load keeps
    the live training model (and its optimizer state) untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = io.BytesIO()
    torch.jit.save(model, serialized)
    serialized.seek(0)
    portable_model = torch.jit.load(serialized, map_location="cpu").cpu()
    if hasattr(portable_model, "device"):
        portable_model.device = "cpu"
    torch.jit.save(portable_model, str(path))

    # Omit map_location to verify compatibility with DIA-NN's CPU-only loader.
    reloaded = torch.jit.load(str(path))
    parameter_devices = {parameter.device.type for parameter in reloaded.parameters()}
    if parameter_devices != {"cpu"} or getattr(reloaded, "device", "cpu") != "cpu":
        raise RuntimeError(
            f"Portable checkpoint validation failed for {path}: "
            f"parameters={sorted(parameter_devices)}, "
            f"scripted_device={getattr(reloaded, 'device', '<absent>')!r}"
        )


def prepare_examples(frame: pd.DataFrame, encoder: TokenEncoder, default_charge: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    encoded = encoder.encode_many(frame["diann_sequence"].tolist(), charge=default_charge)
    targets = torch.tensor(frame["rt_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return encoded, targets


def build_loss_function(name: str, huber_delta: float) -> torch.nn.Module:
    """Build an unreduced regression loss so optional row weights can be applied."""
    if name == "mae":
        return torch.nn.L1Loss(reduction="none")
    if name == "huber":
        if huber_delta <= 0:
            raise ValueError("Huber delta must be positive")
        return torch.nn.HuberLoss(delta=huber_delta, reduction="none")
    if name == "mse":
        return torch.nn.MSELoss(reduction="none")
    raise ValueError(f"Unknown loss: {name}")


def training_sample_weights(
    frame: pd.DataFrame,
    *,
    normal_weight: float = 1.0,
    heavy_nterm_weight: float,
    heavy_lysine_weight: float,
    heavy_both_weight: float | None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Assign transparent category weights to heavy-dimethyl training rows.

    N-terminal-only and lysine-only rows use their respective weights. Rows
    containing both sites use ``heavy_both_weight`` when given, otherwise the
    product of both site weights. All remaining rows use ``normal_weight``.
    The four values may be derived from desired *group-level* loss mass; the
    caller is responsible for accounting for group sizes after oversampling.
    """
    values = (normal_weight, heavy_nterm_weight, heavy_lysine_weight)
    if any(value <= 0 for value in values) or (
        heavy_both_weight is not None and heavy_both_weight <= 0
    ):
        raise ValueError("Every sample weight must be positive")
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False)
    lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False)
    both = nterm & lysine
    weights = np.full(len(frame), normal_weight, dtype=np.float32)
    weights[nterm & ~lysine] = heavy_nterm_weight
    weights[lysine & ~nterm] = heavy_lysine_weight
    weights[both] = (
        heavy_both_weight
        if heavy_both_weight is not None
        else heavy_nterm_weight * heavy_lysine_weight
    )
    counts = {
        "unweighted": int((~nterm & ~lysine).sum()),
        "heavy_nterm_only": int((nterm & ~lysine).sum()),
        "heavy_lysine_only": int((lysine & ~nterm).sum()),
        "heavy_nterm_and_lysine": int(both.sum()),
    }
    return torch.from_numpy(weights), counts


def length_grouped_batches(
    encoded: Sequence[torch.Tensor],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    indices: Sequence[int] | None = None,
) -> Iterator[List[int]]:
    groups: Dict[int, List[int]] = {}
    selected = range(len(encoded)) if indices is None else indices
    for index in selected:
        tensor = encoded[index]
        groups.setdefault(tensor.numel(), []).append(index)
    rng = random.Random(seed)
    batches: List[List[int]] = []
    for indices in groups.values():
        indices = list(indices)
        if shuffle:
            rng.shuffle(indices)
        batches.extend(indices[start : start + batch_size] for start in range(0, len(indices), batch_size))
    if shuffle:
        rng.shuffle(batches)
    yield from batches


ORGANISM_SAMPLING_STRATEGIES = (
    "natural",
    "round_robin_batches",
    "ratio_1_1_1",
    "ratio_2_1_1",
    "ratio_4_1_1",
    "ratio_8_1_1",
    "ratio_1_2_1",
    "ratio_1_1_2",
    "ratio_2_2_1",
    "ratio_2_1_2",
    "ratio_4_2_1",
    "ratio_4_1_2",
    "inverse_frequency",
    "temperature_0_25",
    "temperature_0_50",
    "temperature_0_75",
    "curriculum_natural_to_uniform",
    "curriculum_uniform_to_natural",
    "curriculum_ce4_to_uniform",
    "curriculum_uniform_to_ce4",
)
ORGANISM_ORDER = ("celegans", "human", "triplemix")
ORGANISM_RATIO_STRATEGIES = {
    "ratio_1_1_1": (1, 1, 1),
    "ratio_2_1_1": (2, 1, 1),
    "ratio_4_1_1": (4, 1, 1),
    "ratio_8_1_1": (8, 1, 1),
    "ratio_1_2_1": (1, 2, 1),
    "ratio_1_1_2": (1, 1, 2),
    "ratio_2_2_1": (2, 2, 1),
    "ratio_2_1_2": (2, 1, 2),
    "ratio_4_2_1": (4, 2, 1),
    "ratio_4_1_2": (4, 1, 2),
}
ORGANISM_TEMPERATURES = {
    "temperature_0_25": 0.25,
    "temperature_0_50": 0.50,
    "temperature_0_75": 0.75,
}


def sample_organism_indices(
    organisms: Sequence[str], strategy: str, sample_rows: int, seed: int,
    progress: float = 0.0,
) -> List[int]:
    """Draw one epoch's row indices for a requested organism mixture.

    Ratio strategies use exact global quotas. ``inverse_frequency`` is the
    stochastic weighted-sampling counterpart of 1:1:1. Sampling uses
    replacement only when a source's requested quota exceeds its row count.
    """
    if strategy not in ORGANISM_SAMPLING_STRATEGIES:
        raise ValueError(f"Unknown organism sampling strategy: {strategy}")
    if strategy == "round_robin_batches":
        raise ValueError("round_robin_batches is constructed at batch level")
    if sample_rows < 1:
        raise ValueError("sample_rows must be positive")
    values = np.asarray([str(value) for value in organisms], dtype=object)
    missing = [name for name in ORGANISM_ORDER if not np.any(values == name)]
    if missing:
        raise ValueError(f"Training data lacks source organisms: {missing}")
    rng = np.random.default_rng(seed)
    counts = np.asarray([int(np.sum(values == name)) for name in ORGANISM_ORDER], dtype=float)
    if strategy == "natural":
        return rng.choice(len(values), size=sample_rows, replace=True).tolist()
    if strategy == "inverse_frequency":
        count_map = dict(zip(ORGANISM_ORDER, counts))
        probabilities = np.asarray([1.0 / count_map[value] for value in values], dtype=float)
        probabilities /= probabilities.sum()
        return rng.choice(len(values), size=sample_rows, replace=True, p=probabilities).tolist()
    if strategy in ORGANISM_TEMPERATURES:
        alpha = ORGANISM_TEMPERATURES[strategy]
        source_probabilities = counts ** alpha
        source_probabilities /= source_probabilities.sum()
        count_map = dict(zip(ORGANISM_ORDER, counts))
        source_map = dict(zip(ORGANISM_ORDER, source_probabilities))
        probabilities = np.asarray(
            [source_map[value] / count_map[value] for value in values], dtype=float
        )
        probabilities /= probabilities.sum()
        return rng.choice(len(values), size=sample_rows, replace=True, p=probabilities).tolist()

    if strategy in ORGANISM_RATIO_STRATEGIES:
        source_probabilities = np.asarray(ORGANISM_RATIO_STRATEGIES[strategy], dtype=float)
        source_probabilities /= source_probabilities.sum()
    else:
        progress = min(1.0, max(0.0, float(progress)))
        natural = counts / counts.sum()
        uniform = np.full(len(ORGANISM_ORDER), 1.0 / len(ORGANISM_ORDER))
        ce4 = np.asarray((4, 1, 1), dtype=float) / 6.0
        endpoints = {
            "curriculum_natural_to_uniform": (natural, uniform),
            "curriculum_uniform_to_natural": (uniform, natural),
            "curriculum_ce4_to_uniform": (ce4, uniform),
            "curriculum_uniform_to_ce4": (uniform, ce4),
        }
        if strategy not in endpoints:
            raise ValueError(f"Strategy does not define source probabilities: {strategy}")
        start, end = endpoints[strategy]
        source_probabilities = (1.0 - progress) * start + progress * end

    raw_quotas = source_probabilities * sample_rows
    quotas = np.floor(raw_quotas).astype(int)
    for position in np.argsort(-(raw_quotas - quotas))[: sample_rows - int(quotas.sum())]:
        quotas[position] += 1
    sampled: List[int] = []
    for name, quota in zip(ORGANISM_ORDER, quotas):
        candidates = np.flatnonzero(values == name)
        sampled.extend(
            rng.choice(candidates, size=int(quota), replace=int(quota) > len(candidates)).tolist()
        )
    rng.shuffle(sampled)
    return sampled


def round_robin_organism_batches(
    encoded: Sequence[torch.Tensor],
    organisms: Sequence[str],
    batch_size: int,
    *,
    steps: int,
    seed: int,
) -> Iterator[List[int]]:
    """Yield exactly ``steps`` homogeneous batches, alternating organisms."""
    if steps < 1:
        raise ValueError("steps must be positive")
    values = np.asarray([str(value) for value in organisms], dtype=object)
    rng = random.Random(seed)
    queues: Dict[str, List[List[int]]] = {}
    for offset, name in enumerate(ORGANISM_ORDER):
        source_indices = np.flatnonzero(values == name).tolist()
        if not source_indices:
            raise ValueError(f"Training data lacks source organism: {name}")
        queues[name] = list(
            length_grouped_batches(
                encoded, batch_size, shuffle=True, seed=seed + offset, indices=source_indices
            )
        )
        rng.shuffle(queues[name])
    positions = {name: 0 for name in ORGANISM_ORDER}
    for step in range(steps):
        name = ORGANISM_ORDER[step % len(ORGANISM_ORDER)]
        batches = queues[name]
        position = positions[name]
        if position and position % len(batches) == 0:
            rng.shuffle(batches)
        yield batches[position % len(batches)]
        positions[name] += 1


def distributed_batch_shard(
    indices: Sequence[int], rank: int, world_size: int,
) -> Tuple[List[int], np.ndarray]:
    """Shard one global batch while keeping every DDP rank in every step.

    Short tail batches are padded by repeating rows. ``valid`` marks those
    repeats so they contribute neither loss nor reported training metrics.
    """
    indices = list(indices)
    if not indices:
        raise ValueError("Cannot shard an empty batch")
    if world_size <= 1:
        return indices, np.ones(len(indices), dtype=bool)
    padded_size = math.ceil(len(indices) / world_size) * world_size
    padded = indices + [indices[offset % len(indices)] for offset in range(padded_size - len(indices))]
    positions = list(range(rank, padded_size, world_size))
    return [padded[position] for position in positions], np.asarray(
        [position < len(indices) for position in positions], dtype=bool
    )


def reduce_training_totals(values: Sequence[float], device: torch.device) -> List[float]:
    """Sum scalar epoch totals across DDP ranks."""
    totals = torch.tensor(list(values), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.cpu().tolist()


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    delta = predictions - targets
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - np.sum(delta * delta) / denominator) if denominator else float("nan")
    pearson = float(np.corrcoef(targets, predictions)[0, 1]) if len(targets) > 1 else float("nan")
    if len(targets) > 1:
        target_ranks = pd.Series(targets).rank(method="average").to_numpy()
        prediction_ranks = pd.Series(predictions).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
    else:
        spearman = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}


def heavy_dimethyl_masks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return overlapping Heavy-Dimethyl subsets used for loss tracking.

    ``heavy_nterm_dimethyl`` intentionally includes peptides carrying both an
    N-terminal and a lysine heavy dimethyl modification: the question is
    whether the N-terminal heavy label is present, not whether it is alone.
    """
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False)
    lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False)
    return {
        "heavy_dimethyl": (nterm | lysine).to_numpy(),
        "heavy_nterm_dimethyl": nterm.to_numpy(),
    }


def objective_loss_values(
    residuals: np.ndarray, loss_name: str, huber_delta: float,
) -> np.ndarray:
    """Calculate the unreduced DIA-NN objective for reporting subsets."""
    absolute = np.abs(np.asarray(residuals, dtype=np.float32))
    if loss_name == "mae":
        return absolute
    if loss_name == "mse":
        return absolute * absolute
    if loss_name == "huber":
        return np.where(
            absolute < huber_delta,
            0.5 * absolute * absolute,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    raise ValueError(f"Unknown loss: {loss_name}")


def heavy_subset_loss_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    loss_name: str,
    huber_delta: float,
    prefix: str,
) -> Dict[str, float]:
    """Report unweighted objective values for the two tracked heavy subsets."""
    targets = frame["rt_norm"].to_numpy(dtype=np.float32)
    residuals = np.asarray(predictions, dtype=np.float32).reshape(-1) - targets
    losses = objective_loss_values(residuals, loss_name, huber_delta)
    result: Dict[str, float] = {}
    for name, mask in heavy_dimethyl_masks(frame).items():
        result[f"{prefix}{name}_rows"] = int(mask.sum())
        result[f"{prefix}{name}_loss"] = float(np.mean(losses[mask])) if mask.any() else float("nan")
    return result


def heavy_stratified_regression_metrics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """Calculate unweighted validation metrics for exact heavy strata."""
    labels = heavy_dimethyl_row_strata(frame).reset_index(drop=True)
    targets = frame["rt_norm"].to_numpy(dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if len(labels) != len(predictions):
        raise ValueError("Validation frame and prediction length differ")
    masks = {
        "normal": labels.eq("normal").to_numpy(),
        "heavy_nterm": labels.eq("heavy_nterm").to_numpy(),
        "heavy_lysine": labels.eq("heavy_lysine").to_numpy(),
        "heavy_both": labels.eq("heavy_both").to_numpy(),
        "heavy_dimethyl": labels.ne("normal").to_numpy(),
        "heavy_nterm_dimethyl": heavy_dimethyl_masks(frame)["heavy_nterm_dimethyl"],
    }
    result: Dict[str, float] = {}
    for stratum, mask in masks.items():
        result[f"{stratum}_rows"] = int(mask.sum())
        if mask.any():
            for metric, value in regression_metrics(targets[mask], predictions[mask]).items():
                result[f"{stratum}_{metric}"] = value
    return result


def load_model_with_custom_embeddings(
    model_path: Path,
    device: torch.device,
    embedding_initializations: Dict[int, int],
) -> Tuple[torch.jit.ScriptModule, Dict[int, torch.Tensor]]:
    """Load DIA-NN and initialize custom token rows from related known tokens."""
    model = load_torchscript_model(model_path, device)
    if not hasattr(model, "embedding") or not hasattr(model.embedding, "weight"):
        raise RuntimeError("DIA-NN model does not expose the expected embedding.weight parameter")
    embedding_rows = model.embedding.weight.shape[0]
    initialized_rows: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for new_id, source_id in embedding_initializations.items():
            if max(new_id, source_id) >= embedding_rows:
                raise RuntimeError(
                    f"Token IDs {new_id}/{source_id} exceed model embedding size {embedding_rows}"
                )
            model.embedding.weight[new_id].copy_(model.embedding.weight[source_id])
            initialized_rows[new_id] = model.embedding.weight[new_id].detach().cpu().clone()
    return model, initialized_rows


def configure_optimizer(
    model: torch.jit.ScriptModule,
    strategy: str,
    embedding_initializations: Dict[int, int],
    learning_rate: float,
    backbone_learning_rate: float | None,
    weight_decay: float,
    embedding_learning_rate: float | None = None,
) -> Tuple[torch.optim.Optimizer, List[torch.Tensor], List[float]]:
    """Apply a named layer policy and build explicit optimizer groups."""
    strategy = "head" if strategy == "output" else strategy
    strategies = {
        "all",
        "embeddings",
        "head",
        "rt_head",
        "cnn_head",
        "gru_head",
        "discriminative",
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown training strategy: {strategy}")
    for parameter in model.parameters():
        parameter.requires_grad_(strategy == "all")

    embedding = model.embedding.weight
    embedding_lr = embedding_learning_rate or learning_rate
    if strategy != "all":
        # Only custom rows may move; all original token embeddings remain fixed.
        embedding.requires_grad_(True)
        gradient_mask = torch.zeros_like(embedding)
        for token_id in embedding_initializations:
            gradient_mask[token_id] = 1
        embedding.register_hook(lambda gradient: gradient * gradient_mask)

    if strategy == "embeddings":
        groups = [
            {"params": [embedding], "lr": embedding_lr, "name": "custom_embeddings"},
        ]
    elif strategy == "head":
        for parameter in model.out.parameters():
            parameter.requires_grad_(True)
        groups = [
            {"params": [embedding], "lr": embedding_lr, "name": "custom_embeddings"},
            {"params": list(model.out.parameters()), "lr": learning_rate, "name": "output"},
        ]
    elif strategy in {"rt_head", "cnn_head", "gru_head"}:
        head_parameters = list(model.prepare.parameters()) + list(model.out.parameters())
        for parameter in head_parameters:
            parameter.requires_grad_(True)
        feature_parameters: List[torch.Tensor] = []
        if strategy == "cnn_head":
            feature_parameters = list(model.cnn.parameters()) + list(model.cnn2.parameters())
        elif strategy == "gru_head":
            feature_parameters = list(model.rnn.parameters())
        for parameter in feature_parameters:
            parameter.requires_grad_(True)
        groups = [
            {
                "params": feature_parameters,
                "lr": backbone_learning_rate or learning_rate / 10.0,
                "name": strategy.removesuffix("_head"),
            },
            {"params": head_parameters, "lr": learning_rate, "name": "head"},
            {"params": [embedding], "lr": embedding_lr, "name": "custom_embeddings"},
        ]
    elif strategy == "discriminative":
        backbone_lr = backbone_learning_rate or learning_rate / 10.0
        head_parameters = list(model.prepare.parameters()) + list(model.out.parameters())
        excluded = {id(embedding), *(id(parameter) for parameter in head_parameters)}
        backbone_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in excluded
        ]
        for parameter in (*backbone_parameters, *head_parameters):
            parameter.requires_grad_(True)
        groups = [
            {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"},
            {"params": head_parameters, "lr": learning_rate, "name": "head"},
            {"params": [embedding], "lr": embedding_lr, "name": "custom_embeddings"},
        ]
    else:
        groups = [
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": learning_rate,
                "name": "all",
            }
        ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)
    trainable_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return optimizer, trainable_parameters, base_lrs


def evaluate(
    model: torch.nn.Module,
    encoded: Sequence[torch.Tensor],
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    flatten_rnn_parameters(base_model)
    prediction_array = np.empty(len(encoded), dtype=np.float32)
    with torch.no_grad():
        for indices in length_grouped_batches(encoded, batch_size, shuffle=False, seed=0):
            inputs = torch.stack([encoded[index] for index in indices]).to(device)
            output = model(inputs).reshape(-1)
            prediction_array[np.asarray(indices)] = output.cpu().numpy()
    target_array = targets.cpu().numpy()
    return regression_metrics(target_array, prediction_array), prediction_array


def train_rt_model(
    model_path: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    encoder: TokenEncoder,
    output_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    warmup_epochs: int,
    batch_size: int,
    learning_rate: float,
    backbone_learning_rate: float | None,
    weight_decay: float,
    seed: int,
    default_charge: int,
    trainable: str,
    embedding_initializations: Dict[int, int],
    loss_name: str = "mae",
    huber_delta: float = 0.03,
    normal_weight: float = 1.0,
    heavy_nterm_weight: float = 1.0,
    heavy_lysine_weight: float = 1.0,
    heavy_both_weight: float | None = None,
    embedding_learning_rate: float | None = None,
    organism_sampling: str | None = None,
    steps_per_epoch: int | None = None,
    save_heavy_checkpoint_grid: bool = False,
    epoch_callback: Callable[[Dict[str, float]], None] | None = None,
) -> Dict[str, object]:
    verify_torchscript_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = distributed_rank()
    world_size = distributed_world_size()
    set_seed(seed + rank)
    model, initialized_rows = load_model_with_custom_embeddings(
        model_path, device, embedding_initializations
    )
    optimizer, parameters, base_lrs = configure_optimizer(
        model,
        trainable,
        embedding_initializations,
        learning_rate,
        backbone_learning_rate,
        weight_decay,
        embedding_learning_rate,
    )
    if not parameters:
        raise RuntimeError("No trainable model parameters selected")
    training_model: torch.nn.Module = model
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("WORLD_SIZE is greater than one, but torch.distributed is not initialized")
        training_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )

    train_encoded, train_targets = prepare_examples(train_frame, encoder, default_charge)
    val_encoded, val_targets = prepare_examples(validation_frame, encoder, default_charge)
    train_weights, weight_category_counts = training_sample_weights(
        train_frame,
        normal_weight=normal_weight,
        heavy_nterm_weight=heavy_nterm_weight,
        heavy_lysine_weight=heavy_lysine_weight,
        heavy_both_weight=heavy_both_weight,
    )
    train_heavy_masks = heavy_dimethyl_masks(train_frame)
    pretrained_metrics, pretrained_predictions = evaluate(
        training_model, val_encoded, val_targets, batch_size, device
    )
    pretrained_metrics.update(
        heavy_stratified_regression_metrics(validation_frame, pretrained_predictions)
    )

    loss_fn = build_loss_function(loss_name, huber_delta)
    history: List[Dict[str, float]] = []
    best_loss = math.inf
    best_epoch = 0
    best_path = output_dir / "rt.d0.finetuned.pt"
    checkpoint_paths = {"100_0": best_path}
    if save_heavy_checkpoint_grid:
        checkpoint_paths.update({
            label: output_dir / "checkpoints" / f"selection_{label}" / "rt.d0.finetuned.pt"
            for label in HEAVY_CHECKPOINT_SELECTIONS if label != "100_0"
        })
    checkpoint_best_scores = {label: math.inf for label in checkpoint_paths}
    checkpoint_selection = {
        label: {
            "label": label,
            "overall_weight": HEAVY_CHECKPOINT_SELECTIONS[label][0],
            "heavy_macro_weight": HEAVY_CHECKPOINT_SELECTIONS[label][1],
            "checkpoint": str(path),
            "best_epoch": None,
            "score": None,
            "validation_metrics": None,
        }
        for label, path in checkpoint_paths.items()
    }
    started = time.monotonic()

    if organism_sampling is not None:
        if organism_sampling not in ORGANISM_SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown organism sampling strategy: {organism_sampling}")
        if steps_per_epoch is None or steps_per_epoch < 1:
            raise ValueError("Organism sampling requires a positive steps_per_epoch")
        if "source_organism" not in train_frame.columns:
            raise ValueError("Organism sampling requires a source_organism column")

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            scale = epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs - 1) / max(1, epochs - warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale
        current_lr = max(float(group["lr"]) for group in optimizer.param_groups)

        training_model.train()
        flatten_rnn_parameters(model)
        total_loss = 0.0
        total_weight = 0.0
        total_absolute_error = 0.0
        total_rows = 0
        tracked_loss_totals = {name: 0.0 for name in train_heavy_masks}
        tracked_loss_rows = {name: 0 for name in train_heavy_masks}
        source_values = (
            train_frame["source_organism"].astype(str).to_numpy()
            if "source_organism" in train_frame.columns else None
        )
        tracked_source_rows = {name: 0 for name in ORGANISM_ORDER}
        if organism_sampling == "round_robin_batches":
            epoch_batches = round_robin_organism_batches(
                train_encoded,
                train_frame["source_organism"].astype(str).tolist(),
                batch_size,
                steps=int(steps_per_epoch),
                seed=seed + epoch,
            )
        elif organism_sampling is not None:
            epoch_indices = sample_organism_indices(
                train_frame["source_organism"].astype(str).tolist(),
                organism_sampling,
                int(steps_per_epoch) * batch_size,
                seed + epoch,
                progress=(epoch - 1) / max(1, epochs - 1),
            )
            epoch_batches = length_grouped_batches(
                train_encoded,
                batch_size,
                shuffle=True,
                seed=seed + epoch,
                indices=epoch_indices,
            )
        elif steps_per_epoch is not None:
            # A fixed update budget makes oversampling-dose comparisons fair:
            # every profile receives the same number of optimizer steps. Rows
            # are sampled from the already oversampled frame with replacement,
            # so smaller profiles can still fill the common budget.
            rng = np.random.default_rng(seed + epoch)
            epoch_indices = rng.choice(
                len(train_encoded),
                size=int(steps_per_epoch) * batch_size,
                replace=True,
            ).tolist()
            epoch_batches = length_grouped_batches(
                train_encoded,
                batch_size,
                shuffle=True,
                seed=seed + epoch,
                indices=epoch_indices,
            )
        else:
            epoch_batches = length_grouped_batches(
                train_encoded, batch_size, shuffle=True, seed=seed + epoch
            )
        for step, global_indices in enumerate(epoch_batches, start=1):
            if steps_per_epoch is not None and step > steps_per_epoch:
                break
            indices, valid_rows = distributed_batch_shard(global_indices, rank, world_size)
            inputs = torch.stack([train_encoded[index] for index in indices]).to(device)
            targets = train_targets[indices].to(device)
            sample_weights = train_weights[indices].to(device)
            valid_tensor = torch.as_tensor(valid_rows, dtype=sample_weights.dtype, device=device)
            effective_weights = sample_weights * valid_tensor
            optimizer.zero_grad(set_to_none=True)
            predictions = training_model(inputs).reshape(-1)
            per_sample_loss = loss_fn(predictions, targets)
            weighted_loss = per_sample_loss * effective_weights
            global_weight = effective_weights.sum().detach()
            if world_size > 1:
                dist.all_reduce(global_weight, op=dist.ReduceOp.SUM)
            loss = weighted_loss.sum() * world_size / global_weight.clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            total_loss += weighted_loss.detach().sum().item()
            total_weight += effective_weights.sum().item()
            total_absolute_error += (
                torch.abs(predictions.detach() - targets) * valid_tensor
            ).sum().item()
            total_rows += int(valid_rows.sum())
            batch_losses = per_sample_loss.detach().cpu().numpy().reshape(-1)
            batch_indices = np.asarray(indices)
            if source_values is not None:
                for name in ORGANISM_ORDER:
                    tracked_source_rows[name] += int(
                        ((source_values[batch_indices] == name) & valid_rows).sum()
                    )
            for name, mask in train_heavy_masks.items():
                selected = mask[batch_indices] & valid_rows
                if selected.any():
                    tracked_loss_totals[name] += float(batch_losses[selected].sum())
                    tracked_loss_rows[name] += int(selected.sum())

        ordered_subsets = list(train_heavy_masks)
        reduced = reduce_training_totals(
            [
                total_loss,
                total_weight,
                total_absolute_error,
                total_rows,
                *(tracked_loss_totals[name] for name in ordered_subsets),
                *(tracked_loss_rows[name] for name in ordered_subsets),
                *(tracked_source_rows[name] for name in ORGANISM_ORDER),
            ],
            device,
        )
        total_loss, total_weight, total_absolute_error, total_rows = reduced[:4]
        offset = 4
        tracked_loss_totals = dict(zip(ordered_subsets, reduced[offset:offset + len(ordered_subsets)]))
        offset += len(ordered_subsets)
        tracked_loss_rows = dict(zip(ordered_subsets, reduced[offset:offset + len(ordered_subsets)]))
        offset += len(ordered_subsets)
        tracked_source_rows = dict(zip(ORGANISM_ORDER, reduced[offset:offset + len(ORGANISM_ORDER)]))

        validation_metrics, validation_predictions = evaluate(
            training_model, val_encoded, val_targets, batch_size, device
        )
        validation_metrics.update(
            heavy_stratified_regression_metrics(validation_frame, validation_predictions)
        )
        validation_metrics.update(
            heavy_subset_loss_metrics(
                validation_frame,
                validation_predictions,
                loss_name=loss_name,
                huber_delta=huber_delta,
                prefix="",
            )
        )
        if save_heavy_checkpoint_grid:
            checkpoint_scores = heavy_checkpoint_selection_scores(validation_metrics)
            validation_metrics["heavy_macro_mae"] = sum(
                float(validation_metrics[name])
                for name in ("heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae")
            ) / 3.0
        else:
            # Preserve the historical training behavior exactly for existing
            # callers and datasets that do not contain every Heavy stratum.
            checkpoint_scores = {"100_0": float(validation_metrics["mae"])}
        record = {
            "epoch": float(epoch),
            "learning_rate": current_lr,
            "train_loss": total_loss / max(1.0, total_weight),
            "train_mae": total_absolute_error / max(1, total_rows),
            "validation_loss": validation_metrics["mae"],
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"validation_checkpoint_score_{label}": score for label, score in checkpoint_scores.items()},
        }
        for name in train_heavy_masks:
            record[f"train_{name}_loss"] = (
                tracked_loss_totals[name] / tracked_loss_rows[name]
                if tracked_loss_rows[name] else float("nan")
            )
        if source_values is not None:
            for name in ORGANISM_ORDER:
                record[f"train_source_{name}_rows"] = int(tracked_source_rows[name])
        for group in optimizer.param_groups:
            record[f"lr_{group['name']}"] = float(group["lr"])
        history.append(record)
        if epoch_callback is not None and is_primary_process():
            epoch_callback(record)
        objective_text = f"train_{loss_name}={record['train_loss']:.6f}"
        if loss_name != "mae" or any(
            weight != 1.0
            for weight in (normal_weight, heavy_nterm_weight, heavy_lysine_weight, heavy_both_weight or 1.0)
        ):
            objective_text += f" train_mae={record['train_mae']:.6f}"
        if is_primary_process():
            print(
                f"epoch={epoch}/{epochs} lr={current_lr:.3g} {objective_text} "
                f"val_mae={record['validation_loss']:.6f} pearson={validation_metrics['pearson']:.4f}",
                flush=True,
            )
        for label, path in checkpoint_paths.items():
            score = checkpoint_scores[label]
            if score >= checkpoint_best_scores[label]:
                continue
            checkpoint_best_scores[label] = score
            snapshot = {"overall_mae": float(validation_metrics["mae"])}
            for metric_name in (
                "heavy_nterm_mae", "heavy_lysine_mae", "heavy_both_mae", "heavy_macro_mae",
            ):
                if metric_name in validation_metrics:
                    snapshot[metric_name] = float(validation_metrics[metric_name])
            checkpoint_selection[label].update({
                "best_epoch": epoch,
                "score": score,
                "validation_metrics": snapshot,
            })
            if is_primary_process():
                save_portable_torchscript(model, path)
        best_loss = checkpoint_best_scores["100_0"]
        best_epoch = int(checkpoint_selection["100_0"]["best_epoch"] or 0)

    last_path = output_dir / "rt.d0.last.pt"
    if is_primary_process():
        save_portable_torchscript(model, last_path)
        (output_dir / "checkpoint_selection.json").write_text(
            json.dumps({
                "metric_scale": "rt_norm",
                "heavy_macro_definition": "mean(exact heavy_nterm, heavy_lysine, heavy_both validation MAE)",
                "saved_grid": save_heavy_checkpoint_grid,
                "selections": checkpoint_selection,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()
    best_model = load_torchscript_model(best_path, device)
    final_metrics, final_predictions = evaluate(best_model, val_encoded, val_targets, batch_size, device)
    final_metrics.update(
        heavy_stratified_regression_metrics(validation_frame, final_predictions)
    )
    history_path = output_dir / "training_history.csv"
    if is_primary_process():
        pd.DataFrame(history).to_csv(history_path, index=False)
    predictions = validation_frame.copy()
    predictions["rt_pred_pretrained"] = np.nan
    pretrained_model, _ = load_model_with_custom_embeddings(
        model_path, device, embedding_initializations
    )
    _, pre_predictions = evaluate(
        pretrained_model,
        val_encoded,
        val_targets,
        batch_size,
        device,
    )
    predictions["rt_pred_pretrained"] = pre_predictions
    predictions["rt_pred_finetuned"] = final_predictions
    if is_primary_process():
        predictions.to_csv(output_dir / "validation_predictions.tsv", sep="\t", index=False)
    embedding_updates = {
        str(token_id): float(
            torch.linalg.vector_norm(
                best_model.embedding.weight[token_id].detach().cpu() - initial_row
            ).item()
        )
        for token_id, initial_row in initialized_rows.items()
    }
    optimizer_parameter_elements = sum(parameter.numel() for parameter in parameters)
    effective_trainable_parameters = optimizer_parameter_elements
    normalized_strategy = "head" if trainable == "output" else trainable
    if normalized_strategy != "all":
        effective_trainable_parameters -= model.embedding.weight.numel()
        effective_trainable_parameters += (
            len(embedding_initializations) * model.embedding.weight.shape[1]
        )
    return {
        "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "distributed_world_size": world_size,
        "pretrained_metrics": pretrained_metrics,
        "finetuned_metrics": final_metrics,
        "trainable_parameters": effective_trainable_parameters,
        "optimizer_parameter_elements": optimizer_parameter_elements,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_embedding_rows": (
            sorted(embedding_initializations) if normalized_strategy != "all" else "all"
        ),
        "trainable_parameter_tensors": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "custom_embedding_l2_updates": embedding_updates,
        "strategy": normalized_strategy,
        "loss": {
            "name": loss_name,
            "huber_delta": huber_delta if loss_name == "huber" else None,
            "normal_weight": normal_weight,
            "heavy_nterm_weight": heavy_nterm_weight,
            "heavy_lysine_weight": heavy_lysine_weight,
            "heavy_both_weight": (
                heavy_both_weight
                if heavy_both_weight is not None
                else heavy_nterm_weight * heavy_lysine_weight
            ),
            "training_category_counts": weight_category_counts,
        },
        "organism_sampling": {
            "strategy": organism_sampling or (
                "uniform_from_oversampled_training_frame" if steps_per_epoch is not None else "full_oversampled_epoch"
            ),
            "steps_per_epoch": steps_per_epoch,
            "examples_per_epoch_budget": (
                steps_per_epoch * batch_size if steps_per_epoch is not None else None
            ),
            "source_rows": (
                train_frame["source_organism"].astype(str).value_counts().sort_index().to_dict()
                if "source_organism" in train_frame.columns else None
            ),
        },
        "optimizer_groups": {
            str(group["name"]): base_lr
            for group, base_lr in zip(optimizer.param_groups, base_lrs)
        },
        "best_model": str(best_path),
        "checkpoint_selection_grid": checkpoint_selection,
        "checkpoint_selection_manifest": str(output_dir / "checkpoint_selection.json"),
        "last_model": str(last_path),
        "history": str(history_path),
    }


def write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
