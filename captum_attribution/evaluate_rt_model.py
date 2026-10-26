"""CLI for a self-contained DIA-NN RT model × dataset evaluation."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import torch

from .core import (embedding_adapter_parity_check, load_runtime, manifest,
                   metric_dict, predict, prepare_dataset, stratified_metrics)
from .plots import attribution_plots, prediction_plots


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    required = ["evaluation_name", "model", "dataset", "output"]
    missing = [key for key in required if key not in config]
    if missing: raise ValueError(f"Config lacks required keys: {missing}")
    for section, names in {"model": ["name", "path", "token_dictionary"], "dataset": ["name", "path", "sequence_column", "observed_rt_column"], "output": ["root"]}.items():
        absent=[name for name in names if name not in config[section]]
        if absent: raise ValueError(f"Config {section} lacks: {absent}")
    return config


def _run_dir(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root"]) / f"{config['model']['name']}__{config['dataset']['name']}__{config['evaluation_name']}"


def _write_yaml(value: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _reference_prediction_validation(result: pd.DataFrame, spec: dict[str, Any], output: Path) -> None:
    """Optional explicit parity gate against established DIA-NN predictions."""
    column = spec.get("reference_prediction_column")
    if not column or column not in result:
        return
    reference = pd.to_numeric(result[column], errors="coerce")
    valid = reference.notna()
    if not valid.any():
        raise ValueError(f"No finite values in reference_prediction_column={column!r}")
    difference = result.loc[valid, "predicted_rt"] - reference.loc[valid]
    report = {"reference_prediction_column": column, "n_compared": int(valid.sum()), "max_absolute_difference": float(difference.abs().max()), "mean_absolute_difference": float(difference.abs().mean()), "tolerance": float(spec.get("reference_prediction_tolerance", 1e-5))}
    (output / "inference_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["max_absolute_difference"] > report["tolerance"]:
        raise RuntimeError(f"Inference parity check failed: max difference {report['max_absolute_difference']:.6g} exceeds {report['tolerance']:.6g}; inspect inference_validation.json")


def _baseline(input_ids, tokens: dict[str, int], method: str):
    """Return explicit LIG references without assuming token 0 is semantic padding."""
    import torch
    if method == "zeros": return torch.zeros_like(input_ids)
    if method == "pad":
        pad = 0
        return torch.full_like(input_ids, pad)
    if method == "boundary_only":
        # Preserve charge and DIA-NN boundary markers; interior is token 0.
        result = torch.zeros_like(input_ids)
        result[:, 0] = input_ids[:, 0]
        result[:, 1] = tokens["<"]; result[:, -1] = tokens[">"]
        return result
    raise ValueError("captum.baseline must be one of: boundary_only, pad, zeros")


def captum_attributions(model, encoder, frame: pd.DataFrame, predictions: np.ndarray, config: dict[str, Any], output: Path, device, cache_path: Path | None = None, progress_label: str = "Captum", progress_position: int = 0) -> pd.DataFrame:
    """Integrated Gradients over continuous DIA-NN token embeddings.

    Captum cannot install forward hooks on a TorchScript ``ScriptModule``.
    An exact post-embedding adapter is therefore used as the forward function;
    integer token IDs are used only for lookup, never differentiated.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("Captum is required for --captum; install environment/captum_eval.yml") from exc
    import torch
    opts = config.get("captum", {}); n_steps=int(opts.get("n_steps", 64)); baseline_name=opts.get("baseline", "boundary_only")
    tokens = dict(encoder.tokens)
    integrated_gradients = None
    rows=[]; cache_path=cache_path or output/"attributions.parquet"
    completed=set(pd.read_parquet(cache_path).peptide_id) if cache_path.exists() else set()
    peptide_batch_size = int(opts.get("peptide_batch_size", 32))
    checkpoint_every = int(opts.get("checkpoint_every", 100))
    if peptide_batch_size < 1 or checkpoint_every < 1:
        raise ValueError("captum.peptide_batch_size and captum.checkpoint_every must be positive")

    # The original implementation submitted one peptide at a time.  That is
    # mathematically valid but leaves the GPUs nearly idle.  Grouping only by
    # *identical encoded length* keeps DIA-NN's no-padding representation while
    # allowing Captum to evaluate many independent peptides in one call.
    pending: dict[int, list[tuple[Any, float, torch.Tensor]]] = {}
    for row, pred in zip(frame.itertuples(index=False), predictions):
        if row.peptide_id in completed:
            continue
        encoded = encoder.encode(row.predictor_sequence, charge=int(row.charge))
        pending.setdefault(len(encoded), []).append((row, float(pred), encoded))

    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=sum(len(group) for group in pending.values()), desc=progress_label, unit="peptide", dynamic_ncols=True, position=progress_position)
    except ImportError:  # Keep the evaluator runnable in minimal environments.
        progress = None

    inverse = {value: key for key, value in tokens.items()}
    pending_peptides = 0
    def flush() -> None:
        nonlocal rows, pending_peptides
        if not rows:
            return
        addition = pd.DataFrame(rows)
        if cache_path.exists():
            addition = pd.concat([pd.read_parquet(cache_path), addition], ignore_index=True)
        addition.to_parquet(cache_path, index=False)
        rows = []
        pending_peptides = 0

    for length in sorted(pending):
        group = pending[length]
        for start in range(0, len(group), peptide_batch_size):
            batch = group[start:start + peptide_batch_size]
            batch_rows = [item[0] for item in batch]
            batch_predictions = [item[1] for item in batch]
            input_ids = torch.stack([item[2] for item in batch]).to(device)
            reference_ids = _baseline(input_ids, tokens, baseline_name)
            # This fails closed before Captum accepts any values.  It checks
            # the whole first batch, not merely one representative peptide.
            if integrated_gradients is None:
                integrated_gradients = IntegratedGradients(embedding_adapter_parity_check(model, input_ids))
            input_embeddings = model.embedding(input_ids[:, 1:]).detach()
            reference_embeddings = model.embedding(reference_ids[:, 1:]).detach()
            attributes, delta = integrated_gradients.attribute(
                input_embeddings, baselines=reference_embeddings, n_steps=n_steps,
                return_convergence_delta=True, internal_batch_size=opts.get("internal_batch_size"),
            )
            # ``input_ids`` includes charge at column 0; embeddings do not.
            reduced = attributes.sum(dim=-1).detach().cpu().numpy()
            token_id_batches = input_ids.detach().cpu().tolist()
            delta_values = delta.reshape(-1).detach().cpu().tolist()
            for row, pred, token_ids_with_charge, scores, convergence_delta in zip(
                batch_rows, batch_predictions, token_id_batches, reduced, delta_values,
            ):
                token_ids = token_ids_with_charge[1:]
                if len(scores) != len(token_ids):
                    raise RuntimeError(f"Embedding attribution shape {len(scores)} does not match token sequence {len(token_ids)}")
                residue_tokens = [inverse.get(token_id, f"<unknown:{token_id}>") for token_id in token_ids]
                for position, (token_id, token, signed) in enumerate(zip(token_ids, residue_tokens, scores)):
                    # Boundary and modification tokens have no one-to-one chemical AA;
                    # retain them explicitly rather than inventing chemical identities.
                    residue = token[0] if len(token) == 1 and token.isalpha() else None
                    rows.append({"evaluation_id": config["evaluation_name"],"model_name":config["model"]["name"],"dataset_name":config["dataset"]["name"],"peptide_id":row.peptide_id,"modified_sequence":row.modified_sequence,"predictor_sequence":row.predictor_sequence,"position":position,"relative_position":position/(len(token_ids)-1) if len(token_ids)>1 else 0.,"residue":residue,"chemical_modification":row.n_terminal_modification if position==1 else None,"predictor_token_string":token,"token_id":token_id,"signed_attribution":float(signed),"absolute_attribution":float(abs(signed)),"captum_convergence_delta":float(convergence_delta),"predicted_rt":pred,"observed_rt":float(row.observed_rt),"signed_error":float(pred-row.observed_rt),"absolute_error":float(abs(pred-row.observed_rt)),"organism":getattr(row,"organism",None),"semi_specific":row.semi_specific,"peptide_length":int(row.peptide_length),"dimethyl_class":row.dimethyl_class})
            pending_peptides += len(batch)
            if progress is not None:
                progress.update(len(batch))
            if pending_peptides >= checkpoint_every:
                flush()
    if rows:
        flush()
    if progress is not None:
        progress.close()
    return pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()


def _configured_devices(config: dict[str, Any]) -> list[str]:
    devices=config.get("runtime",{}).get("devices")
    if not devices: return [config.get("runtime",{}).get("device","auto")]
    if not isinstance(devices,list) or not devices or not all(isinstance(x,str) for x in devices):
        raise ValueError("runtime.devices must be a non-empty list, e.g. [cuda:0, cuda:1, cuda:2]")
    return devices


def _distributed_predictions(config: dict[str, Any], encoded, devices: list[str]):
    """Partition peptide indices deterministically over independently loaded GPUs."""
    loaded=[load_runtime(config, device_override=device) for device in devices]
    predictions=np.empty(len(encoded),dtype=np.float32)
    def worker(worker_id: int):
        model,_,device,_=loaded[worker_id]
        indices=list(range(worker_id,len(encoded),len(loaded)))
        values=predict(model,[encoded[i] for i in indices],int(config.get("runtime",{}).get("batch_size",64)),device)
        return indices,values
    with ThreadPoolExecutor(max_workers=len(loaded)) as pool:
        for future in as_completed([pool.submit(worker,i) for i in range(len(loaded))]):
            indices,values=future.result(); predictions[indices]=values
    return predictions,loaded


def run(config: dict[str, Any], *, prediction_only: bool, do_captum: bool, plots_only: bool, max_peptides: int | None) -> Path:
    output=_run_dir(config); output.mkdir(parents=True,exist_ok=True); (output/"plots").mkdir(exist_ok=True); (output/"logs").mkdir(exist_ok=True)
    frame=prepare_dataset(config["dataset"])
    if max_peptides is not None: frame=frame.head(max_peptides).copy()
    if plots_only:
        predictions=pd.read_parquet(output/"predictions.parquet"); prediction_plots(predictions,output/"plots")
        if (output/"attributions.parquet").exists(): attribution_plots(pd.read_parquet(output/"attributions.parquet"),output/"plots")
        return output
    devices=_configured_devices(config)
    # One copy of each model is loaded per selected device. This is data
    # parallelism over peptides, not a fragile attempt to split one peptide
    # sequence across GPUs.
    if len(devices) > 1:
        loaded=None
        # Load once initially to obtain the dictionary and compatibility report;
        # distributed worker loading below supplies actual inference models.
        model,encoder,device,compatibility=load_runtime(config, device_override=devices[0])
    else:
        model,encoder,device,compatibility=load_runtime(config)
    encoded=encoder.encode_many(frame.predictor_sequence, charge=2) if frame.charge.nunique()==1 and frame.charge.iloc[0]==2 else [encoder.encode(p,int(c)) for p,c in zip(frame.predictor_sequence,frame.charge)]
    if len(devices) > 1:
        # Free the metadata/introspection copy before creating worker copies.
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        predictions,loaded=_distributed_predictions(config,encoded,devices)
    else:
        predictions=predict(model,encoded,int(config.get("runtime",{}).get("batch_size",64)),device); loaded=[(model,encoder,device,compatibility)]
    result=frame.copy(); result["token_ids"]=[x.tolist()[1:] for x in encoded]; result["predicted_rt"]=predictions; result["signed_residual"]=result.predicted_rt-result.observed_rt; result["absolute_error"]=result.signed_residual.abs(); result["squared_error"]=result.signed_residual**2
    _reference_prediction_validation(result, config["dataset"], output)
    result.to_parquet(output/"predictions.parquet",index=False); result.to_parquet(output/"peptide_summary.parquet",index=False)
    metrics=metric_dict(result.observed_rt,result.predicted_rt); (output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True),encoding="utf-8")
    strata=stratified_metrics(result,["organism","peptide_length","semi_specific","dimethyl_class","n_terminal_modification"]); strata.to_parquet(output/"stratified_metrics.parquet",index=False)
    _write_yaml(manifest(config,Path(config["model"]["path"]),Path(config["model"]["token_dictionary"]),Path(config["dataset"]["path"]),compatibility,len(result)),output/"manifest.yaml")
    prediction_plots(result,output/"plots")
    if do_captum and not prediction_only:
        if len(loaded) == 1:
            attrs=captum_attributions(model,encoder,result,predictions,config,output,device,progress_label=f"Captum {device}")
        else:
            final_cache=output/"attributions.parquet"
            expected_peptide_ids=set(result.peptide_id)
            completed_peptide_ids=(set(pd.read_parquet(final_cache).peptide_id)
                                  if final_cache.exists() else set())
            # A smoke test contains only five peptides. It may share this
            # directory with a later full run, but it must never make that
            # run look complete. Per-device partial caches resume the missing
            # peptide IDs safely.
            if expected_peptide_ids.issubset(completed_peptide_ids):
                attrs=pd.read_parquet(final_cache)
            else:
                def attr_worker(worker_id: int):
                    indices=list(range(worker_id,len(result),len(loaded)))
                    worker_model,worker_encoder,worker_device,_=loaded[worker_id]
                    cache=output/f"attributions_gpu{worker_id}.partial.parquet"
                    return captum_attributions(worker_model,worker_encoder,result.iloc[indices].copy(),predictions[indices],config,output,worker_device,cache,progress_label=f"Captum {worker_device}",progress_position=worker_id)
                with ThreadPoolExecutor(max_workers=len(loaded)) as pool:
                    pieces=[future.result() for future in as_completed([pool.submit(attr_worker,i) for i in range(len(loaded))])]
                attrs=pd.concat(pieces,ignore_index=True).drop_duplicates(["peptide_id","position"])
                attrs.to_parquet(final_cache,index=False)
        if not attrs.empty: attribution_plots(attrs,output/"plots")
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",required=True,type=Path); parser.add_argument("--prediction-only",action="store_true"); parser.add_argument("--captum",action="store_true"); parser.add_argument("--plots-only",action="store_true"); parser.add_argument("--smoke-test",action="store_true",help="evaluate at most five peptides")
    args=parser.parse_args(); out=run(_config(args.config),prediction_only=args.prediction_only,do_captum=args.captum,plots_only=args.plots_only,max_peptides=5 if args.smoke_test else None); print(out)
if __name__ == "__main__": main()
