"""Data preparation, inference, provenance, and Captum helpers.

This module deliberately reuses the project's DIA-NN loader and TokenEncoder.
It is kept independent of a particular Carafe split so that output tables can
be joined across model/dictionary/dataset combinations later.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from diann_finetune.data import heavy_dimethyl_row_strata
from diann_finetune.encoding import TokenEncoder, carafe_modified_sequence, read_tokens
from diann_finetune.training import load_torchscript_model, resolve_device, verify_torchscript_model


class DIA_NN_RTEmbeddingAdapter(torch.nn.Module):
    """Run DIA-NN's RT network from continuous token embeddings onward.

    DIA-NN distributes this model as a TorchScript ``ScriptModule``. Captum
    cannot place a layer hook on such a module, whereas ordinary Integrated
    Gradients works with continuous tensors. This adapter reproduces the
    scripted forward pass after ``embedding`` using the original CNN, GRU, and
    linear modules; it does not create learned parameter copies.
    """

    def __init__(self, scripted_model: torch.nn.Module):
        super().__init__()
        required = ("cnn", "cnn2", "rnn", "prepare", "out", "directions", "layers", "hidden_size")
        missing = [name for name in required if not hasattr(scripted_model, name)]
        if missing or not hasattr(scripted_model.rnn, "forward__0"):
            raise RuntimeError(
                "This TorchScript RT model does not match the supported DIA-NN "
                f"embedding-to-output architecture (missing: {missing or ['rnn.forward__0']})."
            )
        self.scripted_model = scripted_model

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return one RT score per peptide from ``[batch, tokens, embedding]``."""
        if embeddings.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, embedding] embeddings, got {tuple(embeddings.shape)}")
        sequence_length = embeddings.size(1)
        position = torch.arange(sequence_length, device=embeddings.device, dtype=embeddings.dtype).sqrt() - 2.5
        reverse_position = torch.flip(position, [0])
        batch_size = embeddings.size(0)
        position = position[None, :, None].expand(batch_size, -1, 1)
        reverse_position = reverse_position[None, :, None].expand(batch_size, -1, 1)
        values = torch.cat([embeddings, position, reverse_position], dim=2)
        values = torch.relu(self.scripted_model.cnn(values.permute(0, 2, 1)))
        values = torch.relu(self.scripted_model.cnn2(values)).permute(0, 2, 1)
        hidden = torch.zeros(
            int(self.scripted_model.directions) * int(self.scripted_model.layers),
            batch_size,
            int(self.scripted_model.hidden_size),
            device=values.device,
            dtype=values.dtype,
        )
        # GRU has overloads in TorchScript, so its callable method is named
        # forward__0. This is the exact overload used in model.forward.
        values, _ = self.scripted_model.rnn.forward__0(values, hidden)
        values = F.rrelu(self.scripted_model.prepare(values[:, :1, :]), training=False)
        return self.scripted_model.out(values).reshape(-1)


def embedding_adapter_parity_check(model: torch.nn.Module, input_ids: torch.Tensor) -> DIA_NN_RTEmbeddingAdapter:
    """Fail closed unless the Captum adapter exactly reproduces model inference."""
    adapter = DIA_NN_RTEmbeddingAdapter(model).eval()
    with torch.no_grad():
        expected = model(input_ids)
        actual = adapter(model.embedding(input_ids[:, 1:]))
    maximum_difference = (expected - actual).abs().max().item()
    if maximum_difference > 1e-6:
        raise RuntimeError(
            "Embedding adapter parity check failed: its output differs from the "
            f"DIA-NN TorchScript model by up to {maximum_difference:.3g}."
        )
    return adapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_peptide_id(sequence: str, predictor_sequence: str, charge: int, original_id: object = None) -> str:
    """Use supplied identifiers when available, otherwise a content-derived ID."""
    if original_id is not None and not pd.isna(original_id) and str(original_id).strip():
        return str(original_id)
    value = f"{sequence}|{predictor_sequence}|{charge}".encode("utf-8")
    return "pep_" + hashlib.sha256(value).hexdigest()[:20]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",")


def prepare_dataset(spec: dict[str, Any]) -> pd.DataFrame:
    """Normalise arbitrary labelled RT tables to the canonical prediction schema."""
    frame = read_table(Path(spec["path"])).copy()
    sequence_col = spec["sequence_column"]
    target_col = spec["observed_rt_column"]
    missing = [name for name in (sequence_col, target_col) if name not in frame]
    if missing:
        raise ValueError(f"Dataset lacks configured columns: {missing}")
    frame["stripped_sequence"] = frame[sequence_col].astype(str).str.strip().str.upper()
    frame["observed_rt"] = pd.to_numeric(frame[target_col], errors="coerce")
    if "predictor_sequence_column" in spec and spec["predictor_sequence_column"] in frame:
        frame["predictor_sequence"] = frame[spec["predictor_sequence_column"]].astype(str)
    elif {"mods_column", "mod_sites_column"}.issubset(spec):
        def convert(row: pd.Series) -> str:
            mods = "" if pd.isna(row[spec["mods_column"]]) else row[spec["mods_column"]]
            sites = "" if pd.isna(row[spec["mod_sites_column"]]) else row[spec["mod_sites_column"]]
            return carafe_modified_sequence(row[sequence_col], mods, sites)[0]
        frame["predictor_sequence"] = frame.apply(convert, axis=1)
    else:
        # For unmodified DIA-NN input a stripped sequence is a valid predictor sequence.
        frame["predictor_sequence"] = frame["stripped_sequence"]
    charge_col = spec.get("charge_column")
    frame["charge"] = (pd.to_numeric(frame[charge_col], errors="coerce") if charge_col in frame else spec.get("default_charge", 2))
    frame["charge"] = frame["charge"].fillna(spec.get("default_charge", 2)).astype(int)
    id_col = spec.get("peptide_id_column")
    supplied_id = frame[id_col] if id_col and id_col in frame else pd.Series([None] * len(frame), index=frame.index)
    frame["peptide_id"] = [stable_peptide_id(s, p, c, identifier) for s, p, c, identifier in zip(frame.stripped_sequence, frame.predictor_sequence, frame.charge, supplied_id)]
    frame = frame.dropna(subset=["observed_rt"]).reset_index(drop=True)
    if frame.peptide_id.duplicated().any():
        raise ValueError("peptide_id is not unique; configure a unique peptide_id_column")
    for column in spec.get("metadata_columns", []):
        if column not in frame:
            frame[column] = pd.NA
    frame["peptide_length"] = frame.stripped_sequence.str.len()
    frame["modified_sequence"] = frame.get(spec.get("modified_sequence_column", "modified_sequence"), frame.predictor_sequence)
    mods = frame.get(spec.get("mods_column", "mods"), pd.Series("", index=frame.index)).astype("string").fillna("").astype(str)
    frame["dimethyl_class"] = heavy_dimethyl_row_strata(pd.DataFrame({"mods": mods})).to_numpy()
    frame["n_terminal_modification"] = mods.str.extract(r"^([^;]+@N-term)", expand=False).fillna("none")
    frame["modified_lysines"] = mods.str.count(r"@K")
    frame["semi_specific"] = frame.get(spec.get("semi_specific_column", "semi_specific"), pd.Series(pd.NA, index=frame.index))
    organism_col = spec.get("organism_column", "organism")
    frame["organism"] = frame.get(organism_col, pd.Series(pd.NA, index=frame.index))
    return frame


def compatibility_check(model: torch.nn.Module, tokens: dict[str, int]) -> dict[str, Any]:
    if not hasattr(model, "embedding") or not hasattr(model.embedding, "weight"):
        raise RuntimeError("Model does not expose embedding.weight; cannot validate token compatibility")
    rows = int(model.embedding.weight.shape[0])
    maximum = max(tokens.values())
    if maximum >= rows:
        raise RuntimeError(f"Dictionary maximum token ID {maximum} exceeds model embedding rows {rows}")
    inverse: dict[int, list[str]] = {}
    for token, token_id in tokens.items():
        inverse.setdefault(token_id, []).append(token)
    return {"embedding_rows": rows, "embedding_dimension": int(model.embedding.weight.shape[1]), "dictionary_max_token_id": maximum, "token_zero": inverse.get(0, []), "vocabulary_entries": len(tokens)}


def predict(model: torch.nn.Module, encoded: list[torch.Tensor], batch_size: int, device: torch.device) -> np.ndarray:
    result = np.empty(len(encoded), dtype=np.float32)
    model.eval()
    # DIA-NN TorchScript models consume rectangular tensors.  Grouping by
    # token length preserves the established project's no-padding convention.
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(encoded): groups.setdefault(len(item), []).append(index)
    with torch.no_grad():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                current = indices[start:start + batch_size]
                batch = torch.stack([encoded[i] for i in current]).to(device)
                result[current] = model(batch).reshape(-1).detach().cpu().numpy()
    return result


def metric_dict(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    observed = np.asarray(list(observed), dtype=float); predicted = np.asarray(list(predicted), dtype=float)
    error = predicted - observed; absolute = np.abs(error)
    n = len(observed)
    pearson = float(np.corrcoef(observed, predicted)[0, 1]) if n > 1 and np.std(observed) and np.std(predicted) else float("nan")
    spearman = float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")) if n > 1 else float("nan")
    denom = np.sum((observed - observed.mean()) ** 2)
    return {"n": n, "mae": float(absolute.mean()), "median_absolute_error": float(np.median(absolute)), "mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson": pearson, "spearman": spearman, "r2": float(1 - np.sum(error ** 2) / denom) if denom else float("nan"), "absolute_error_p90": float(np.quantile(absolute, .90)), "absolute_error_p95": float(np.quantile(absolute, .95))}


def stratified_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for group in groups:
        if group not in frame:
            continue
        for label, subset in frame.groupby(group, dropna=False):
            rows.append({"stratifier": group, "group": str(label), **metric_dict(subset.observed_rt, subset.predicted_rt)})
    return pd.DataFrame(rows)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(config: dict[str, Any], model_path: Path, dictionary_path: Path, dataset_path: Path, compatibility: dict[str, Any], n: int) -> dict[str, Any]:
    return {"schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_name": config["evaluation_name"], "model": {**config["model"], "path": str(model_path.resolve()), "sha256": sha256(model_path)}, "dataset": {**config["dataset"], "path": str(dataset_path.resolve()), "sha256": sha256(dataset_path)}, "token_dictionary": {"path": str(dictionary_path.resolve()), "sha256": sha256(dictionary_path), **compatibility}, "n_peptides": n, "captum": config.get("captum", {}), "config": config, "git_commit": git_commit(Path(__file__).resolve().parents[1]), "software": {"python": platform.python_version(), "torch": str(torch.__version__), "pandas": str(pd.__version__), "numpy": str(np.__version__)}}


def load_runtime(config: dict[str, Any], device_override: str | None = None) -> tuple[torch.nn.Module, TokenEncoder, torch.device, dict[str, Any]]:
    device = resolve_device(device_override or config.get("runtime", {}).get("device", "auto"))
    model_path = Path(config["model"]["path"]); dictionary_path = Path(config["model"]["token_dictionary"])
    verify_torchscript_model(model_path)
    model = load_torchscript_model(model_path, device).eval()
    tokens = read_tokens(dictionary_path)
    return model, TokenEncoder(tokens), device, compatibility_check(model, tokens)
