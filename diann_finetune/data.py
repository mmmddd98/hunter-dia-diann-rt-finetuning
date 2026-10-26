"""Carafe-compatible RT data loading and sequence-disjoint splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .encoding import carafe_modified_sequence


REQUIRED_RT_COLUMNS = {"sequence", "rt_norm"}


@dataclass
class PreparedRTData:
    frame: pd.DataFrame
    custom_token_counts: Dict[str, int]
    dropped_rows: int


def load_carafe_rt(path: Path, *, unknown_mods: str = "error", max_rows: int | None = None) -> PreparedRTData:
    """Load Carafe's ``rt_train_data.tsv`` and add DIA-NN peptide strings."""
    if unknown_mods not in {"error", "drop"}:
        raise ValueError("unknown_mods must be 'error' or 'drop'")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    frame = pd.read_csv(path, sep="\t", dtype={"mods": str, "mod_sites": str})
    if "sequence" not in frame.columns and "peptide" in frame.columns:
        frame["sequence"] = frame["peptide"]
    missing = REQUIRED_RT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if "mods" not in frame.columns:
        frame["mods"] = ""
    if "mod_sites" not in frame.columns:
        frame["mod_sites"] = ""
    frame["mods"] = frame["mods"].fillna("").astype(str)
    frame["mod_sites"] = frame["mod_sites"].fillna("").astype(str)
    frame["rt_norm"] = pd.to_numeric(frame["rt_norm"], errors="coerce")
    # Some evaluation tables also contain the observed RT in minutes.  Keep
    # it through aggregation so metrics can be reported on that native scale.
    if "rt" in frame.columns:
        frame["rt"] = pd.to_numeric(frame["rt"], errors="coerce")
    frame = frame.dropna(subset=["sequence", "rt_norm"]).copy()
    if not frame["rt_norm"].map(math.isfinite).all():
        raise ValueError(f"RT targets must be finite in {path}")
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=1337).copy()

    converted: List[str | None] = []
    custom_token_counts: Dict[str, int] = {}
    dropped = 0
    for row in frame.itertuples(index=False):
        try:
            modified, custom_tokens = carafe_modified_sequence(row.sequence, row.mods, row.mod_sites)
        except ValueError:
            if unknown_mods == "drop":
                converted.append(None)
                dropped += 1
                continue
            raise
        converted.append(modified)
        for custom_token in set(custom_tokens):
            custom_token_counts[custom_token] = custom_token_counts.get(custom_token, 0) + 1
    frame["diann_sequence"] = converted
    frame = frame.dropna(subset=["diann_sequence"]).copy()

    # Match Carafe2: one median target for each sequence/modification form.
    # Keep organism provenance when evaluating cross-organism hold-outs.  It
    # must be part of the grouping key because the same sequence can occur in
    # more than one organism with different observed RTs.
    group_columns = ["sequence", "mods", "mod_sites", "diann_sequence"]
    if "source_organism" in frame.columns:
        group_columns.append("source_organism")
    target_columns = ["rt_norm"]
    if "rt" in frame.columns:
        target_columns.append("rt")
    frame = (
        frame.groupby(group_columns, dropna=False, as_index=False)[target_columns]
        .median()
        .reset_index(drop=True)
    )
    frame["nAA"] = frame["sequence"].str.len()
    if frame.empty:
        raise ValueError(f"No usable peptide observations remain in {path}")
    return PreparedRTData(frame=frame, custom_token_counts=custom_token_counts, dropped_rows=dropped)


def _sample_with_important_mods(
    frame: pd.DataFrame,
    n_sample: int,
    *,
    top_n_mods: int = 10,
    rows_per_mod: int = 50,
    random_state: int = 1337,
) -> pd.DataFrame:
    """Mirror Carafe2's RT sampling, including modification-enrichment rows."""
    if n_sample >= len(frame):
        return frame.copy()
    base_sample = frame.sample(n_sample, replace=False, random_state=random_state).copy()
    sampled = [base_sample]
    # Carafe enriches important modifications after its base sample. Restrict
    # enrichment to already selected sequences so it cannot leak a hold-out
    # sequence back into training on small datasets.
    enrichment_pool = frame[frame["sequence"].isin(set(base_sample["sequence"]))]
    counts: Dict[str, int] = {}
    for value in enrichment_pool["mods"].fillna(""):
        for mod in set(item for item in value.split(";") if item):
            counts[mod] = counts.get(mod, 0) + 1
    for mod, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n_mods]:
        candidates = enrichment_pool[
            enrichment_pool["mods"].str.contains(mod, regex=False, na=False)
        ]
        n = min(rows_per_mod, len(candidates))
        if n:
            sampled.append(candidates.sample(n, replace=False, random_state=random_state).copy())
    return pd.concat(sampled, ignore_index=True)


def carafe_sequence_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Reproduce Carafe2's seeded, sequence-disjoint RT hold-out split."""
    if frame["sequence"].nunique() < 2:
        raise ValueError("Sequence-disjoint training and validation require at least two distinct sequences")
    n_test = max(1, min(1000, int(math.ceil(len(frame) * 0.1) - 10)))
    n_train = max(1, len(frame) - n_test)
    train = _sample_with_important_mods(frame, n_train)
    validation = frame[~frame["sequence"].isin(set(train["sequence"]))].copy()
    if len(validation) > n_test:
        validation = validation.sample(n_test, random_state=1337).copy()
    elif validation.empty:
        # Preserve sequence separation for duplicate-heavy/small inputs instead
        # of using Carafe's leakage-prone fallback to the complete frame.
        unique_sequences = pd.Series(frame["sequence"].unique())
        n_validation_sequences = max(1, min(n_test, len(unique_sequences) - 1))
        validation_sequences = set(
            unique_sequences.sample(n_validation_sequences, random_state=1337).tolist()
        )
        validation = frame[frame["sequence"].isin(validation_sequences)].copy()
        train_pool = frame[~frame["sequence"].isin(validation_sequences)].copy()
        train = _sample_with_important_mods(train_pool, min(n_train, len(train_pool)))
    metadata = {
        "source_rows": len(frame),
        "requested_training_rows": n_train,
        "training_rows_after_mod_enrichment": len(train),
        "training_unique_sequences": train["sequence"].nunique(),
        "validation_rows": len(validation),
        "validation_unique_sequences": validation["sequence"].nunique(),
    }
    return train.reset_index(drop=True), validation.reset_index(drop=True), metadata


def nested_training_subset(
    frame: pd.DataFrame,
    fraction: float,
    *,
    random_state: int = 2024,
) -> pd.DataFrame:
    """Return a deterministic, nested fraction of an existing training split.

    A single seeded permutation is used for every fraction. Consequently, the
    25% subset is contained in the 50% subset, which is contained in 75%, and
    100% returns the original frame unchanged. Sampling happens only after the
    sequence-disjoint hold-out has been created, so validation never changes.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Training fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0 or len(frame) <= 1:
        return frame.copy().reset_index(drop=True)
    order = pd.Series(range(len(frame))).sample(
        frac=1.0, replace=False, random_state=random_state
    )
    n_rows = max(1, int(math.ceil(len(frame) * fraction)))
    return frame.iloc[order.iloc[:n_rows].to_numpy()].copy().reset_index(drop=True)


HEAVY_NTERM = "Dimethyl:2H(4)13C(2)@N-term"
HEAVY_LYSINE = "Dimethyl:2H(4)13C(2)@K"


def heavy_dimethyl_row_strata(frame: pd.DataFrame) -> pd.Series:
    """Classify each modification form into one exact Heavy-Dimethyl stratum."""
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains(HEAVY_NTERM, regex=False)
    lysine = mods.str.contains(HEAVY_LYSINE, regex=False)
    labels = pd.Series("normal", index=frame.index, dtype="object")
    labels.loc[nterm & ~lysine] = "heavy_nterm"
    labels.loc[lysine & ~nterm] = "heavy_lysine"
    labels.loc[nterm & lysine] = "heavy_both"
    return labels


def oversample_heavy_dimethyl_rows(
    frame: pd.DataFrame,
    *,
    nterm_factor: float = 1.0,
    lysine_factor: float = 1.0,
    both_factor: float = 1.0,
    random_state: int = 2024,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Deterministically duplicate selected Heavy-Dimethyl training rows.

    This function is intended to run only after the sequence-disjoint split is
    fixed. A factor of 2.0 adds one extra copy of every base row in that
    stratum; fractional factors sample the corresponding additional rows.
    """
    factors = {
        "normal": 1.0,
        "heavy_nterm": float(nterm_factor),
        "heavy_lysine": float(lysine_factor),
        "heavy_both": float(both_factor),
    }
    if any(not math.isfinite(value) or value < 1.0 for value in factors.values()):
        raise ValueError("Heavy-Dimethyl oversampling factors must be finite and >= 1")

    strata = heavy_dimethyl_row_strata(frame)
    pieces = [frame.copy()]
    base_counts = {name: int((strata == name).sum()) for name in factors}
    added_counts: Dict[str, int] = {}
    for offset, (stratum, factor) in enumerate(factors.items()):
        extra = int(round((factor - 1.0) * base_counts[stratum]))
        candidates = frame.loc[strata == stratum]
        if extra <= 0 or candidates.empty:
            added_counts[stratum] = 0
            continue
        pieces.append(
            candidates.sample(
                n=extra,
                replace=extra > len(candidates),
                random_state=random_state + offset + 1,
            )
        )
        added_counts[stratum] = extra
    result = pd.concat(pieces, ignore_index=True)
    metadata: Dict[str, object] = {
        "factors": factors,
        "base_counts": base_counts,
        "added_counts": added_counts,
        "base_rows": len(frame),
        "oversampled_rows": len(result),
    }
    return result, metadata


def heavy_dimethyl_sequence_strata(frame: pd.DataFrame) -> pd.Series:
    """Assign every amino-acid sequence to one exact heavy-dimethyl stratum.

    A sequence can occur with more than one modification form.  In that case
    the most specific form wins (both sites > lysine > N-terminus > neither).
    This gives a single, deterministic stratum per sequence for split and
    sampling purposes; no sequence can then occur in train and test.
    """
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains(HEAVY_NTERM, regex=False)
    lysine = mods.str.contains(HEAVY_LYSINE, regex=False)
    labels = pd.Series("none", index=frame.index, dtype="object")
    labels.loc[nterm] = "heavy_nterm"
    labels.loc[lysine] = "heavy_lysine"
    labels.loc[nterm & lysine] = "heavy_both"
    priority = {"none": 0, "heavy_nterm": 1, "heavy_lysine": 2, "heavy_both": 3}
    grouped = pd.DataFrame({"sequence": frame["sequence"], "stratum": labels})
    grouped["priority"] = grouped["stratum"].map(priority)
    chosen = grouped.sort_values("priority").groupby("sequence", sort=True)["stratum"].last()
    return chosen


def stratified_sequence_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    random_state: int = 2024,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Create a seeded, sequence-disjoint split stratified by heavy status."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    strata = heavy_dimethyl_sequence_strata(frame)
    validation_sequences: List[str] = []
    for offset, (stratum, sequences) in enumerate(strata.groupby(strata, sort=True)):
        values = sequences.index.to_list()
        if len(values) < 2:
            continue
        shuffled = pd.Series(values).sample(frac=1.0, random_state=random_state + offset)
        n_validation = max(1, int(round(len(values) * validation_fraction)))
        n_validation = min(n_validation, len(values) - 1)
        validation_sequences.extend(shuffled.iloc[:n_validation].tolist())
    validation_set = set(validation_sequences)
    train = frame[~frame["sequence"].isin(validation_set)].copy().reset_index(drop=True)
    validation = frame[frame["sequence"].isin(validation_set)].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Could not create non-empty sequence-disjoint train/validation split")
    metadata: Dict[str, object] = {
        "validation_fraction_requested": validation_fraction,
        "seed": random_state,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "training_unique_sequences": train["sequence"].nunique(),
        "validation_unique_sequences": validation["sequence"].nunique(),
        "sequence_strata": strata.value_counts().sort_index().to_dict(),
    }
    return train, validation, metadata


def nested_stratified_sequence_subset(
    frame: pd.DataFrame,
    fraction: float,
    *,
    random_state: int = 2024,
) -> pd.DataFrame:
    """Return nested, stratum-preserving sequence subsets of a training frame."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Training fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return frame.copy().reset_index(drop=True)
    strata = heavy_dimethyl_sequence_strata(frame)
    selected: List[str] = []
    for offset, (_, sequences) in enumerate(strata.groupby(strata, sort=True)):
        values = sequences.index.to_list()
        n_selected = max(1, int(math.ceil(len(values) * fraction)))
        order = pd.Series(values).sample(frac=1.0, random_state=random_state + offset)
        selected.extend(order.iloc[:n_selected].tolist())
    return frame[frame["sequence"].isin(set(selected))].copy().reset_index(drop=True)
