"""Convert Carafe peptide rows to DIA-NN's token representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import torch


# DIA-NN 2.5 has no distributed heavy-dimethyl token. Add stable IDs directly
# after its highest existing ID; they fit the model's 256-row embedding table.
HEAVY_DIMETHYL_TOKEN_IDS = {
    "<(UniMod:510)": 40,
    "K(UniMod:510)": 41,
}

HEAVY_DIMETHYL_ALIASES = {
    "<(unimod:510)": 40,
    "<(Dimethyl:2H(4)13C(2))": 40,
    "K(unimod:510)": 41,
    "K(Dimethyl:2H(4)13C(2))": 41,
}

CARAFE_TO_DIANN: Mapping[str, Tuple[str, str]] = {
    "Acetyl@N-term": ("N-term", "UniMod:1"),
    "Carbamidomethyl@C": ("C", "UniMod:4"),
    "Oxidation@M": ("M", "UniMod:35"),
    "Dimethyl@N-term": ("N-term", "UniMod:255"),
    "Dimethyl@K": ("K", "UniMod:255"),
    "Dimethyl:2H(4)13C(2)@N-term": ("N-term", "UniMod:510"),
    "Dimethyl:2H(4)13C(2)@K": ("K", "UniMod:510"),
}

CUSTOM_TOKEN_USAGE = {
    "Dimethyl:2H(4)13C(2)@N-term": "UniMod:510 N-term token (ID 40)",
    "Dimethyl:2H(4)13C(2)@K": "UniMod:510 K token (ID 41)",
}


def read_tokens(path: Path) -> Dict[str, int]:
    """Read DIA-NN's tab-separated token dictionary."""
    tokens: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        token, token_id = line.split("\t")
        tokens[token] = int(token_id)
    if "<" not in tokens or ">" not in tokens:
        raise ValueError(f"Invalid DIA-NN dictionary: {path}")
    return tokens


def augment_tokens_for_heavy_dimethyl(tokens: Mapping[str, int]) -> Tuple[Dict[str, int], Dict[int, int]]:
    """Add UniMod 510 tokens and return new->light embedding initialization IDs."""
    augmented = dict(tokens)
    source_tokens = {
        "<(UniMod:510)": "<(UniMod:255)",
        "K(UniMod:510)": "K(UniMod:255)",
    }
    initializations: Dict[int, int] = {}
    occupied = set(augmented.values())
    for token, token_id in HEAVY_DIMETHYL_TOKEN_IDS.items():
        if token in augmented and augmented[token] != token_id:
            raise ValueError(f"Conflicting token ID for {token}: {augmented[token]} vs {token_id}")
        if token_id in occupied and augmented.get(token) != token_id:
            owners = [name for name, value in augmented.items() if value == token_id]
            raise ValueError(f"Token ID {token_id} already used by {owners}")
        source = source_tokens[token]
        if source not in augmented:
            raise ValueError(f"DIA-NN dictionary lacks initialization token {source}")
        augmented[token] = token_id
        occupied.add(token_id)
        initializations[token_id] = augmented[source]
    augmented.update(HEAVY_DIMETHYL_ALIASES)
    return augmented, initializations


def write_tokens(path: Path, tokens: Mapping[str, int]) -> None:
    """Write a DIA-NN dictionary, preserving aliases as separate lines."""
    path.write_text(
        "".join(f"{token}\t{token_id}\n" for token, token_id in tokens.items()),
        encoding="utf-8",
    )


def carafe_modified_sequence(sequence: str, mods: str = "", mod_sites: str = "") -> Tuple[str, List[str]]:
    """Translate Carafe ``mods``/``mod_sites`` into DIA-NN UniMod syntax.

    Carafe sites are one-based residue positions; site 0 denotes the peptide
    N-terminus. The returned string deliberately omits DIA-NN's ``<``/``>``
    markers because :class:`TokenEncoder` adds them during tokenization.
    """
    sequence = str(sequence).strip().upper()
    if not sequence or not sequence.isalpha():
        raise ValueError(f"Invalid peptide sequence: {sequence!r}")

    mod_names = [item.strip() for item in str(mods or "").split(";") if item.strip()]
    sites = [item.strip() for item in str(mod_sites or "").split(";") if item.strip()]
    if not mod_names:
        if sites:
            raise ValueError(f"Modification sites without modifications for {sequence}")
        return sequence, []
    if len(mod_names) != len(sites):
        raise ValueError(
            f"Modification/site count mismatch for {sequence}: {mod_names!r} vs {sites!r}"
        )

    residue_mods: Dict[int, List[str]] = {}
    nterm_mods: List[str] = []
    custom_tokens: List[str] = []
    for mod_name, site_text in zip(mod_names, sites):
        if mod_name not in CARAFE_TO_DIANN:
            raise ValueError(f"Unsupported Carafe modification: {mod_name}")
        expected_site, unimod = CARAFE_TO_DIANN[mod_name]
        try:
            site = int(site_text)
        except ValueError as exc:
            raise ValueError(f"Invalid modification site {site_text!r} for {sequence}") from exc

        if expected_site == "N-term":
            if site != 0:
                raise ValueError(f"N-terminal modification {mod_name} has nonzero site {site}")
            nterm_mods.append(unimod)
        else:
            if site < 1 or site > len(sequence):
                raise ValueError(f"Modification site {site} outside {sequence}")
            if sequence[site - 1] != expected_site:
                raise ValueError(
                    f"{mod_name} expects {expected_site} at site {site}, found {sequence[site - 1]}"
                )
            residue_mods.setdefault(site, []).append(unimod)
        if mod_name in CUSTOM_TOKEN_USAGE:
            custom_tokens.append(CUSTOM_TOKEN_USAGE[mod_name])

    prefix = "".join(f"({unimod})" for unimod in nterm_mods)
    parts: List[str] = [prefix]
    for site, residue in enumerate(sequence, start=1):
        parts.append(residue)
        parts.extend(f"({unimod})" for unimod in residue_mods.get(site, []))
    return "".join(parts), custom_tokens


@dataclass(frozen=True)
class TokenEncoder:
    """Encode modified peptide strings for DIA-NN TorchScript modules."""

    tokens: Mapping[str, int]

    def tokenize(self, peptide: str) -> List[int]:
        text = f"<{peptide}>"
        candidates = sorted(self.tokens, key=len, reverse=True)
        result: List[int] = []
        pos = 0
        while pos < len(text):
            match = next((token for token in candidates if text.startswith(token, pos)), None)
            if match is None:
                raise ValueError(f"No DIA-NN token at character {pos}: {text[pos:]!r}")
            result.append(self.tokens[match])
            pos += len(match)
        return result

    def encode(self, peptide: str, charge: int = 2) -> torch.Tensor:
        return torch.tensor([charge, *self.tokenize(peptide)], dtype=torch.long)

    def encode_many(self, peptides: Iterable[str], charge: int = 2) -> List[torch.Tensor]:
        return [self.encode(peptide, charge=charge) for peptide in peptides]
