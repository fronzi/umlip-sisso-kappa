"""Curate the benchmark dataset of crystalline insulators with reference κ_L.

The seed CSV (data/benchmark_seed.csv) is hand-curated by the user from
published tables:

  * Póta et al. (npj Comput. Mater. 2025) — 103 solids, DFT-PBE Wigner κ_L
  * Carrete et al. (PRX 2014)             — half-Heuslers
  * Morelli & Slack (2006)                — experimental compilation
  * Toberer et al. (J. Mater. Chem. 2011) — experimental thermoelectrics

Expected columns:
    mp_id, formula, kappa_ref, kappa_ref_uncertainty, ref_method, ref_doi

This module:
  1. Pulls relaxed structures from Materials Project (mp-api),
  2. Applies physical filters (band gap, n_atoms, stability),
  3. Performs a decile-stratified train/test split of log10(κ_L).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .io_utils import get_logger

log = get_logger()


# -------------------------------------------------------------- MP retrieval
def fetch_mp_structures(mp_ids: list[str], api_key: str | None = None) -> dict:
    """Fetch relaxed structures + band gap from Materials Project.

    Parameters
    ----------
    mp_ids : list of str
        Like ['mp-149', 'mp-19006', ...].
    api_key : str, optional
        If None, reads MP_API_KEY from env.

    Returns
    -------
    dict mp_id -> {'structure': pymatgen.Structure, 'band_gap': float, 'energy_above_hull': float}
    """
    from mp_api.client import MPRester

    out = {}
    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=mp_ids,
            fields=["material_id", "structure", "band_gap", "energy_above_hull"],
        )
        for d in docs:
            out[d.material_id] = {
                "structure": d.structure,
                "band_gap": d.band_gap,
                "energy_above_hull": d.energy_above_hull,
            }
    log.info("Fetched %d/%d MP structures", len(out), len(mp_ids))
    return out


# -------------------------------------------------------------- filtering
def apply_filters(seed_df: pd.DataFrame, mp_data: dict, cfg: dict) -> pd.DataFrame:
    """Drop materials that are metallic, unstable, too big, or out of κ_L range."""
    dcfg = cfg["dataset"]
    rows = []
    rejected = {"missing_mp": 0, "metal": 0, "unstable": 0, "too_big": 0, "k_range": 0}

    for _, r in seed_df.iterrows():
        mp_id = r["mp_id"]
        if mp_id not in mp_data:
            rejected["missing_mp"] += 1
            continue
        m = mp_data[mp_id]
        if m["band_gap"] is None or m["band_gap"] < 0.05:
            rejected["metal"] += 1
            continue
        if m["energy_above_hull"] is not None and m["energy_above_hull"] > 0.05:
            rejected["unstable"] += 1
            continue
        n_at = len(m["structure"])
        if n_at > dcfg["max_atoms_primitive"]:
            rejected["too_big"] += 1
            continue
        if not (dcfg["kappa_min"] <= r["kappa_ref"] <= dcfg["kappa_max"]):
            rejected["k_range"] += 1
            continue
        rows.append(
            {
                **r.to_dict(),
                "band_gap": m["band_gap"],
                "energy_above_hull": m["energy_above_hull"],
                "n_atoms_primitive": n_at,
                "log10_kappa": float(np.log10(r["kappa_ref"])),
            }
        )

    log.info("Filtering rejected: %s; kept %d", rejected, len(rows))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- splitting


def stratified_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Decile-stratified train/test split of log10(κ_L).

    For small datasets (N < 20), uses an unstratified shuffle split instead.
    """
    from sklearn.model_selection import ShuffleSplit
    n = len(df)
    test_size = 1 - cfg["dataset"]["train_frac"]
    seed = cfg["dataset"]["random_seed"]

    if n < 20:
        log.info("N=%d < 20; using unstratified shuffle split", n)
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(ss.split(df))
    else:
        n_bins = min(10, max(2, n // 3))      # adapt #bins to N
        deciles = pd.qcut(df["log10_kappa"], q=n_bins, labels=False, duplicates="drop")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(df, deciles))

    df = df.copy()
    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df


def assemble_benchmark(seed_csv: str | Path, cfg: dict, api_key: str | None = None) -> pd.DataFrame:
    seed = pd.read_csv(seed_csv)
    required = {"mp_id", "formula", "kappa_ref", "ref_method", "ref_doi"}
    missing = required - set(seed.columns)
    if missing:
        raise ValueError(f"Seed CSV missing columns: {missing}")

    mp_data = fetch_mp_structures(seed["mp_id"].tolist(), api_key=api_key)
    df = apply_filters(seed, mp_data, cfg)
    df = stratified_split(df, cfg)
    # Attach the structure dict serialised, so downstream stages do not
    # need a second MP roundtrip:
    import json as _json
    df["structure_dict"] = df["mp_id"].apply(
        lambda i: _json.dumps(mp_data[i]["structure"].as_dict()))
    return df
