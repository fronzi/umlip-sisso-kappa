"""Tier T0: composition + simple structure features (no uMLIP, no DFT).

Uses matminer's magpie preset (Ward et al. 2016) plus density features from
pymatgen. Produces ~70–130 columns.
"""
from __future__ import annotations

import pandas as pd

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import DensityFeatures
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core import Composition, Structure


def compute_T0_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide dataframe of T0 features keyed by mp_id.

    Parameters
    ----------
    df : DataFrame with columns ['mp_id', 'formula', 'structure_dict'].
    """
    work = df[["mp_id", "formula"]].copy()
    work["composition"] = work["formula"].apply(Composition)

    ep = ElementProperty.from_preset("magpie")
    feats = ep.featurize_dataframe(work, col_id="composition", ignore_errors=True, return_errors=False)

    # Drop near-constant columns to keep SISSO tractable. Scale the cutoff
    # with N so the filter is meaningful on a single dataset but does not
    # nuke everything for small smoke-test inputs.
    feat_cols = [c for c in feats.columns if c.startswith("MagpieData ")]
    n_rows = len(feats)
    min_unique = max(2, int(0.05 * n_rows))
    cols_keep = [c for c in feat_cols if feats[c].nunique(dropna=True) >= min_unique]

    # Add a few structure-level features (density, packing fraction).
    structures = [Structure.from_dict(d) for d in df["structure_dict"]]
    dens = DensityFeatures()
    dens_rows = []
    for s in structures:
        try:
            dens_rows.append(dens.featurize(s))
        except Exception:
            dens_rows.append([float("nan")] * len(dens.feature_labels()))
    dens_df = pd.DataFrame(dens_rows, columns=dens.feature_labels())
    dens_df.insert(0, "mp_id", df["mp_id"].values)

    out = feats[["mp_id"] + cols_keep].merge(dens_df, on="mp_id", how="left")
    return out
