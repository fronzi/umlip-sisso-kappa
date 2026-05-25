#!/usr/bin/env python
"""Assemble features (T0, T1, T2 union) and run symbolic regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from umlip_kappa.io_utils import ensure_dir, get_logger, load_config
from umlip_kappa.sr import pareto_front, run_symbolic_regression

log = get_logger()


def assemble_feature_table(cfg: dict, umlip: str, tier: str) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["output_dir"])
    bench = pd.read_csv(cfg["paths"]["benchmark_csv"])[
        ["mp_id", "kappa_ref", "log10_kappa", "split", "formula", "n_atoms_primitive"]
    ]
    T0 = pd.read_csv(out_dir / "features_T0.csv")
    if tier == "T0":
        df = bench.merge(T0, on="mp_id", how="inner")
    elif tier == "T1":
        T1 = pd.read_csv(out_dir / f"features_T1_{umlip}.csv")
        df = bench.merge(T0, on="mp_id").merge(T1, on="mp_id", how="inner")
    elif tier == "T2":
        T1 = pd.read_csv(out_dir / f"features_T1_{umlip}.csv")
        T2 = pd.read_csv(out_dir / f"features_T2_{umlip}.csv")
        df = (bench.merge(T0, on="mp_id")
                   .merge(T1, on="mp_id")
                   .merge(T2, on="mp_id", how="inner"))
    else:
        raise ValueError(tier)

    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))      # drop sparse columns
    df = df.dropna(axis=0)                                  # drop incomplete rows
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--tier", "-t", choices=["T0", "T1", "T2"], required=True)
    ap.add_argument("--umlip", "-u", required=True)
    ap.add_argument("--max-complexity", type=int, default=None,
                    help="Override sr.max_complexity in the config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_complexity is not None:
        cfg["sr"]["max_complexity"] = args.max_complexity

    df = assemble_feature_table(cfg, args.umlip, args.tier)
    log.info("Assembled %d materials × %d features", len(df), df.shape[1])

    # Target = log10 κ_L; features = numeric columns excluding metadata
    meta_cols = {"mp_id", "kappa_ref", "log10_kappa", "split", "formula", "umlip"}
    feature_cols = [c for c in df.columns if c not in meta_cols
                    and pd.api.types.is_numeric_dtype(df[c])]
    df_train = df[df["split"] == "train"]
    X = df_train[feature_cols]
    y = df_train["log10_kappa"]

    results = run_symbolic_regression(X, y, cfg)
    pf = pareto_front(results)

    out_dir = ensure_dir(Path(cfg["paths"]["output_dir"]) / "sr")
    pf_records = [
        {
            "formula": r.formula,
            "complexity": r.complexity,
            "cv_mae": r.cv_mae,
            "train_mae": r.train_mae,
            "backend": r.backend,
        }
        for r in pf
    ]
    out = out_dir / f"pareto_{args.tier}_{args.umlip}.json"
    with open(out, "w") as f:
        json.dump(pf_records, f, indent=2)
    log.info("Wrote %d Pareto-optimal models to %s", len(pf_records), out)

    # Also write the full feature table used (for downstream validation)
    df.to_csv(out_dir / f"data_{args.tier}_{args.umlip}.csv", index=False)


if __name__ == "__main__":
    main()
