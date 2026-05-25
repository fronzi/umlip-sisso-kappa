#!/usr/bin/env python
"""Generate the publication figures from the cached pipeline artefacts."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from umlip_kappa.io_utils import ensure_dir, get_logger, load_config
from umlip_kappa.plots import (
    fig_dataset, fig_pareto, fig_parity_kappa, fig_softening, fig_umlip_parity,
)
from umlip_kappa.slack import slack_from_features

log = get_logger()


@dataclass
class Lite:
    formula: str
    complexity: int
    cv_mae: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--umlip", "-u", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["paths"]["output_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])

    bench = pd.read_csv(cfg["paths"]["benchmark_csv"])
    fig_dataset(bench, fig_dir / "fig1_dataset.png")

    # Pareto fronts
    for tier in ["T0", "T1", "T2"]:
        p = out_dir / "sr" / f"pareto_{tier}_{args.umlip}.json"
        if p.exists():
            recs = json.loads(p.read_text())
            lite = [Lite(**{k: r[k] for k in ("formula", "complexity", "cv_mae")}) for r in recs]
            fig_pareto(lite, fig_dir / f"fig3_pareto_{tier}.png", title=f"Tier {tier} ({args.umlip})")

    # Parity (T2 best vs Slack baseline)
    p_t2 = out_dir / "sr" / f"pareto_T2_{args.umlip}.json"
    data_t2 = out_dir / "sr" / f"data_T2_{args.umlip}.csv"
    if p_t2.exists() and data_t2.exists():
        recs = json.loads(p_t2.read_text())
        df = pd.read_csv(data_t2)
        T0 = pd.read_csv(out_dir / "features_T0.csv")
        T1 = pd.read_csv(out_dir / f"features_T1_{args.umlip}.csv")
        T2 = pd.read_csv(out_dir / f"features_T2_{args.umlip}.csv")
        y_slack = slack_from_features(T0, T1, T2)
        # Best-by-CV-MAE entry
        best = min(recs, key=lambda r: r["cv_mae"])
        from umlip_kappa.validate import _formula_to_sympy
        import sympy as sp
        expr = _formula_to_sympy(
            best["formula"],
            [c for c in df.columns if df[c].dtype.kind in "fi"],
        )
        feats = sorted(str(s) for s in expr.free_symbols)
        f = sp.lambdify(feats, expr, "numpy")
        y_pred = f(*[df[c].values for c in feats])
        fig_parity_kappa(
            y_true=df["log10_kappa"].values,
            y_pred=y_pred,
            y_slack=__import__("numpy").log10(y_slack.reindex(df["mp_id"]).values),
            out=fig_dir / "fig4_parity.png",
        )

    # Softening
    soft = out_dir / "sr" / f"softening_T2_{args.umlip}_c2.csv"
    if soft.exists():
        decomp = pd.read_csv(soft)
        fig_softening(decomp, fig_dir / "fig5_softening.png")

    log.info("Figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
