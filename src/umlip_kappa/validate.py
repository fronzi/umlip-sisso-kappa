"""Softening-bias analysis.

Given (a) the discovered SR formula and (b) a small DFT-reference subset
with reliable γ_G and σ_A values, we replace each uMLIP-derived feature
in the formula with its DFT counterpart and measure the change in
predicted log10(κ_L). The decomposition

    Δ log10 κ̂_L = Σ_i  (∂F̂/∂x_i) · ( x_i^DFT - x_i^uMLIP )

(computed numerically) tells us which features dominate the softening bias.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import sympy as sp


def _formula_to_sympy(formula: str, feature_names: list[str]) -> sp.Expr:
    syms = {n: sp.Symbol(n) for n in feature_names}
    # PySR uses Python-like syntax with `square`, `cube` as functions.
    formula = re.sub(r"\bsquare\(([^()]*)\)", r"((\1)**2)", formula)
    formula = re.sub(r"\bcube\(([^()]*)\)", r"((\1)**3)", formula)
    formula = formula.replace("neg(", "(-1)*(")
    expr = sp.sympify(formula, locals={**syms, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
    return expr


def softening_decomposition(
    formula: str,
    feature_names: list[str],
    X_umlip: pd.DataFrame,
    X_dft: pd.DataFrame,
) -> pd.DataFrame:
    """Per-feature contribution to Δ log10 κ_L when uMLIP features are
    swapped for their DFT counterparts.

    Both `X_umlip` and `X_dft` must have the same columns (subset of
    `feature_names`) and the same index (mp_id).
    """
    common = X_umlip.index.intersection(X_dft.index)
    Xu = X_umlip.loc[common, feature_names].copy()
    Xd = X_dft.loc[common, feature_names].copy()

    expr = _formula_to_sympy(formula, feature_names)
    f_expr = sp.lambdify(feature_names, expr, "numpy")

    y_umlip = f_expr(*[Xu[c].values for c in feature_names])
    y_dft = f_expr(*[Xd[c].values for c in feature_names])

    rows = []
    for c in feature_names:
        Xmix = Xu.copy()
        Xmix[c] = Xd[c]
        y_mix = f_expr(*[Xmix[col].values for col in feature_names])
        # Single-feature substitution contribution
        contrib = y_mix - y_umlip
        rows.append({
            "feature": c,
            "mean_delta": float(np.mean(contrib)),
            "median_delta": float(np.median(contrib)),
            "rms_delta": float(np.sqrt(np.mean(contrib**2))),
            "fraction_explained": float(
                np.var(contrib) / max(np.var(y_dft - y_umlip), 1e-12)
            ),
        })

    return pd.DataFrame(rows).sort_values("rms_delta", ascending=False).reset_index(drop=True)


def parity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedAE": float(np.median(np.abs(err))),
        "Bias": float(np.mean(err)),
        "R2": float(1 - np.var(err) / max(np.var(y_true), 1e-12)),
    }
