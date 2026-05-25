"""Symbolic regression on the assembled feature table.

Two back-ends:

* **PySR** (recommended, pure-Python install) — multi-population genetic
  programming with parsimony pressure; returns a full Pareto front.

* **SISSO++** (optional) — compressed-sensing search over operator-built
  feature spaces; we call the official `sisso++` binary via a subprocess
  wrapper and parse its output.

Both back-ends share a common input/output interface so the rest of the
pipeline does not care which is used.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ===================================================================
# Common dataclasses
# ===================================================================
@dataclass
class SRResult:
    formula: str
    complexity: int
    cv_mae: float
    train_mae: float
    coefficients: dict
    feature_names: list[str]
    backend: str
    aux: dict = field(default_factory=dict)


# ===================================================================
# PySR back-end
# ===================================================================
def run_pysr(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    cv_folds: int = 5,
    random_state: int = 0,
) -> list[SRResult]:
    """Run PySR with `cv_folds`-fold cross-validation; return the Pareto front."""
    from pysr import PySRRegressor

    pcfg = cfg["sr"]["pysr"]
    feat = list(X.columns)

    base_kwargs = dict(
        binary_operators=cfg["sr"]["binary_operators"],
        unary_operators=cfg["sr"]["unary_operators"],
        populations=pcfg["populations"],
        niterations=pcfg["niterations"],
        population_size=pcfg["population_size"],
        parsimony=pcfg["parsimony"],
        model_selection=pcfg["model_selection"],
        maxsize=cfg["sr"]["max_complexity"] * 8 + 5,
        random_state=random_state,
        deterministic=True,
        procs=0,  # use Julia's threading
        verbosity=0,
        progress=False,
    )

    # Cross-validated MAE per Pareto point
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    eq_to_cv = {}
    for tr, te in kf.split(X):
        model = PySRRegressor(**base_kwargs)
        model.fit(X.iloc[tr].values, y.iloc[tr].values, variable_names=feat)
        # Predict with each equation in the Pareto front
        for _, row in model.equations_.iterrows():
            eq_str = row["equation"]
            complexity = int(row["complexity"])
            y_pred = model.predict(X.iloc[te].values, index=row.name)
            mae = float(np.mean(np.abs(y_pred - y.iloc[te].values)))
            key = (eq_str, complexity)
            eq_to_cv.setdefault(key, []).append(mae)

    # Final fit on all data
    final = PySRRegressor(**base_kwargs)
    final.fit(X.values, y.values, variable_names=feat)
    results = []
    for _, row in final.equations_.iterrows():
        eq_str = row["equation"]
        complexity = int(row["complexity"])
        cv_list = eq_to_cv.get((eq_str, complexity), [np.nan])
        results.append(
            SRResult(
                formula=eq_str,
                complexity=complexity,
                cv_mae=float(np.mean(cv_list)),
                train_mae=float(row["loss"]),
                coefficients={},
                feature_names=feat,
                backend="pysr",
                aux={"score": float(row.get("score", 0.0))},
            )
        )
    return results


# ===================================================================
# SISSO++ back-end
# ===================================================================
def run_sisso(
    X: pd.DataFrame, y: pd.Series, cfg: dict, sisso_binary: str = "sisso++",
) -> list[SRResult]:
    """Run SISSO++ via a temporary directory.

    Requires `sisso++` on the PATH (build from
    https://gitlab.com/sissopp_developers/sissopp).
    """
    if shutil.which(sisso_binary) is None:
        raise RuntimeError(f"`{sisso_binary}` not found on PATH")

    scfg = cfg["sr"]["sisso"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Build SISSO++ input table
        df = X.copy()
        df.insert(0, "Property", y.values)
        df.insert(0, "Sample", np.arange(len(df)))
        df.to_csv(tmp / "data.csv", index=False)
        # Build SISSO++ input file
        operators = " ".join(["(+)", "(-)", "(*)", "(/)", "(sqrt)", "(log)", "(exp)", "(^2)", "(^3)"])
        inp = f"""data_file = data.csv
property_key = Property
desc_dim = {scfg['desc_dim']}
n_residual = 1
n_models_store = 10
n_sis_select = {scfg['n_top']}
max_rung = {scfg['rung']}
opset = {operators}
calc_type = regression
"""
        (tmp / "sisso.json").write_text(inp)

        out = subprocess.run(
            [sisso_binary, str(tmp / "sisso.json")],
            cwd=tmp, check=True, capture_output=True, text=True,
        )
        # Parse SISSO++ output (omitted details: read `models/train_*.dat`)
        # Below: minimal parser; refine for production.
        results = []
        for log_path in sorted((tmp / "models").glob("train_*.dat")):
            text = log_path.read_text().splitlines()
            # Each file contains: header lines, then 'Equation: ...' and 'RMSE = ...'
            eq, rmse = None, None
            for line in text:
                if line.startswith("Equation"):
                    eq = line.split(":", 1)[1].strip()
                if "RMSE" in line:
                    try:
                        rmse = float(line.split("=", 1)[1].strip().split()[0])
                    except Exception:
                        pass
            if eq is not None and rmse is not None:
                # SISSO complexity ≈ #features in the model
                complexity = eq.count("+") + 1
                results.append(
                    SRResult(
                        formula=eq, complexity=complexity,
                        cv_mae=float("nan"), train_mae=rmse,
                        coefficients={}, feature_names=list(X.columns),
                        backend="sisso", aux={"raw": "\n".join(text)},
                    )
                )
        return results


# ===================================================================
# Public API
# ===================================================================
def run_symbolic_regression(
    X: pd.DataFrame, y: pd.Series, cfg: dict,
) -> list[SRResult]:
    """Dispatch on the configured back-end."""
    backend = cfg["sr"]["backend"]
    if backend == "pysr":
        return run_pysr(X, y, cfg, cv_folds=cfg["validation"]["cv_folds"])
    if backend == "sisso":
        return run_sisso(X, y, cfg)
    raise ValueError(f"Unknown SR back-end '{backend}'")


def pareto_front(results: Sequence[SRResult]) -> list[SRResult]:
    """Filter `results` to only the non-dominated points in
    (complexity, cv_mae) space."""
    front: list[SRResult] = []
    sorted_r = sorted(results, key=lambda r: (r.complexity, r.cv_mae))
    best = float("inf")
    for r in sorted_r:
        if r.cv_mae < best:
            front.append(r)
            best = r.cv_mae
    return front
