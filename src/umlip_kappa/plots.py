"""Publication-quality plots for the manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_dataset(df: pd.DataFrame, out: Path) -> None:
    """Fig. 1: dataset histograms (κ_L, n_atoms, chemistry)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].hist(np.log10(df["kappa_ref"]), bins=20, edgecolor="k")
    axes[0].set_xlabel(r"$\log_{10}\kappa_L(300\,\mathrm{K})$ [W m$^{-1}$ K$^{-1}$]")
    axes[0].set_ylabel("count")
    axes[1].hist(df["n_atoms_primitive"], bins=20, edgecolor="k")
    axes[1].set_xlabel(r"atoms in primitive cell $N$")
    axes[2].hist(df["band_gap"], bins=20, edgecolor="k")
    axes[2].set_xlabel(r"PBE band gap [eV]")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_umlip_parity(
    df_uMLIP: pd.DataFrame, df_DFT: pd.DataFrame, features: list[str], out: Path,
) -> None:
    """Fig. 2: uMLIP vs DFT parity panels."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for ax, f in zip(axes.flat, features):
        common = df_uMLIP.index.intersection(df_DFT.index)
        x = df_DFT.loc[common, f].values
        y = df_uMLIP.loc[common, f].values
        ax.scatter(x, y, s=12, alpha=0.7)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"DFT {f}")
        ax.set_ylabel(f"uMLIP {f}")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(results, out: Path, title: str = "") -> None:
    """Fig. 3: Pareto front (CV-MAE vs complexity)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    cs = [r.complexity for r in results]
    es = [r.cv_mae for r in results]
    ax.plot(cs, es, "o-", mfc="white")
    for r in results:
        ax.annotate(
            f"  {r.formula[:24]}{'…' if len(r.formula) > 24 else ''}",
            (r.complexity, r.cv_mae), fontsize=7, alpha=0.8,
        )
    ax.set_xlabel("complexity")
    ax.set_ylabel(r"5-fold CV MAE on $\log_{10}\kappa_L$")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_parity_kappa(y_true, y_pred, y_slack, out: Path) -> None:
    """Fig. 4: parity vs reference κ_L, with Slack baseline."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_slack, s=14, c="0.5", label="Slack", marker="s")
    ax.scatter(y_true, y_pred, s=18, c="C0", label="this work")
    lo, hi = -1.5, 3.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.fill_between([lo, hi], [lo - 0.3, hi - 0.3], [lo + 0.3, hi + 0.3], color="0.9")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"reference $\log_{10}\kappa_L$")
    ax.set_ylabel(r"predicted $\log_{10}\kappa_L$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_softening(decomp: pd.DataFrame, out: Path) -> None:
    """Fig. 5: per-feature contribution to softening bias."""
    fig, ax = plt.subplots(figsize=(5, 3))
    decomp_sorted = decomp.sort_values("rms_delta")
    ax.barh(decomp_sorted["feature"], decomp_sorted["rms_delta"])
    ax.set_xlabel(r"RMS $\Delta\log_{10}\kappa_L$ from uMLIP→DFT swap")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
