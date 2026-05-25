"""Slack semi-empirical κ_L baseline.

   κ_L(T) = A * M̄ * Θ_D^3 * δ / ( γ² * N^(2/3) * T )

where M̄ is in atomic mass units (amu), Θ_D in K, δ ≡ V_a^(1/3) in Å (with
V_a the volume per atom in Å³), γ dimensionless, N the number of atoms in the
primitive cell, T in K. With the empirical prefactor A ≈ 3.04 × 10⁻⁶ the
output is in W/(m·K). See Morelli & Slack (2006), Eq. (4) and Table 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def slack_kappa(
    M_bar_amu: np.ndarray,
    theta_D_K: np.ndarray,
    V_a_A3: np.ndarray,
    gamma: np.ndarray,
    N_atoms: np.ndarray,
    T_K: float = 300.0,
    A: float = 3.04e-6,
) -> np.ndarray:
    """Vectorised Slack κ_L (W/m/K). Inputs in mixed units (amu, K, Å)."""
    delta_A = V_a_A3 ** (1.0 / 3.0)               # Å
    num = A * M_bar_amu * theta_D_K**3 * delta_A
    den = gamma**2 * N_atoms ** (2.0 / 3.0) * T_K
    return num / den


def slack_from_features(
    df_T0: pd.DataFrame, df_T1: pd.DataFrame, df_T2: pd.DataFrame,
    T_K: float = 300.0, A: float = 3.04e-6,
) -> pd.Series:
    """Compute Slack κ_L per material from the assembled feature tables."""
    m = df_T0.set_index("mp_id")[["MagpieData mean AtomicWeight"]].rename(
        columns={"MagpieData mean AtomicWeight": "M_bar"}
    )
    t1 = df_T1.set_index("mp_id")[["theta_D_K", "volume_per_atom", "n_atoms_primitive"]]
    t2 = df_T2.set_index("mp_id")[["gamma_G_300K"]]
    j = m.join(t1, how="inner").join(t2, how="inner")
    k = slack_kappa(
        M_bar_amu=j["M_bar"].values,
        theta_D_K=j["theta_D_K"].values,
        V_a_A3=j["volume_per_atom"].values,
        gamma=j["gamma_G_300K"].values,
        N_atoms=j["n_atoms_primitive"].values,
        T_K=T_K, A=A,
    )
    return pd.Series(k, index=j.index, name="kappa_slack_300K")
