"""Tier T2: anharmonic uMLIP-derived features.

* Quasi-harmonic mode-Grüneisen parameters
      γ_qs = -(V/ω_qs) ∂ω_qs/∂V
  evaluated at 5 volumes ±5 % around the relaxed equilibrium.

* Cv-weighted mode-Grüneisen at 300 K:
      γ_G(T) = Σ C_v(ω_qs,T) γ_qs / Σ C_v(ω_qs,T)

* Distribution moments: ⟨γ⟩, ⟨γ²⟩, Var(γ), |γ|_max.

* Knoop anharmonicity score
      σ^A = √( ⟨|F − F_harm|²⟩ / ⟨|F|²⟩ )
  from a short NVT trajectory, sampled every `sample_every` steps.

Cost per material per uMLIP: ~10–60 CPU-min on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .calculators import make_calculator
from .features_T1 import _relax, _scale_cell, _supercell_matrix, compute_phonons
from .io_utils import (
    get_logger,
    has_checkpoint,
    load_checkpoint,
    material_dir,
    save_checkpoint,
)

log = get_logger()
HBAR_J_S = 1.054571817e-34
KB_J_PER_K = 1.380649e-23


@dataclass
class T2Features:
    mp_id: str
    umlip: str
    # Grüneisen statistics (all scalar)
    gamma_mean: float
    gamma_var: float
    gamma_sq_mean: float           # ⟨γ²⟩
    gamma_max_abs: float
    gamma_G_300K: float            # heat-capacity-weighted at 300 K
    gamma_low_omega_mean: float    # γ averaged over modes < 5 meV
    # Anharmonicity
    sigma_A_300K: float
    F_rms_eV_per_A: float          # ⟨|F|²⟩^(1/2)
    F_anh_rms_eV_per_A: float      # ⟨|F−F_harm|²⟩^(1/2)


# ===================================================================
# QHA mode-Grüneisen
# ===================================================================
def _phonon_frequencies_on_mesh(phonon, mesh):
    """Return frequencies (THz) for each q on `mesh`, shape (Nq, Nb)."""
    phonon.run_mesh(mesh, with_eigenvectors=False)
    md = phonon.mesh
    return md.frequencies, md.weights  # (Nq, Nb), (Nq,)


def compute_QHA_gruneisen(
    atoms: Atoms, calc: Calculator, cfg: dict, save_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute mode-Grüneisen by 3-point central difference in ln V.

    Returns a dict with arrays `frequencies` (THz, ref volume), `gamma` (dim-less),
    `weights` (q-point weights), shape (Nq, Nb).
    """
    qcfg = cfg["qha"]
    strain = qcfg["strain_range"]
    n_v = qcfg["n_volumes"]
    factors = np.linspace(1 - strain, 1 + strain, n_v) ** (1 / 3)
    if n_v % 2 == 0:
        raise ValueError("qha.n_volumes must be odd to centre at V_ref")
    mid = n_v // 2

    mesh = cfg["phonons"]["q_mesh"]

    freq_list = []
    V_list = []
    phonon_ref = None
    weights = None

    for i, f in enumerate(factors):
        a = _scale_cell(atoms, f)
        a.calc = calc
        phon, _ = compute_phonons(a, calc, cfg, save_dir=None)
        freqs, w = _phonon_frequencies_on_mesh(phon, mesh)
        freq_list.append(freqs)  # THz
        V_list.append(a.get_volume())
        if i == mid:
            phonon_ref = phon
            weights = w

    freq_arr = np.stack(freq_list)        # (n_v, Nq, Nb)
    V_arr = np.asarray(V_list)
    lnV = np.log(V_arr)
    ln_omega = np.log(np.clip(freq_arr, 1e-6, None))

    # Central differences: γ = - d ln ω / d ln V
    # Use ordinary least squares per (q,b) for noise robustness.
    Nq, Nb = freq_arr.shape[1:]
    gamma = np.zeros((Nq, Nb))
    for q in range(Nq):
        for b in range(Nb):
            slope, _ = np.polyfit(lnV, ln_omega[:, q, b], 1)
            gamma[q, b] = -slope

    freqs_ref = freq_list[mid]
    out = {
        "frequencies_THz": freqs_ref,    # shape (Nq, Nb)
        "gamma": gamma,
        "weights": weights,
    }
    if save_dir is not None:
        np.savez(save_dir / "qha_gruneisen.npz", **out)
    return out


def _heat_capacity_per_mode(omega_THz: np.ndarray, T: float) -> np.ndarray:
    """Einstein/harmonic Cv per mode in kB units."""
    omega = 2 * np.pi * omega_THz * 1e12  # rad/s
    x = HBAR_J_S * omega / (KB_J_PER_K * max(T, 1e-6))
    x = np.clip(x, 1e-8, 50.0)
    return x**2 * np.exp(x) / (np.exp(x) - 1) ** 2  # in units of kB


def gruneisen_moments(qha: dict, T: float = 300.0, low_omega_cut_THz: float = 1.21) -> dict:
    """Scalar reductions of the mode-Grüneisen distribution."""
    freqs = qha["frequencies_THz"]  # (Nq, Nb)
    gamma = qha["gamma"]             # (Nq, Nb)
    w = qha["weights"]               # (Nq,)
    mask = freqs > 1e-3              # discard ω ≤ 0 (imaginary)

    g_flat = gamma[mask]
    f_flat = freqs[mask]
    w_q = np.broadcast_to(w[:, None], freqs.shape)[mask]

    # Plain stats (q-weighted)
    norm = w_q.sum()
    gamma_mean = float((g_flat * w_q).sum() / norm)
    gamma_sq_mean = float((g_flat**2 * w_q).sum() / norm)
    gamma_var = gamma_sq_mean - gamma_mean**2

    # Heat-capacity-weighted at T
    cv = _heat_capacity_per_mode(f_flat, T)
    cv_w = cv * w_q
    gamma_G = float((g_flat * cv_w).sum() / cv_w.sum())

    # Average γ over low-frequency modes
    low_mask = f_flat < low_omega_cut_THz
    if low_mask.any():
        gamma_low = float((g_flat[low_mask] * w_q[low_mask]).sum() / w_q[low_mask].sum())
    else:
        gamma_low = float("nan")

    return {
        "gamma_mean": gamma_mean,
        "gamma_var": float(gamma_var),
        "gamma_sq_mean": gamma_sq_mean,
        "gamma_max_abs": float(np.max(np.abs(g_flat))),
        "gamma_G_300K": gamma_G,
        "gamma_low_omega_mean": gamma_low,
    }


# ===================================================================
# Knoop σ^A from NVT trajectory
# ===================================================================
def _harmonic_forces(
    supercell: Atoms, equilibrium_positions: np.ndarray, force_constants: np.ndarray
) -> np.ndarray:
    """F_harm = - Φ · u, with u = r - r_eq (cell-image-corrected)."""
    n = len(supercell)
    cell = supercell.get_cell()
    u = supercell.get_positions() - equilibrium_positions
    # Minimum-image displacement
    u_frac = np.linalg.solve(cell.T, u.T).T
    u_frac -= np.round(u_frac)
    u = u_frac @ cell  # (n, 3)
    # Phi has shape (n, n, 3, 3); F_i = - Σ_j Φ_ij u_j
    F = -np.einsum("ijab,jb->ia", force_constants, u)
    return F


def compute_sigma_A(
    atoms: Atoms, calc: Calculator, cfg: dict, save_dir: Path | None = None,
) -> dict[str, float]:
    """Run NVT MD with the uMLIP, compare actual forces with harmonic estimate.

    The harmonic reference uses the force-constant matrix from a Phonopy
    finite-displacement calculation on a matched supercell.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    sccfg = cfg["sigma_a"]
    sc = _supercell_matrix(atoms, cfg)

    # 1) Build the harmonic reference: force constants on the same supercell
    pa = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell(),
    )
    phonon = Phonopy(pa, supercell_matrix=sc, primitive_matrix=np.eye(3))
    phonon.generate_displacements(distance=cfg["phonons"]["displacement"])
    forces = []
    for sup in phonon.supercells_with_displacements:
        a = Atoms(symbols=sup.symbols, positions=sup.positions, cell=sup.cell, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()
    if cfg["phonons"]["symmetrise"]:
        phonon.symmetrize_force_constants()
    fc = phonon.force_constants  # (n_sc, n_sc, 3, 3)
    eq_super = Atoms(
        symbols=phonon.supercell.symbols,
        positions=phonon.supercell.positions,
        cell=phonon.supercell.cell,
        pbc=True,
    )
    eq_positions = eq_super.get_positions()

    # 2) Run NVT
    md_super = eq_super.copy()
    md_super.calc = calc
    MaxwellBoltzmannDistribution(md_super, temperature_K=sccfg["T_md"])
    dyn = Langevin(
        md_super,
        timestep=sccfg["dt_fs"] * units.fs,
        temperature_K=sccfg["T_md"],
        friction=sccfg["friction_fs"] / units.fs,
    )

    # Equilibration
    dyn.run(sccfg["n_steps_equilibration"])

    # Production: sample forces and harmonic forces
    F_actual = []
    F_harm = []
    every = sccfg["sample_every"]
    n_prod = sccfg["n_steps_production"]

    def _sample():
        F = md_super.get_forces()
        Fh = _harmonic_forces(md_super, eq_positions, fc)
        F_actual.append(F.copy())
        F_harm.append(Fh.copy())

    sampler_counter = {"i": 0}

    def _obs():
        sampler_counter["i"] += 1
        if sampler_counter["i"] % every == 0:
            _sample()

    dyn.attach(_obs, interval=1)
    dyn.run(n_prod)

    F_actual = np.stack(F_actual)   # (N_s, n_sc, 3)
    F_harm = np.stack(F_harm)
    diff = F_actual - F_harm
    F_rms = float(np.sqrt((F_actual**2).mean()))
    F_anh_rms = float(np.sqrt((diff**2).mean()))
    sigmaA = float(F_anh_rms / max(F_rms, 1e-12))

    if save_dir is not None:
        np.savez(
            save_dir / "sigmaA.npz",
            F_actual=F_actual, F_harm=F_harm,
            sigmaA=sigmaA, F_rms=F_rms, F_anh_rms=F_anh_rms,
        )

    return {"sigma_A_300K": sigmaA, "F_rms_eV_per_A": F_rms, "F_anh_rms_eV_per_A": F_anh_rms}


# ===================================================================
# Top-level driver
# ===================================================================
def compute_T2_for_material(
    mp_id: str,
    structure_dict: dict,
    umlip: str,
    cfg: dict,
    force: bool = False,
) -> dict[str, Any] | None:
    tag = f"T2_{umlip}"
    if has_checkpoint(cfg["paths"]["output_dir"], mp_id, tag) and not force:
        log.info("[%s][%s][T2] cached", mp_id, umlip)
        return load_checkpoint(cfg["paths"]["output_dir"], mp_id, tag)

    try:
        struct = Structure.from_dict(structure_dict)
        atoms = AseAtomsAdaptor.get_atoms(struct)
        calc = make_calculator(umlip, cfg["umlips"].get(umlip, {}))

        atoms = _relax(atoms, calc, cfg)
        sdir = material_dir(cfg["paths"]["output_dir"], mp_id)

        qha = compute_QHA_gruneisen(atoms, calc, cfg, save_dir=sdir)
        gmom = gruneisen_moments(qha, T=cfg["qha"]["T_ref"])
        sig = compute_sigma_A(atoms, calc, cfg, save_dir=sdir)

        feats = T2Features(mp_id=mp_id, umlip=umlip, **gmom, **sig)
        payload = feats.__dict__
        save_checkpoint(cfg["paths"]["output_dir"], mp_id, tag, payload)
        log.info("[%s][%s][T2] done  γ_G=%.2f σA=%.3f", mp_id, umlip,
                 payload["gamma_G_300K"], payload["sigma_A_300K"])
        return payload
    except Exception as e:
        log.exception("[%s][%s][T2] FAILED: %s", mp_id, umlip, e)
        save_checkpoint(
            cfg["paths"]["output_dir"], mp_id, f"{tag}_ERROR",
            {"error": str(e), "type": type(e).__name__},
        )
        return None
