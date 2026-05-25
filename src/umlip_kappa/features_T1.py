"""Tier T1: harmonic uMLIP-derived features.

For each material × uMLIP we compute:
  • Relaxed cell (cell + atoms, F < fmax, σ_ii < smax).
  • Birch–Murnaghan equation of state -> B0, B'.
  • Elastic tensor via energy-strain Hessian -> bulk, shear, sound velocities.
  • Phonon DOS via Phonopy finite-displacement -> Θ_D, ⟨ω^n⟩, low-ω fraction.

Cost per material per uMLIP: ~1–10 CPU-min on GPU; ~30 min CPU-only.

All intermediate products are checkpointed to outputs/<mp_id>/T1_<umlip>.json
and the dynamical matrix is saved as Phonopy YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
# NumPy 2.0 dropped np.trapz; the replacement is np.trapezoid (added in 1.25).
# Restore np.trapz so all downstream call sites in this module keep working.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
try:
    from ase.filters import ExpCellFilter  # ASE ≥ 3.23
except ImportError:  # pragma: no cover
    from ase.constraints import ExpCellFilter  # older ASE
from ase.optimize import LBFGS
from ase.units import GPa, kB
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .calculators import make_calculator
from .io_utils import (
    get_logger,
    has_checkpoint,
    load_checkpoint,
    material_dir,
    save_checkpoint,
)

log = get_logger()
KCM_PER_MEV = 8.065543937  # 1 meV = 8.0655 cm^-1
HBAR_J_S = 1.054571817e-34
KB_J_PER_K = 1.380649e-23


# ===================================================================
# Data class for T1 feature vector
# ===================================================================
@dataclass
class T1Features:
    mp_id: str
    umlip: str
    n_atoms_primitive: int
    volume_per_atom: float            # Å^3
    density: float                    # g/cm^3
    # EOS
    B0_GPa: float                     # bulk modulus from Birch-Murnaghan
    Bp: float                         # B'
    E0_per_atom_eV: float
    # Elastic (Voigt-Reuss-Hill)
    K_VRH_GPa: float
    G_VRH_GPa: float
    poisson: float
    v_L: float                        # longitudinal sound velocity, m/s
    v_T: float                        # transverse
    v_m: float                        # mean
    # Phonon DOS moments
    omega_max_THz: float
    omega_mean_THz: float
    omega_rms_THz: float
    omega_cube_THz: float             # (⟨ω^3⟩)^(1/3)
    theta_D_K: float
    dos_low_freq_frac: float          # ω < 5 meV fraction of DOS
    has_imaginary: bool


# ===================================================================
# Helpers
# ===================================================================
def _relax(atoms: Atoms, calc: Calculator, cfg: dict) -> Atoms:
    """Relax cell + atoms with the requested optimiser/filter."""
    atoms = atoms.copy()
    atoms.calc = calc
    rcfg = cfg["relaxation"]
    if rcfg["filter"] == "FrechetCellFilter":
        ucf = FrechetCellFilter(atoms)
    else:
        ucf = ExpCellFilter(atoms)
    opt = LBFGS(ucf, logfile=None)
    opt.run(fmax=rcfg["fmax"], steps=rcfg["steps_max"])
    return atoms


def _scale_cell(atoms: Atoms, factor: float) -> Atoms:
    """Isotropic volume rescaling."""
    a = atoms.copy()
    a.set_cell(a.get_cell() * factor, scale_atoms=True)
    return a


def _birch_murnaghan(V: np.ndarray, V0: float, E0: float, B0: float, Bp: float) -> np.ndarray:
    """Birch-Murnaghan 3rd-order EOS, E(V)."""
    eta = (V0 / V) ** (2 / 3) - 1.0
    return E0 + 9 * V0 * B0 / 16 * (eta**3 * Bp + eta**2 * (6 - 4 * (V0 / V) ** (2 / 3)))


def _fit_eos(volumes: np.ndarray, energies: np.ndarray) -> tuple[float, float, float, float]:
    """Fit BM3 EOS. Returns V0 (Å^3), E0 (eV), B0 (eV/Å^3), B'."""
    from scipy.optimize import curve_fit

    # initial guesses from polynomial
    p = np.polyfit(volumes, energies, 2)
    V0_guess = -p[1] / (2 * p[0])
    E0_guess = np.polyval(p, V0_guess)
    B0_guess = 2 * p[0] * V0_guess
    Bp_guess = 4.0
    popt, _ = curve_fit(
        _birch_murnaghan, volumes, energies,
        p0=[V0_guess, E0_guess, B0_guess, Bp_guess], maxfev=5000,
    )
    return popt[0], popt[1], popt[2], popt[3]


def compute_eos(atoms: Atoms, calc: Calculator, cfg: dict) -> dict:
    """Return EOS parameters and reference relaxed volume."""
    n_v = cfg["eos"]["n_volumes"]
    strain = cfg["eos"]["strain_range"]
    factors = np.linspace(1 - strain, 1 + strain, n_v) ** (1 / 3)
    V_ref = atoms.get_volume()

    Vs, Es = [], []
    for f in factors:
        a = _scale_cell(atoms, f)
        a.calc = calc
        Es.append(a.get_potential_energy())
        Vs.append(a.get_volume())
    Vs = np.asarray(Vs)
    Es = np.asarray(Es)
    V0, E0, B0_eVA3, Bp = _fit_eos(Vs, Es)
    B0_GPa = B0_eVA3 / GPa
    return {
        "V0": float(V0),
        "E0": float(E0),
        "B0_GPa": float(B0_GPa),
        "Bp": float(Bp),
        "V_ref": float(V_ref),
        "volumes": Vs.tolist(),
        "energies": Es.tolist(),
    }


# --------------------------------------------------------------- elastic
def compute_elastic_VRH(atoms: Atoms, calc: Calculator, cfg: dict) -> dict:
    """Energy-strain method for the 6×6 elastic tensor, then Voigt-Reuss-Hill.

    Uses pymatgen's deformation utility because it correctly handles symmetry.
    """
    from pymatgen.analysis.elasticity.elastic import ElasticTensor
    from pymatgen.analysis.elasticity.strain import DeformedStructureSet, Strain

    delta = cfg["elastic"]["delta"]
    struct = AseAtomsAdaptor.get_structure(atoms)
    # 24 deformations (6 Voigt × 4 magnitudes) by default
    # pymatgen 2024.11 dropped num_norm/num_shear kwargs.
    # Defaults are 4 normal strains (±0.5%, ±1%) and 4 shear strains (±3%, ±6%).
    defset = DeformedStructureSet(struct)

    # Compute equilibrium stress in pymatgen convention (GPa, sign-flipped)
    EV_PER_A3_TO_GPA = 160.21766208
    atoms_eq = AseAtomsAdaptor.get_atoms(struct)
    atoms_eq.calc = calc
    eq_stress = atoms_eq.get_stress(voigt=False) * EV_PER_A3_TO_GPA  # GPa, ASE convention matches pymatgen
    stresses = []
    strains = []
    for d in defset.deformations:
        deformed = d.apply_to_structure(struct)
        a = AseAtomsAdaptor.get_atoms(deformed)
        a.calc = calc
        s = a.get_stress(voigt=False) * EV_PER_A3_TO_GPA  # eV/Å^3 -> GPa
        stresses.append(s)
        strains.append(Strain.from_deformation(d))
    et = ElasticTensor.from_independent_strains(strains, stresses, eq_stress=eq_stress)

    K = float(et.k_vrh)  # GPa already
    G = float(et.g_vrh)
    nu = float(et.homogeneous_poisson)
    rho = struct.density * 1000.0  # kg/m^3

    v_L = np.sqrt((K * 1e9 + 4 / 3 * G * 1e9) / rho)
    v_T = np.sqrt(G * 1e9 / rho)
    v_m = (3 / (1 / v_L**3 + 2 / v_T**3)) ** (1 / 3)
    return {
        "K_VRH_GPa": K,
        "G_VRH_GPa": G,
        "poisson": nu,
        "v_L": float(v_L),
        "v_T": float(v_T),
        "v_m": float(v_m),
    }


# --------------------------------------------------------------- phonons
def _supercell_matrix(atoms: Atoms, cfg: dict) -> np.ndarray:
    """Pick a supercell matrix that gets ≥ n_min_atoms atoms."""
    base = np.diag(cfg["phonons"]["supercell"])
    n_at = len(atoms)
    n_min = cfg["phonons"]["n_min_atoms_supercell"]
    while np.linalg.det(base) * n_at < n_min:
        base = base + np.eye(3, dtype=int)
    return base.astype(int)


def compute_phonons(atoms: Atoms, calc: Calculator, cfg: dict, save_dir: Path | None = None):
    """Run phonopy finite-displacement on a uMLIP and return moments of the DOS.

    Returns a tuple (Phonopy, dos_features_dict).
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    sc = _supercell_matrix(atoms, cfg)
    pa = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell(),
    )
    phonon = Phonopy(pa, supercell_matrix=sc, primitive_matrix="auto")
    phonon.generate_displacements(distance=cfg["phonons"]["displacement"])
    sets = phonon.supercells_with_displacements

    forces = []
    for sup in sets:
        a = Atoms(
            symbols=sup.symbols,
            positions=sup.positions,
            cell=sup.cell,
            pbc=True,
        )
        a.calc = calc
        f = a.get_forces()
        forces.append(f)
    phonon.forces = forces
    phonon.produce_force_constants()
    if cfg["phonons"]["symmetrise"]:
        phonon.symmetrize_force_constants()

    mesh = cfg["phonons"]["q_mesh"]
    phonon.run_mesh(mesh, with_eigenvectors=False)
    phonon.run_total_dos(freq_min=-5.0, freq_max=None)  # THz
    tdos = phonon.total_dos
    freqs = tdos.frequency_points  # THz
    dos = tdos.dos

    pos = freqs > 0
    freqs_p = freqs[pos]
    dos_p = dos[pos]
    norm = np.trapz(dos_p, freqs_p)
    if norm <= 0:
        norm = 1.0
    dos_p = dos_p / norm

    omega_mean = float(np.trapz(freqs_p * dos_p, freqs_p))
    omega_rms = float(np.sqrt(np.trapz(freqs_p**2 * dos_p, freqs_p)))
    omega_cube = float(np.trapz(freqs_p**3 * dos_p, freqs_p) ** (1 / 3))
    omega_max = float(freqs_p.max())

    # Debye T from <ω^2>: kB Θ_D = ℏ ω_D where ω_D is fitted to first 2 moments
    # of the acoustic part. Cheap approximation: ω_D ≈ √(5/3) ω_rms.
    omega_D_THz = np.sqrt(5 / 3) * omega_rms
    theta_D = float(HBAR_J_S * 2 * np.pi * omega_D_THz * 1e12 / KB_J_PER_K)

    # Low-frequency DOS fraction below 5 meV (≈ 1.21 THz)
    f_cut_THz = 5.0 / 4.1357  # meV -> THz
    low_mask = freqs_p < f_cut_THz
    low_frac = float(np.trapz(dos_p[low_mask], freqs_p[low_mask]))

    # Tolerance set to -0.1 THz (~0.4 meV) to absorb finite-difference noise
    # near Γ in the acoustic branches. Real soft modes are typically << -1 THz.
    has_imag = bool((freqs < -0.1).any())

    if save_dir is not None:
        phonon.save(filename=str(save_dir / "phonopy.yaml"))

    return phonon, {
        "omega_max_THz": omega_max,
        "omega_mean_THz": omega_mean,
        "omega_rms_THz": omega_rms,
        "omega_cube_THz": omega_cube,
        "theta_D_K": theta_D,
        "dos_low_freq_frac": low_frac,
        "has_imaginary": has_imag,
    }


# ===================================================================
# Top-level driver
# ===================================================================
def compute_T1_for_material(
    mp_id: str,
    structure_dict: dict,
    umlip: str,
    cfg: dict,
    force: bool = False,
) -> dict[str, Any] | None:
    """Compute the full T1 feature row for one material × one uMLIP."""
    tag = f"T1_{umlip}"
    if has_checkpoint(cfg["paths"]["output_dir"], mp_id, tag) and not force:
        log.info("[%s][%s] cached", mp_id, umlip)
        return load_checkpoint(cfg["paths"]["output_dir"], mp_id, tag)

    try:
        struct = Structure.from_dict(structure_dict)
        atoms = AseAtomsAdaptor.get_atoms(struct)
        calc = make_calculator(umlip, cfg["umlips"].get(umlip, {}))

        atoms = _relax(atoms, calc, cfg)
        eos = compute_eos(atoms, calc, cfg)
        ela = compute_elastic_VRH(atoms, calc, cfg)
        sdir = material_dir(cfg["paths"]["output_dir"], mp_id)
        _, dos = compute_phonons(atoms, calc, cfg, save_dir=sdir)

        struct_rel = AseAtomsAdaptor.get_structure(atoms)
        feats = T1Features(
            mp_id=mp_id,
            umlip=umlip,
            n_atoms_primitive=len(atoms),
            volume_per_atom=float(atoms.get_volume() / len(atoms)),
            density=float(struct_rel.density),
            B0_GPa=eos["B0_GPa"],
            Bp=eos["Bp"],
            E0_per_atom_eV=eos["E0"] / len(atoms),
            **ela,
            **dos,
        )
        payload = feats.__dict__
        save_checkpoint(cfg["paths"]["output_dir"], mp_id, tag, payload)
        log.info("[%s][%s] done", mp_id, umlip)
        return payload
    except Exception as e:  # never let one bad material kill the whole run
        log.exception("[%s][%s] FAILED: %s", mp_id, umlip, e)
        save_checkpoint(
            cfg["paths"]["output_dir"], mp_id, f"{tag}_ERROR",
            {"error": str(e), "type": type(e).__name__},
        )
        return None
