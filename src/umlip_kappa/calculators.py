"""Factory for ASE calculators wrapping universal MLIPs.

Each branch is a *lazy import* so that the user only needs to install the
specific uMLIPs they want to test.
"""
from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator


SUPPORTED = ("mace-mp-0", "orb-v3", "mattersim-v1", "chgnet", "sevennet")


def make_calculator(name: str, options: dict[str, Any] | None = None) -> Calculator:
    """Return an ASE calculator for the requested uMLIP.

    Parameters
    ----------
    name : str
        One of `SUPPORTED`.
    options : dict, optional
        Model-specific options (size, device, dtype, ...) from
        `cfg['umlips'][name]`.

    Notes
    -----
    Phonon calculations need double precision; we pass float64 by default
    when the model exposes a `default_dtype` argument.
    """
    options = options or {}
    name = name.lower()

    if name == "mace-mp-0":
        from mace.calculators import mace_mp

        return mace_mp(
            model=options.get("size", "medium"),
            device=options.get("device", "cuda"),
            default_dtype=options.get("default_dtype", "float64"),
        )

    if name == "orb-v3":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        model_fn = getattr(pretrained, options.get("name", "orb_v3_conservative_inf_omat"))
        device = options.get("device", "cuda")
        orbff = model_fn(device=device)
        return ORBCalculator(orbff, device=device)

    if name == "mattersim-v1":
        from mattersim.forcefield import MatterSimCalculator

        return MatterSimCalculator(
            load_path=options.get("checkpoint", "mattersim-v1.0.0-5M"),
            device=options.get("device", "cuda"),
        )

    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator()

    if name == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        return SevenNetCalculator(
            model=options.get("model", "7net-mf-ompa"),
            device=options.get("device", "cuda"),
        )

    raise ValueError(f"Unknown uMLIP '{name}'. Supported: {SUPPORTED}")
