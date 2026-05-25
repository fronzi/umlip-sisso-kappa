"""umlip_kappa: descriptors for lattice thermal conductivity from universal MLIPs."""
from importlib.metadata import version as _version

try:
    __version__ = _version("umlip_kappa")
except Exception:  # pragma: no cover
    __version__ = "0.0.dev0"
