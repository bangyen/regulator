"""
Regulator: Market Competition & Collusion Detection

A Python package for simulating market competition and detecting collusive behavior using machine learning.
"""

from importlib import metadata as _metadata

from regulator.agents.regulator import Regulator
from regulator.cartel.cartel_env import CartelEnv

# Single source of truth is pyproject.toml
try:
    __version__ = _metadata.version("regulator")
except _metadata.PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"
__author__ = "Bangyen Pham"
__email__ = "bangyen99@gmail.com"

__all__ = ["CartelEnv", "Regulator"]
