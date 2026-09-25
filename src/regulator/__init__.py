"""
Regulator: can label-free screens detect algorithmic collusion?

Simulated oligopoly markets with competing, colluding and Q-learning firms,
plus collusion screens and regulators to evaluate against them.
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
