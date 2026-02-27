"""
Regulator: Market Competition & Collusion Detection

A Python package for simulating market competition and detecting collusive behavior using machine learning.
"""

from regulator.agents.regulator import Regulator
from regulator.cartel.cartel_env import CartelEnv

__version__ = "0.1.0"
__author__ = "Bangyen Pham"
__email__ = "bangyenp@gmail.com"

__all__ = ["CartelEnv", "Regulator"]
