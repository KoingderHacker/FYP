"""Shared pytest setup.

The weather-only engine does not use torch/chronos, but the legacy wind and
demand modules import them at module load. When they are not installed (e.g.
a lightweight CI box) lightweight stand-ins are registered so the agents can
still be imported and the weather-only path exercised.
"""
import importlib.util
import sys
import types

import scipy.stats  # noqa: F401  (import real scipy/sklearn before any stub)
import sklearn.metrics  # noqa: F401

if importlib.util.find_spec("torch") is None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch
if importlib.util.find_spec("chronos") is None:
    chronos = types.ModuleType("chronos")
    chronos.Chronos2Pipeline = object
    sys.modules["chronos"] = chronos
