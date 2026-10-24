"""Policy lookup by key.

Keys from the starter kit's ``POLICY_MAP`` keep working, so ``pid_water`` and
``all_off`` resolve as before. Our own controllers are added here rather than
by editing ``evaluate.py``, which keeps the starter kit a clean upstream.
"""

from __future__ import annotations

from typing import Any

# Our controllers, imported lazily so that listing keys does not import numpy
# or the simulator.
_OURS: dict[str, tuple[str, str]] = {
    "occ": ("hvaclab.policies.occupancy_gated", "OccupancyGatedPolicy"),
}


def available() -> list[str]:
    """Every key that ``resolve_policy`` accepts."""
    from smart_buildings.evaluate import POLICY_MAP

    return sorted(set(POLICY_MAP) | set(_OURS))


def resolve_policy(key: str) -> type[Any]:
    """Return the policy class for ``key``."""
    if key in _OURS:
        module_name, class_name = _OURS[key]
        import importlib

        return getattr(importlib.import_module(module_name), class_name)

    from smart_buildings.evaluate import POLICY_MAP

    if key in POLICY_MAP:
        return POLICY_MAP[key]

    raise KeyError(f"unknown policy {key!r}; available: {available()}")
