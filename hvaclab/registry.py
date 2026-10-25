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
    # First attempt, kept because it is the measured negative result the
    # current design is a response to. See deadline_band's module docstring.
    "occ": ("hvaclab.policies.occupancy_gated", "OccupancyGatedPolicy"),
    "band": ("hvaclab.policies.deadline_band", "DeadlineBandPolicy"),
    "fan": ("hvaclab.policies.fan_economy", "FanEconomyPolicy"),
}

# Parameter variants, for ablation. The right setback depth is climate
# dependent -- `fixed_setpoint`, which sets back not at all, beats `pid_water`
# on `fairbanks_1` -- so it is measured rather than assumed. Each entry is
# (policy key, constructor kwargs).
#
# `band_flat` is the control: zero setback, so it isolates how much of any
# change comes from the band logic rather than from the setback itself.
_VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "band_flat": ("band", {"setback_k": 0.0, "setup_k": 0.0}),
    "band_s3": ("band", {"setback_k": 3.0, "setup_k": 3.0}),
    "band_s5": ("band", {"setback_k": 5.0, "setup_k": 3.0}),
    "band_s8": ("band", {"setback_k": 8.0, "setup_k": 5.0}),
    "band_s5_lead6": ("band", {"setback_k": 5.0, "setup_k": 3.0,
                               "lead_hours": 6.0}),
    "band_s8_lead6": ("band", {"setback_k": 8.0, "setup_k": 5.0,
                               "lead_hours": 6.0}),
    # Cooling is the cheapest thermal kilowatt-hour here (0.0018 per kWh at
    # COP 3.5, against 0.0034 for gas heat), so the hot side can afford a
    # deeper setback than the cold side. The shipped eco band uses 5 K down
    # and only 1 K up; this inverts that asymmetry to test it.
    "band_asym": ("band", {"setback_k": 2.0, "setup_k": 8.0}),

    # Fan-work variants. `fan_ctrl` reproduces pid's own pressure law, so it
    # isolates the effect of the heating clamp from everything else in the
    # class; any difference between it and the matching `band_*` entry is a
    # bug, not a result.
    "fan_ctrl": ("fan", {"heating_pressure": None,
                         "cooling_pressure_gain": 1.0}),
    "fan_heat_floor": ("fan", {"heating_pressure": -1.0,
                               "cooling_pressure_gain": 1.0}),
    "fan_both": ("fan", {"heating_pressure": -1.0,
                         "cooling_pressure_gain": 0.6}),
    "fan_cool_only": ("fan", {"heating_pressure": None,
                              "cooling_pressure_gain": 0.6}),
}


def available() -> list[str]:
    """Every key that ``resolve_policy`` accepts."""
    from smart_buildings.evaluate import POLICY_MAP

    return sorted(set(POLICY_MAP) | set(_OURS) | set(_VARIANTS))


def _load(key: str) -> type[Any]:
    module_name, class_name = _OURS[key]
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def resolve_policy(key: str) -> type[Any]:
    """Return a zero-extra-argument policy class for ``key``.

    A variant is returned as a subclass with its parameters bound, rather than
    as a factory, because the starter kit's ``initialize_policy`` instantiates
    whatever it is handed as ``cls(building_meta_data, action_spec)`` and a
    partial would not survive that.
    """
    if key in _VARIANTS:
        base_key, kwargs = _VARIANTS[key]
        base = _load(base_key)

        class Variant(base):  # type: ignore[valid-type,misc]
            _variant_name = key
            _variant_kwargs = kwargs

            def __init__(self, building_meta_data, action_spec):
                super().__init__(building_meta_data, action_spec, **kwargs)

            @property
            def name(self) -> str:
                return key

        Variant.__name__ = f"Variant_{key}"
        Variant.__qualname__ = Variant.__name__
        return Variant

    if key in _OURS:
        return _load(key)

    from smart_buildings.evaluate import POLICY_MAP

    if key in POLICY_MAP:
        return POLICY_MAP[key]

    raise KeyError(f"unknown policy {key!r}; available: {available()}")
