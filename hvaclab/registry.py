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

# Parameter variants, for ablation. Each entry is (policy key, kwargs).
#
# ---------------------------------------------------------------------------
# MEASURED: the setback hypothesis is wrong. Deeper is worse.
# ---------------------------------------------------------------------------
# Full 8,640-step episodes on fairbanks_1 / singapore_6 / phoenix_4, mean of
# the three (lower is better):
#
#   band_flat  (setback 0)      0.12321    <- best
#   fixed_setpoint             0.12575
#   pid_water                  0.13478
#   band_s5                    0.16017
#   band_s8                    0.18320
#   band_asym                  0.19597
#
# Holding the comfort band around the clock beats every setback depth tried,
# and beats pid_water by 8.6%. The decomposition says why: band_flat's comfort
# term is 0.0347 against pid_water's 0.0603, bought with energy 0.0411 against
# 0.0371 and carbon 0.0474 against 0.0374.
#
# Which is exactly the 35:1 price ratio asserting itself. Comfort is worth far
# more than energy *while it is scored*, and a building held in band overnight
# starts the morning with no deviation to recover -- so it never pays the
# recovery penalty, and the recovery penalty is larger than the holding cost.
# Steady beats cycling, as fixed_setpoint already hinted by beating pid_water.
#
# So the setback entries are kept as the record of a refuted hypothesis, and
# everything built after this point starts from the flat band.
_SETBACK_ABLATION: dict[str, tuple[str, dict[str, Any]]] = {
    "band_flat": ("band", {"setback_k": 0.0, "setup_k": 0.0}),
    "band_s3": ("band", {"setback_k": 3.0, "setup_k": 3.0}),
    "band_s5": ("band", {"setback_k": 5.0, "setup_k": 3.0}),
    "band_s8": ("band", {"setback_k": 8.0, "setup_k": 5.0}),
    "band_asym": ("band", {"setback_k": 2.0, "setup_k": 8.0}),
}

# ---------------------------------------------------------------------------
# Built on the flat band. `FLAT` is spread into each so none of them silently
# inherits the refuted 5 K default.
# ---------------------------------------------------------------------------
FLAT: dict[str, Any] = {"setback_k": 0.0, "setup_k": 0.0}

_VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    **_SETBACK_ABLATION,

    # The new reference point: band_flat under its own name, so later tables
    # do not have to explain why the control is called "flat".
    "v0": ("band", {**FLAT}),

    # Aim inside the scored band rather than at its edge. The penalty is flat
    # zero anywhere inside, so the edge buys nothing and risks everything:
    # pid idles the moment a zone is in band, and a zone parked on the edge
    # drifts straight across it.
    "v_m05": ("band", {**FLAT, "comfort_margin_k": 0.5}),
    "v_m10": ("band", {**FLAT, "comfort_margin_k": 1.0}),

    # Fan work. band_flat buys its comfort with energy and carbon, so the
    # pressure law is where that bill can be cut. `v_fan_ctrl` reproduces
    # pid's own law, so any gap between it and `v0` is a bug, not a result.
    "v_fan_ctrl": ("fan", {**FLAT, "heating_pressure": None,
                           "cooling_pressure_gain": 1.0}),
    "v_fan_heat": ("fan", {**FLAT, "heating_pressure": -1.0,
                           "cooling_pressure_gain": 1.0}),
    "v_fan_cool": ("fan", {**FLAT, "heating_pressure": None,
                           "cooling_pressure_gain": 0.6}),
    "v_fan": ("fan", {**FLAT, "heating_pressure": -1.0,
                      "cooling_pressure_gain": 0.6}),
    # Quiet fan paired with a hot loop. Heat delivered is airflow times the
    # coil's temperature rise, so taking the flow away without raising the
    # water temperature would just under-deliver.
    "v_fan_hw": ("fan", {**FLAT, "heating_pressure": -1.0,
                         "cooling_pressure_gain": 0.6,
                         "heating_water_floor": 0.5}),
    # Economise only while the shortfall is not being charged for.
    "v_fan_urg": ("fan", {**FLAT, "heating_pressure": -1.0,
                          "cooling_pressure_gain": 0.6,
                          "heating_water_floor": 0.5,
                          "urgent_deviation_k": 1.0}),
    # The two leading ideas together.
    "v_fan_m05": ("fan", {**FLAT, "comfort_margin_k": 0.5,
                          "heating_pressure": -1.0,
                          "cooling_pressure_gain": 0.6,
                          "heating_water_floor": 0.5}),
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
