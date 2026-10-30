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
    "zone": ("hvaclab.policies.zone_aware", "ZoneAwarePolicy"),
    "zonepi": ("hvaclab.policies.zone_pi", "ZonePIPolicy"),
    "hold": ("hvaclab.policies.cooling_hold", "CoolingHoldPolicy"),
    "heat": ("hvaclab.policies.heat_delivery", "HeatDeliveryPolicy"),
    "heat_sync": ("hvaclab.policies.coordinated_heat", "CoordinatedHeatPolicy"),
    "delay": ("hvaclab.policies.delay_compensated", "DelayCompensatedPolicy"),
    "scheduled_hold": ("hvaclab.policies.scheduled_hold", "ScheduledHoldPolicy"),
    "guarded_pump": ("hvaclab.policies.guarded_pump", "GuardedPumpPolicy"),
    "heating_pump": ("hvaclab.policies.heating_pump", "HeatingPumpPolicy"),
    "cooling_pressure": ("hvaclab.policies.cooling_pressure", "CoolingPressurePolicy"),
    "flow_balanced_cooling": ("hvaclab.policies.flow_balanced_cooling", "FlowBalancedCoolingPolicy"),
    "geometry_pi": ("hvaclab.policies.geometry_pi", "GeometryPIPolicy"),
    "split_geometry_pi": ("hvaclab.policies.split_geometry_pi", "SplitGeometryPIPolicy"),
    "reheat_curve": ("hvaclab.policies.reheat_curve", "ReheatCurvePolicy"),
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

# ---------------------------------------------------------------------------
# MEASURED: the fan clamp is the single biggest win so far.
# ---------------------------------------------------------------------------
# Same three scenarios, mean of the three:
#
#   v_fan       0.04967   flat band + heating fan at its floor + cooling x0.6
#   v_fan_heat  0.05226   heating clamp only
#   v_fan_hw    0.05362   v_fan plus a hot-water floor
#   v_fan_urg   0.07357   v_fan, economy suspended while a zone is out of band
#   v_fan_m05   0.07593   v_fan plus a 0.5 K margin inside the band
#   v_fan_ctrl  0.12321   identical to band_flat, to the digit
#   pid_water   0.13478
#   v_m05       0.23605   margin alone
#
# v_fan is 63% below pid_water, and on fairbanks_1 it is 0.0797 against
# band_flat's 0.264 -- 70% better, and 47% better than fixed_setpoint's 0.149.
#
# It improves ALL THREE terms at once: comfort 0.0347 -> 0.0237, energy
# 0.0411 -> 0.0126, carbon 0.0474 -> 0.0134. Comfort improving alongside the
# energy saving is the tell that this is not a trade. Less static pressure is
# less airflow, and with recirculation capped at 0.70 that is also less
# ambient air dragged in and heated from -42 C -- which in Fairbanks is most
# of the load. Cutting the flow cuts the load, so the coils keep up where
# before they could not.
#
# Two guesses refuted at the same time, both worth keeping as results:
#
#  * `heating_water_floor` makes things worse (carbon 0.0134 -> 0.0162). Once
#    the flow is down, pid_water's own demand reset is already hot enough, and
#    forcing it hotter just burns gas.
#  * `urgent_deviation_k` makes things worse, which is counterintuitive --
#    comfort is worth 35x energy, so running the fan hard while a zone is out
#    of band ought to pay. It does not, because the fan is what *causes* the
#    load here. Spending the expensive fuel to deliver heat drags in more cold
#    air than the extra flow delivers.
#  * `comfort_margin_k` is clearly harmful (0.236 against 0.123 for a flat
#    band). Narrowing a 3 K band to 2 K leaves pid, whose loop saturates at
#    1/kp = 2 K, oscillating across the real edge: comfort gets WORSE, not
#    just more expensive.
#
# `_W` tunes around v_fan, one lever at a time.
_W: dict[str, tuple[str, dict[str, Any]]] = {}
_FAN: dict[str, Any] = {**FLAT, "heating_pressure": -1.0,
                        "cooling_pressure_gain": 0.6}
for _gain in (0.3, 0.45, 0.8, 1.0):
    _W[f"w_cool{int(_gain * 100):03d}"] = (
        "fan", {**_FAN, "cooling_pressure_gain": _gain})
_W["w_reheat05"] = ("fan", {**_FAN, "reheat_full_k": 0.5})
_W["w_reheat10"] = ("fan", {**_FAN, "reheat_full_k": 1.0})
_W["w_pump00"] = ("fan", {**_FAN, "pump_pressure": 0.0})
_W["w_pump-05"] = ("fan", {**_FAN, "pump_pressure": -0.5})

# Airflow is the load, not just the bill -- that is the lesson of v_fan. So
# the other levers that set airflow are worth the same treatment. These pass
# straight through to PIDPolicy's constructor.
#
# zone_damper_full_k is the zone excursion at which a damper is commanded
# fully open; pid's default is 2.0 K, matched to its own loop saturating at
# 1/kp = 2 K. Raising it keeps dampers closer to their 10% floor for the same
# excursion, which is less flow for the same demand.
for _dk in (4.0, 8.0):
    _W[f"w_damper{int(_dk)}"] = ("fan", {**_FAN, "zone_damper_full_k": _dk})
# A gentler loop reaches full supply-air effort later, so it spends less of
# the episode at saturation.
_W["w_kp025"] = ("fan", {**_FAN, "kp": 0.25})

# ---------------------------------------------------------------------------
# MEASURED, round 2. Small gains, and one more refutation.
# ---------------------------------------------------------------------------
#   w_reheat05  0.04730   v_fan + reheat valve saturating at 0.5 K
#   w_reheat10  0.04821
#   w_cool030   0.04870   cooling pressure x0.30
#   w_cool045   0.04887
#   v_fan       0.04967   (reference)
#   w_pump00    0.05090   capping the pump is WORSE -- refuted
#   w_cool080   0.05100
#
# Opening the reheat valve sooner helps, and it helps where predicted:
# fairbanks_1 0.0797 -> 0.0725, with singapore and phoenix unmoved. A deeper
# cooling clamp helps phoenix (0.0430 -> 0.0382) and costs singapore
# (0.0263 -> 0.0282), so it is a climate trade rather than a free gain.
#
# These are 5% effects against the fan clamp's 63%, which is the scale at
# which tuning on three scenarios starts fitting the three scenarios. So the
# combination is carried to the full 30 and to the holdout sets rather than
# tuned further here.
_W["w_combo"] = ("fan", {**_FAN, "reheat_full_k": 0.5,
                         "cooling_pressure_gain": 0.3})

# ---------------------------------------------------------------------------
# MEASURED on the held-out sets. The headline does not transfer at full size.
# ---------------------------------------------------------------------------
#                        pid_water    w_reheat05
#   shipped 30 scenarios   0.05496      0.03682    -33.0%
#   dev   (20 scenarios)   0.04630      0.04367     -5.7%
#   test  ( 8 scenarios)   0.11253      0.09748    -13.4%
#
# The gain is real everywhere but much smaller off the published set, and the
# reason is visible in the worst-case column: pid_water's worst shipped
# scenario is 0.214, its worst dev scenario 0.089. The published 30 contain
# catastrophic failures -- fairbanks_1, phoenix_4, phoenix_5, singapore_4 --
# and this change is mostly repairing those. Where the baseline does not fail
# badly there is less to win.
#
# On dev the decomposition also differs: energy and carbon still fall sharply
# (0.0073 -> 0.0053, 0.0071 -> 0.0052) but comfort rises slightly (0.0318 ->
# 0.0332), where on the published set all three improved. Phoenix is the only
# station that regresses (+2.9%), which points at the cooling clamp rather
# than the heating clamp as the term that trades comfort away.
#
# These test that. dev is the legitimate place to tune; test is not.
_X: dict[str, tuple[str, dict[str, Any]]] = {
    # Heating clamp only -- isolates the cooling clamp as the suspect.
    "x_heatonly": ("fan", {**FLAT, "heating_pressure": -1.0,
                           "cooling_pressure_gain": 1.0,
                           "reheat_full_k": 0.5}),
    # A softer cooling clamp rather than none.
    "x_cool08": ("fan", {**FLAT, "heating_pressure": -1.0,
                         "cooling_pressure_gain": 0.8,
                         "reheat_full_k": 0.5}),
}
_W.update(_X)

# ---------------------------------------------------------------------------
# Per-zone control. After the fan clamp, comfort is 70-75% of the remaining
# loss on all three sets, so this is where the rest of the score is.
#
# The shipped chain drives each damper from abs(zone_error), which never asks
# whether the air about to arrive is on the useful side of that zone's
# temperature. A handler heating for its average zone therefore opens the
# damper of a zone that is already too hot, in proportion to how hot it is.
#
# `z_ctrl` disables the gate, so it must reproduce x_heatonly exactly; any
# difference between them is a bug rather than a result.
_Z: dict[str, tuple[str, dict[str, Any]]] = {
    "z_ctrl": ("zone", {**_FAN, "cooling_pressure_gain": 1.0,
                        "reheat_full_k": 0.5,
                        "require_useful_air": False}),
    "z_gate": ("zone", {**_FAN, "cooling_pressure_gain": 1.0,
                        "reheat_full_k": 0.5,
                        "require_useful_air": True,
                        "useful_air_margin_k": 0.5}),
    "z_gate0": ("zone", {**_FAN, "cooling_pressure_gain": 1.0,
                         "reheat_full_k": 0.5,
                         "require_useful_air": True,
                         "useful_air_margin_k": 0.0}),
}
_W.update(_Z)

# ---------------------------------------------------------------------------
# Comfort levers. Comfort is 70-75% of what is left after the fan clamp, so
# the remaining score is here and not in energy.
#
# The refuted damper gate is what motivates the first two. It failed because
# shutting a damper leaves a zone's 10 W/m2 of internal gain with no way out,
# so the zone drifts past where the supply air would have held it. If more
# mixing is what helps, then dampers should open SOONER, not later -- the
# opposite of w_damper4/w_damper8, which were drawn up before that was known.
_BEST: dict[str, Any] = {**FLAT, "heating_pressure": -1.0,
                         "cooling_pressure_gain": 1.0, "reheat_full_k": 0.5}
_C: dict[str, tuple[str, dict[str, Any]]] = {
    "c_damper1": ("fan", {**_BEST, "zone_damper_full_k": 1.0}),
    "c_damper05": ("fan", {**_BEST, "zone_damper_full_k": 0.5}),
    # A stiffer loop reaches full supply-air authority sooner. pid's default
    # kp = 0.5 saturates at 2 K, which is a long way out when a kelvin of
    # occupied deviation costs 19 kWh.
    "c_kp10": ("fan", {**_BEST, "kp": 1.0}),
    # Read each zone's band from its own setpoint sensors, so the controller
    # and the scorer agree on where the edge is. Described in pid_plus's
    # docstring and then left switched off.
    "c_sensorband": ("fan", {**_BEST, "use_sensor_band": True}),
    # The one shipped refinement never measured in this chain, and the only
    # candidate left that is structural rather than a gain: which zone the
    # single shared supply-air temperature is chosen for.
    "c_worst": ("fan", {**_BEST, "use_worst_zone": True}),
    "c_worst_band": ("fan", {**_BEST, "use_worst_zone": True,
                             "use_sensor_band": True}),
}
_W.update(_C)

# ---------------------------------------------------------------------------
# Per-zone PI. This repository's earlier comfort conclusion was wrong.
#
# We tested a stiffer loop, faster dampers and a tighter target, measured all
# three as worse, and concluded the gains sat near a local optimum. All three
# were pushes in the direction that CAUSES the oscillation. The public "Probe
# ZT Best" notebook measured what was actually happening: zones averaged
# inside the 21-24 C band yet sat outside it 57-61% of working hours, swinging
# 2-4 K every 5-15 minutes.
#
# The fix is a much softer proportional gain AND an integral term, aimed at
# the band centre rather than its edge. Neither half works alone -- soft P
# trades the swing for a standing offset, and an integral on a deadband error
# has no target inside the band to integrate towards. An ablation that moves
# one knob at a time cannot find it.
_PI: dict[str, Any] = {**_BEST}
_P: dict[str, tuple[str, dict[str, Any]]] = {
    "p_ctrl": ("zonepi", {**_PI, "enabled": False}),
    "p_pi": ("zonepi", {**_PI, "kp": 0.1, "ki": 0.005}),
    # Softer and stiffer either side, to check the published gains are not a
    # knife edge on a different controller.
    "p_pi_soft": ("zonepi", {**_PI, "kp": 0.05, "ki": 0.0025}),
    "p_pi_stiff": ("zonepi", {**_PI, "kp": 0.2, "ki": 0.01}),
    # Proportional only at the soft gain, to confirm the integral is load
    # bearing rather than the softness doing all the work.
    "p_p_only": ("zonepi", {**_PI, "kp": 0.1, "ki": 0.0}),
    # Per-zone PI only acts through dampers, and a damper is a share of a flow
    # that has to exist. The inherited loop idles a handler whenever its
    # AGGREGATE zone error reaches zero, so the zone loops can only act in the
    # windows that aggregate happens to leave open. Forcing the handlers on at
    # a fixed moderate supply temperature and a low static pressure is what a
    # real VAV plant does, and it is what the per-zone loop needs underneath
    # it.
    "p_forced": ("zonepi", {**_PI, "kp": 0.1, "ki": 0.005,
                            "force_ahu_on": True}),
    "p_forced_cold": ("zonepi", {**_PI, "kp": 0.1, "ki": 0.005,
                                 "force_ahu_on": True, "forced_sat": -0.9}),
}
_W.update(_P)

# ---------------------------------------------------------------------------
# Cooling-run hold. The shipped loop idles a handler the instant its zones are
# back in band and restarts it at the next excursion, which with an
# oscillating zone loop is 50-107 switches per handler per day. Holding only
# the runs started for cooling keeps the hot-climate gain without stretching
# heating bursts, which a handler that cannot heat has no reason to stretch.
_PIBEST: dict[str, Any] = {**_BEST, "kp": 0.1, "ki": 0.005}
_H: dict[str, tuple[str, dict[str, Any]]] = {
    "h_ctrl": ("hold", {**_PIBEST, "hold_steps": 0}),
    "h_6": ("hold", {**_PIBEST, "hold_steps": 6}),
    "h_96": ("hold", {**_PIBEST, "hold_steps": 96}),
    "h_288": ("hold", {**_PIBEST, "hold_steps": 288}),
}

# All three pieces together. Forcing the handlers on is gated on the zones
# asking to be cooled, which keeps the hot-climate gain (phoenix_4 -44%) and
# removes the cold-climate loss entirely (fairbanks_1 unchanged, because the
# gate never fires there).
_FORCED: dict[str, Any] = {**_PIBEST, "force_ahu_on": True}
_H.update({
    "all_nohold": ("hold", {**_FORCED, "hold_steps": 0}),
    "all_96": ("hold", {**_FORCED, "hold_steps": 96}),
    "all_288": ("hold", {**_FORCED, "hold_steps": 288}),
})
_W.update(_H)

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

    # Open the reheat valve sooner. With the band flat on fairbanks_1,
    # pid-based fixed_setpoint scores 0.149 against pid_water-based 0.264,
    # while on phoenix_4 the same refinements are 2.5x better -- so a
    # refinement tuned to protect a mild climate from overshoot is starving a
    # cold one, where the valve is the only heat source in the building.
    "v_reheat05": ("band", {**FLAT, "reheat_full_k": 0.5}),
    "v_reheat10": ("band", {**FLAT, "reheat_full_k": 1.0}),
    # Everything that should help the cold, together.
    "v_cold": ("fan", {**FLAT, "reheat_full_k": 0.5,
                       "heating_pressure": -1.0,
                       "cooling_pressure_gain": 0.6,
                       "heating_water_floor": 0.5}),

    # The pump is electricity too, and the inherited chain pins its
    # differential pressure at the full 20 units whenever any handler runs.
    "v_pump0": ("fan", {**FLAT, "heating_pressure": None,
                        "cooling_pressure_gain": 1.0,
                        "pump_pressure": 0.0}),
    **_W,
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


# ---------------------------------------------------------------------------
# Heating-mode comfort. After the fan clamp, the per-zone PI and the cooling
# hold, comfort loss correlates -0.78 with mean outdoor temperature: ten times
# worse below 15 C than above it, and the 0-15 C shoulder band is nearly as
# bad as the deep cold on a quarter of the energy. The worst scenario by
# comfort, fairbanks_7, averages +6.4 C and never drops below -6.8 C.
#
# So it is a heating-mode problem rather than a cold one. A reheat coil heats
# the air passing through it, and the decision to run a handler is still the
# inherited loop's, taken on its AGGREGATE zone error -- so when the aggregate
# is in band every zone under it loses its airflow at once, including the cold
# outlier the per-zone loop is trying to rescue.
_BEST288: dict[str, Any] = {**_PIBEST, "hold_steps": 288}
_D: dict[str, tuple[str, dict[str, Any]]] = {
    "d_ctrl": ("heat", {**_BEST288, "heat_airflow": False}),
    "d_heat": ("heat", {**_BEST288, "heat_airflow": True}),
    # Require a share of the handler's zones to want heat before starting it:
    # a single unreachable outlier should not buy airflow for everyone else.
    "d_heat25": ("heat", {**_BEST288, "heat_airflow": True,
                          "heat_demand_frac": 0.25}),
    "d_heat50": ("heat", {**_BEST288, "heat_airflow": True,
                          "heat_demand_frac": 0.5}),
}
_W.update(_D)
_VARIANTS.update(_D)

# Factorial follow-up: water interlock and a physical ventilation-loss guard.
# No building identifiers or weather labels enter the controller.
_E = {
    "e_sync": ("heat_sync", {**_BEST288, "coordinate_water": True}),
    "e_guard_air": ("heat_sync", {**_BEST288, "coordinate_water": False,
                                 "minimum_delivery_efficiency": 0.5}),
    "e_guard_sync": ("heat_sync", {**_BEST288, "coordinate_water": True,
                                  "minimum_delivery_efficiency": 0.5}),
}
_VARIANTS.update(_E)

# Revisit a small heating setback only after fixing the reheat delivery path.
# Daytime comfort and cooling bands are unchanged. The inherited three-hour
# ramp restores the heating band before the scored window opens.
_VARIANTS["n_sync_s1"] = (
    "heat_sync", {**_BEST288, "coordinate_water": True,
                  "setback_k": 1.0, "setup_k": 0.0},
)

# Fixed, bounded delay-compensation screen; plant coordination stays e_sync.
for _key, _blend in (("r_zero", 0.0), ("r_half", 0.5), ("r_full", 1.0)):
    _VARIANTS[_key] = ("delay", {**_BEST288, "coordinate_water": True,
                               "prediction_blend": _blend,
                               "prediction_limit_k": 1.0,
                               "prediction_warmup": 288})

for _key, _night_steps in (("s_hold_ctrl", 288), ("s_night0", 0), ("s_night12", 12)):
    _VARIANTS[_key] = ("scheduled_hold", {**_BEST288, "coordinate_water": True,
                                        "off_window_hold_steps": _night_steps})

for _key, _kwargs in (
    ("u_pump_ctrl", {"pump_limit": None}),
    ("u_pump_cap", {"pump_limit": 0.0, "guard_pump": False}),
    ("u_pump_guard", {"pump_limit": 0.0, "guard_pump": True,
                       "pump_reserve_k": 0.5, "pump_valve_limit": 0.8}),
):
    _VARIANTS[_key] = ("guarded_pump", {**_BEST288, "coordinate_water": True, **_kwargs})

for _key, _kwargs in (
    ("v_pump_ctrl", {"pump_limit": None}),
    ("v_pump_warm", {"pump_limit": 0.0, "require_heat_override": False}),
    ("v_pump_heat", {"pump_limit": 0.0, "require_heat_override": True}),
):
    _VARIANTS[_key] = ("heating_pump", {**_BEST288, "coordinate_water": True,
                                       "guard_pump": True, "pump_reserve_k": 0.5,
                                       "pump_valve_limit": 0.8, **_kwargs})

for _key, _hours in (("a_start_ctrl", 1.0), ("a_start2", 2.0), ("a_start3", 3.0)):
    _VARIANTS[_key] = ("heat_sync", {**_BEST288, "coordinate_water": True,
                                    "prestart_hours": _hours})

for _key, _ceiling in (("cpi_ctrl", None), ("cpi_400", -0.6), ("cpi_600", -0.4)):
    _VARIANTS[_key] = ("cooling_pressure", {**_BEST288, "coordinate_water": True,
                                           "cooling_pressure_ceiling": _ceiling,
                                           "cooling_pressure_floor": -0.8,
                                           "cooling_effort_threshold": 0.8,
                                           "cooling_error_full_k": 1.0})

for _key, _ceiling, _protect in (("cfb_ctrl", None, False),
                                  ("cfb_comp", -0.6, False),
                                  ("cfb_guard", -0.6, True)):
    _VARIANTS[_key] = ("flow_balanced_cooling", {**_BEST288, "coordinate_water": True,
                                                "cooling_pressure_ceiling": _ceiling,
                                                "cooling_pressure_floor": -0.8,
                                                "cooling_effort_threshold": 0.8,
                                                "cooling_error_full_k": 1.0,
                                                "protect_minimum_flow": _protect,
                                                "minimum_damper": 0.1,
                                                "cold_neighbor_margin_k": 0.5})

# One-coordinate zone-PI calibration on the actual e_sync plant. Keep the
# maximum integral contribution equal to one when varying its gain.
for _key, _kp, _ki in (("pi_cal_ctrl", 0.1, 0.005),
                       ("pi_cal_slow_i", 0.1, 0.0025),
                       ("pi_cal_fast_i", 0.1, 0.01),
                       ("pi_cal_soft_p", 0.05, 0.005),
                       ("pi_cal_firm_p", 0.2, 0.005),
                       ("pi_cal_soft_slow", 0.05, 0.0025),
                       ("pi_cal_soft_fast", 0.05, 0.01),
                       ("pi_cal_firm_slow", 0.2, 0.0025),
                       ("pi_cal_firm_fast", 0.2, 0.01)):
    _VARIANTS[_key] = ("heat_sync", {**_BEST288, "coordinate_water": True,
                                    "kp": _kp, "ki": _ki,
                                    "integral_limit": 1.0 / _ki})

for _key, _enabled, _exponent, _scale_i in (
    ("geo_ctrl", False, 0.5, False),
    ("geo_p_sqrt", True, 0.5, False),
    ("geo_pi_sqrt", True, 0.5, True),
    ("geo_p_linear", True, 1.0, False),
):
    _VARIANTS[_key] = ("geometry_pi", {**_BEST288, "coordinate_water": True,
                                      "integral_limit": 200.0,
                                      "area_gain_enabled": _enabled,
                                      "reference_area_cv": 2484.0,
                                      "area_exponent": _exponent,
                                      "scale_integral": _scale_i,
                                      "minimum_gain_scale": 0.5,
                                      "maximum_gain_scale": 2.0})

# Prospective V6 refinement: vary geometry sensitivity around V5's exponent
# 0.5, preserving both integral authority and all plant/schedule settings.
for _key, _exponent in (("geo6_ctrl", 0.5), ("geo6_quarter", 0.25),
                        ("geo6_threequarter", 0.75), ("geo6_linear", 1.0)):
    _VARIANTS[_key] = ("geometry_pi", {**_BEST288, "coordinate_water": True,
                                      "integral_limit": 200.0,
                                      "area_gain_enabled": True,
                                      "reference_area_cv": 2484.0,
                                      "area_exponent": _exponent,
                                      "scale_integral": True,
                                      "minimum_gain_scale": 0.5,
                                      "maximum_gain_scale": 2.0})

# V7 separates response to instantaneous error from accumulated error.
for _key, _p, _i in (("geo7_ctrl", .75, .75),
                     ("geo7_soft_i", .75, .5),
                     ("geo7_strong_i", .75, 1.),
                     ("geo7_soft_p", .5, .75)):
    _VARIANTS[_key] = ("split_geometry_pi", {**_BEST288, "coordinate_water": True,
                                            "integral_limit": 200.,
                                            "area_gain_enabled": True,
                                            "reference_area_cv": 2484.,
                                            "area_exponent": _p,
                                            "integral_area_exponent": _i,
                                            "scale_integral": True,
                                            "minimum_gain_scale": .5,
                                            "maximum_gain_scale": 2.})

# Return to the best organizer-scored V4 plant; no geometry dependence.
for _key, _power in (("valve8_ctrl", 1.), ("valve8_sqrt", .5),
                      ("valve8_gentle", .75), ("valve8_soft", 1.5)):
    _VARIANTS[_key] = ("reheat_curve", {**_BEST288, "coordinate_water": True,
                                      "reheat_exponent": _power})
