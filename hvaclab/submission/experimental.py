"""Build an isolated, self-contained notebook for a measured heat variant.

    python -m hvaclab.submission.experimental --variant e_guard_sync --verify

Writes under experiments/submission/<variant>/; the default submission bundle
is preserved. Verification compares actions on real observations, not scores
from a short run, which must not be mistaken for performance validation.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from hvaclab.submission.build import HEADER, NOTEBOOK_CELL, REPO, build_notebook

SCHEDULED = ("s_hold_ctrl", "s_night0", "s_night12")
GUARDED_PUMP = ("u_pump_ctrl", "u_pump_cap", "u_pump_guard")
HEATING_PUMP = ("v_pump_ctrl", "v_pump_warm", "v_pump_heat")
PRESTART = ("a_start_ctrl", "a_start2", "a_start3")
COOLING_PRESSURE = ("cpi_ctrl", "cpi_400", "cpi_600")
FLOW_BALANCED = ("cfb_ctrl", "cfb_comp", "cfb_guard")
PI_CALIBRATION = ("pi_cal_ctrl", "pi_cal_slow_i", "pi_cal_fast_i", "pi_cal_soft_p", "pi_cal_firm_p",
                  "pi_cal_soft_slow", "pi_cal_soft_fast", "pi_cal_firm_slow", "pi_cal_firm_fast")
GEOMETRY_PI = ("geo_ctrl", "geo_p_sqrt", "geo_pi_sqrt", "geo_p_linear",
               "geo6_ctrl", "geo6_quarter", "geo6_threequarter", "geo6_linear")
SPLIT_GEOMETRY_PI = ("geo7_ctrl", "geo7_soft_i", "geo7_strong_i", "geo7_soft_p")
REHEAT_CURVE = ("valve8_ctrl", "valve8_sqrt", "valve8_gentle", "valve8_soft",
                "persist_ctrl","persist_144","persist_432","persist_576")
GATED_REHEAT = ("valve9_ctrl","valve9_curve_ctrl","valve9_heat","valve9_cold","valve9_heat_cold")
SUPPORTED = ("e_sync", "e_guard_air", "e_guard_sync", "n_sync_s1") + SCHEDULED + GUARDED_PUMP + HEATING_PUMP + PRESTART + COOLING_PRESSURE + FLOW_BALANCED + PI_CALIBRATION + GEOMETRY_PI + SPLIT_GEOMETRY_PI + REHEAT_CURVE + GATED_REHEAT
SOURCES = (
    "smart_buildings/baselines/pid.py",
    "smart_buildings/baselines/pid_plus.py",
    "hvaclab/policies/deadline_band.py",
    "hvaclab/policies/fan_economy.py",
    "hvaclab/policies/zone_pi.py",
    "hvaclab/policies/cooling_hold.py",
    "hvaclab/policies/heat_delivery.py",
    "hvaclab/policies/coordinated_heat.py",
)


def variant_sources(key):
    if key in GATED_REHEAT:
        return SOURCES+("hvaclab/policies/gated_reheat.py",)
    if key in REHEAT_CURVE:
        return SOURCES + ("hvaclab/policies/reheat_curve.py",)
    if key in SPLIT_GEOMETRY_PI:
        return SOURCES + ("hvaclab/policies/geometry_pi.py", "hvaclab/policies/split_geometry_pi.py")
    if key in GEOMETRY_PI:
        return SOURCES + ("hvaclab/policies/geometry_pi.py",)
    if key in FLOW_BALANCED:
        return SOURCES + ("hvaclab/policies/flow_balanced_cooling.py",)
    if key in COOLING_PRESSURE:
        return SOURCES + ("hvaclab/policies/cooling_pressure.py",)
    if key in HEATING_PUMP:
        return SOURCES + ("hvaclab/policies/heating_pump.py",)
    if key in SCHEDULED:
        return SOURCES + ("hvaclab/policies/scheduled_hold.py",)
    if key in GUARDED_PUMP:
        return SOURCES + ("hvaclab/policies/guarded_pump.py",)
    return SOURCES


def assemble_variant(key):
    from hvaclab.registry import resolve_policy
    if key not in SUPPORTED:
        raise ValueError(f"unsupported experimental variant: {key}")
    params = dict(resolve_policy(key)._variant_kwargs)
    parts = [HEADER.replace("python -m hvaclab.submission.build",
                            f"python -m hvaclab.submission.experimental --variant {key}")]
    sources = variant_sources(key)
    for relative in sources:
        source = (REPO / relative).read_text(encoding="utf-8")
        lines = source.splitlines(keepends=True)
        for node in ast.parse(source).body:
            if isinstance(node, ast.ImportFrom) and (
                node.module == "__future__" or
                (node.module or "").startswith(("smart_buildings.", "hvaclab."))
            ):
                for i in range(node.lineno - 1, node.end_lineno):
                    lines[i] = ""
        parts.append(f"\n# Source: {relative}\n" + "".join(lines))
    parent = ("GatedReheatPolicy" if key in GATED_REHEAT else
              "ReheatCurvePolicy" if key in REHEAT_CURVE else
              "SplitGeometryPIPolicy" if key in SPLIT_GEOMETRY_PI else
              "GeometryPIPolicy" if key in GEOMETRY_PI else
              "FlowBalancedCoolingPolicy" if key in FLOW_BALANCED else
              "CoolingPressurePolicy" if key in COOLING_PRESSURE else
              "HeatingPumpPolicy" if key in HEATING_PUMP else
              "GuardedPumpPolicy" if key in GUARDED_PUMP else
              "ScheduledHoldPolicy" if key in SCHEDULED else "CoordinatedHeatPolicy")
    parts.append(f"\nEQUIVALENT_TO = {key!r}\n\n"
                 f"class FinalPolicy({parent}):\n"
                 "    def __init__(self, building_meta_data, action_spec):\n"
                 f"        super().__init__(building_meta_data, action_spec, **{params!r})\n")
    source = "\n".join(parts)
    compile(source, f"{key}_policy.py", "exec")
    return source, params


def verify_variant(source, key, steps=576):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from smart_buildings import evaluate as ev
    from hvaclab.registry import resolve_policy
    namespace = {}
    exec(compile(source, "experimental_policy.py", "exec"), namespace)
    for building, weather in (("building_train_1", "fairbanks_1"),
                              ("building_train_7", "fairbanks_7")):
        scenario = REPO / "smart_buildings/training_scenarios" / building / weather
        env = ev.load_hybrid_environment(scenario / "config.gin")
        meta = ev.create_building_meta_data(env, scenario)
        assembled = namespace["FinalPolicy"](meta, env.action_spec())
        reference = resolve_policy(key)(meta, env.action_spec())
        obs, _ = env.reset()
        for step in range(steps):
            left, right = assembled.get_action(obs), reference.get_action(obs)
            for field in ("continuous_action", "discrete_action"):
                if not np.array_equal(left[field], right[field]):
                    raise AssertionError(f"{building}/{weather}: {field} differs at {step}")
            obs, _, terminated, truncated, _ = env.step(left)
            if terminated or truncated:
                break
        print(f"MATCH {building}/{weather}: {step + 1} action pairs", flush=True)


def notebook_template(source,key):
    """Use the inference wrapper and identify new valve-curve runs in logs."""
    template=(f"# Experimental candidate: {key}\n"
              "# No leaderboard claim. See reports/HEAT_SYNC_REVIEW.md.\n\n"
              +NOTEBOOK_CELL[NOTEBOOK_CELL.index("import sys"):])
    if key in REHEAT_CURVE+GATED_REHEAT:
        marker=(f"HVAC_SUBMISSION variant={key} "
                f"source_lf_sha256={hashlib.sha256(source.encode('utf-8')).hexdigest()}")
        template=template.replace("class SubmissionPolicy(",
                                  f"print({marker!r}, flush=True)\n\nclass SubmissionPolicy(",1)
    return template


def main():
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=SUPPORTED, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    source, params = assemble_variant(args.variant)
    if args.verify:
        verify_variant(source, args.variant)
    out = REPO / "experiments/submission" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission_policy.py").write_text(source, encoding="utf-8")
    # Reuse the inference-server interface without the default policy's
    # historical method description or performance claims.
    template=notebook_template(source,args.variant)
    notebook = build_notebook(source, "", "", "", template=template)
    (out / "submission.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    manifest = dict(variant=args.variant, parameters=params,
                    action_parity_verified=args.verify,
                    # Hash the actual artifact: Windows text output may use
                    # CRLF even though the assembled source string uses LF.
                    policy_sha256=hashlib.sha256((out / "submission_policy.py").read_bytes()).hexdigest(),
                    sources={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
                             for p in variant_sources(args.variant)})
    if args.variant in REHEAT_CURVE+GATED_REHEAT:
        manifest["runtime_source_lf_sha256"]=hashlib.sha256(source.encode("utf-8")).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
