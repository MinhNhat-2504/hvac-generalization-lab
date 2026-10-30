"""Training-only reference area and offline verification of metadata room IDs."""
from __future__ import annotations

import os
from pathlib import Path
import time


def main():
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    import gc
    import cv2
    import numpy as np
    from hvaclab.coverage import digest,write_json
    from hvaclab.policies.geometry_pi import connected_room_areas
    from hvaclab.registry import resolve_policy
    from smart_buildings import evaluate as ev
    roots={"training":Path("smart_buildings/training_scenarios"),
           "dev":Path("smart_buildings/holdout_scenarios/dev")}
    training_areas=[]
    results=[]
    for split,root in roots.items():
        for building in sorted(root.iterdir()):
            cases=sorted(building.glob("*/zone_map.npy"))
            if not cases:
                continue
            path=cases[0]
            matrix=np.load(path)
            started=time.monotonic()
            areas=connected_room_areas(matrix)
            elapsed=time.monotonic()-started
            _,labels,stats,_=cv2.connectedComponentsWithStats((matrix==0).astype(np.uint8),connectivity=4)
            expected=stats[1:,cv2.CC_STAT_AREA].tolist()
            if areas!=expected:
                raise ValueError(f"component order mismatch: {path}")
            if split=="training":
                training_areas.extend(areas)
            env=ev.load_hybrid_environment(path.parent/"config.gin")
            meta=ev.create_building_meta_data(env,path.parent)
            policy=ev.initialize_policy(resolve_policy("geo_p_sqrt"),meta,env.action_spec())
            if policy.geometry_status!="mapped":
                raise ValueError(f"participant metadata mapping failed: {path}")
            actual=env.building.simulator.building.room_dict
            for i,area in enumerate(areas,1):
                if len(actual[f"room_{i}"])!=area:
                    raise ValueError(f"room area mismatch: {path}, room_{i}")
            scales=list(policy._zone_gain_scales.values())
            results.append(dict(split=split,building=building.name,zone_map=str(path),
                                zone_map_sha256=digest(path),areas_cv=areas,
                                component_and_sdk_mapping_verified=True,
                                component_seconds=elapsed,scale_min=min(scales),
                                scale_median=float(np.median(scales)),scale_max=max(scales)))
            print(f"VERIFIED {building.name}: {len(areas)} zones, component pass {elapsed:.3f}s",flush=True)
            del env,meta,policy,labels,stats
            gc.collect()
    reference=float(np.median(training_areas))
    if reference!=2484.:
        raise ValueError(f"training reference changed: {reference}")
    write_json("experiments/plans/geometry_reference.json",dict(
        reference_area_cv=reference,training_zone_count=len(training_areas),
        reference_uses_training_only=True,opencv_version=cv2.__version__,
        component_policy_sha256=digest("hvaclab/policies/geometry_pi.py"),
        verification_script_sha256=digest(__file__),buildings=results,
        note="Simulator geometry is inspected only by this offline mapping check; "
             "the submitted policy reads participant metadata only. Device-label ordering "
             "is verified on the current ten training/dev buildings, not all possible SDK versions."))
    print(f"REFERENCE {reference} CV, {len(training_areas)} training zones",flush=True)


if __name__=="__main__":
    main()
