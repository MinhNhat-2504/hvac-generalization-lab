"""Capture public observations and commands alongside an unchanged diagnostic probe."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
from hvaclab.coverage import write_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weather',required=True)
    parser.add_argument('--building',default='building_9000')
    parser.add_argument('--steps',type=int,default=2016)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 1<=args.steps<=8640:parser.error('steps must be 1..8640')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    from smart_buildings import evaluate as ev
    from hvaclab.thermal_diagnostics import probe
    original=ev.initialize_policy;public=[];schema={}
    def initialize(cls,meta,spec):
        policy=original(cls,meta,spec)
        schema.update(field_names=list(meta['observation_field_names']),
            observation_normalization_metadata=meta['observation_normalization_metadata'],
            action_normalization_metadata=meta['action_normalization_metadata'],
            controllers=[{k:v for k,v in c.items() if k in ('ahu_id','run_discrete_idx','temp_idx','pressure_idx','recirc_idx','oat_idx')} for c in policy.ahu_controllers],
            zones=[dict(vav=v,index=i,handler=h,actions=indices) for h,c in enumerate(policy.ahu_controllers) for v,i,indices in c['zones']],
            water_discrete_idx=policy.water_discrete_idx,water_temp_idx=policy.water_temp_idx,water_pressure_idx=policy.water_pressure_idx)
        before=policy.get_action
        def action(obs):
            command=before(obs)
            public.append(dict(observation=[float(x) for x in obs],
                continuous=[float(x) for x in command['continuous_action']],
                discrete=[int(x) for x in command['discrete_action']]))
            return command
        policy.get_action=action
        return policy
    ev.initialize_policy=initialize
    try:probe('damper12_mid',args.building,args.weather,args.steps,root)
    finally:ev.initialize_policy=original
    trace_path=root/'damper12_mid.json';base=json.loads(trace_path.read_text())
    if len(base['trace'])!=len(public):raise RuntimeError('public/action trace length mismatch')
    write_json(root/'public_actuators.json',dict(schema=schema,steps=len(public),building=args.building,
        weather=args.weather,trace_sha256=digest(trace_path),rows=public,
        semantics='public observations before command; normalized commands. AHU SAT sensor is setpoint proxy, not measured discharge temperature.'))
    print('SAVED_PUBLIC_ACTUATORS',args.weather,len(public),flush=True)


if __name__=='__main__':main()
