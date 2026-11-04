"""Chronological identification with public delayed sensors and prior actuators."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from hvaclab.coverage import digest,write_json
from hvaclab.policies.thermal_response_model import fan_state
from hvaclab.policies.actuator_thermal_model import actuator_features,ActuatorThermalPredictor,FEATURE_NAMES


class PublicActuatorView:
    def __init__(self,payload):
        self.schema=payload['schema'];self.rows=payload['rows'];self.names=self.schema['field_names']
        self.occupancy_index=self.names.index('num_occupants')
        norm=self.schema['observation_normalization_metadata']['zone_air_temperature_sensor']
        self.mean=norm['sample_mean'];self.std=norm['sample_std']
        self.bindings=[]
        for c in self.schema['controllers']:
            prefix='ahs_'+c['ahu_id']+'_'
            self.bindings.append({name:self.names.index(prefix+suffix) for name,suffix in (
                ('sat','supply_air_temperature_sensor'),('pressure','supply_air_static_pressure_sensor'),
                ('recirc','recirculation_percentage_command'),('outside','outside_air_temperature_sensor'))})

    def temperature(self,t,zone):
        return self.rows[t]['observation'][self.schema['zones'][zone]['index']]*self.std+self.mean

    def occupancy(self,t):return self.rows[t]['observation'][self.occupancy_index]

    def state(self,t,zone):
        handler=self.schema['zones'][zone]['handler'];c=self.schema['controllers'][handler];j=c['run_discrete_idx']
        return fan_state(self.rows[t-1]['discrete'][j],self.rows[t-2]['discrete'][j])

    def context(self,t,zone):
        z=self.schema['zones'][zone];handler=z['handler'];c=self.schema['controllers'][handler]
        # Only the command already issued at t-1 is used for the missing interval.
        command=self.rows[t-1];obs=self.rows[t]['observation'];a=command['continuous'];d=command['discrete'];b=self.bindings[handler]
        norm=self.schema['action_normalization_metadata']['supply_water_temperature_setpoint']
        value=a[self.schema['water_temp_idx']]
        fraction=(value-norm['min_normalized_value'])/(norm['max_normalized_value']-norm['min_normalized_value'])
        water_native=norm['min_native_value']+fraction*(norm['max_native_value']-norm['min_native_value'])
        sat=obs[b['sat']];outside=obs[b['outside']]
        if not 100<sat<400 or not 100<outside<400:raise ValueError('expected native Kelvin AHU observations; inspect exact prefixed normalizers')
        return dict(fan=d[c['run_discrete_idx']],pump=d[self.schema['water_discrete_idx']],
            sat_command=a[c['temp_idx']],pressure_command=a[c['pressure_idx']],recirc_command=a[c['recirc_idx']],
            damper=(a[z['actions']['supply_air_damper_percentage_command']]+1)/2,
            valve=(a[z['actions']['reheat_valve_percentage_command']]+1)/2,
            water_temperature_command=value,water_temperature_native=water_native,
            water_pressure_command=a[self.schema['water_pressure_idx']],
            sat_proxy_k=sat,outside_k=outside,pressure_observation=obs[b['pressure']],recirc_observation=obs[b['recirc']])


def records(view,start,end,centre=295.5):
    for t in range(max(2,start),end-1):
        for z in range(len(view.schema['zones'])):
            temperature=view.temperature(t,z);slope=temperature-view.temperature(t-1,z)
            target=view.temperature(t+1,z)
            yield dict(step=t,zone=z,state=view.state(t,z),temperature=temperature,slope=slope,
                occupancy=view.occupancy(t),target=target,
                features=actuator_features(temperature,slope,view.occupancy(t),view.context(t,z),centre))


def masked(x,variant):
    result=x.copy()
    if variant=='state_only':result[4:]=0.
    elif variant=='commands_only':result[13:]=0.
    elif variant!='full_actuators':raise ValueError('unknown feature ablation')
    return result


def errors(rows,variant,radius):
    out={}
    for group in ('all','occupied','empty','off_entry','on_entry','off_steady','on_steady'):
        subset=[r for r in rows if group=='all' or (group=='occupied' and r['occupancy']>0)
                or (group=='empty' and r['occupancy']<=0) or r['state']==group]
        if not subset:out[group]=dict(n=0);continue
        e=np.array([abs(r[variant]-r['target']) for r in subset])
        r=radius.get(group)
        out[group]=dict(n=len(subset),mae=float(e.mean()),p95=float(np.quantile(e,.95)),
                        calibration_radius=r,coverage=None if r is None else float(np.mean(e<=r)))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();directory=Path(args.directory);root=Path(args.output)
    public_path=directory/'public_actuators.json';payload=json.loads(public_path.read_text());trace_path=directory/'damper12_mid.json'
    if digest(trace_path)!=payload['trace_sha256']:raise ValueError('paired trace hash changed')
    n=payload['steps'];split=n//2;cal_end=3*n//4
    protocol=dict(public_sha256=digest(public_path),steps=n,fit_end=split,calibration_end=cal_end,ridge=10.,
        primary_target='current temperature missing due to one-step sensor lag; label is next public observation',
        feature_names=FEATURE_NAMES,source_sha256={p:digest(p) for p in ('hvaclab/actuator_response_study.py','hvaclab/policies/actuator_thermal_model.py','hvaclab/actuator_trace.py')},
        primary_gate='full MAE beats aged, linear, and state-only overall and occupied; occupied p95 no worse than best comparator; occupied empirical interval coverage>=95%, radius<1.5K',
        scope='fixed short diagnostic architecture; same building different weather; no closed-loop reward claim')
    protocol=json.loads(json.dumps(protocol))
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('frozen inputs changed')
    write_json(root/'protocol.json',protocol)
    view=PublicActuatorView(payload);training=list(records(view,2,split));cal=list(records(view,split,cal_end));test=list(records(view,cal_end,n))
    models={}
    for variant in ('state_only','commands_only','full_actuators'):
        model=ActuatorThermalPredictor().fit((r['zone'],r['state'],masked(r['features'],variant),r['target']-r['temperature']) for r in training)
        models[variant]=model
        for r in cal+test:
            prediction=model.predict_delta(r['zone'],r['state'],masked(r['features'],variant))
            if prediction is None:raise ValueError('missing supported fan state; refuse silent sample removal')
            r[variant]=r['temperature']+prediction
    variants=('aged_sensor','linear','state_only','commands_only','full_actuators')
    for r in cal+test:r['aged_sensor']=r['temperature'];r['linear']=r['temperature']+r['slope']
    result={}
    for variant in variants:
        radii={}
        for group in ('all','occupied','empty','off_entry','on_entry','off_steady','on_steady'):
            subset=[r for r in cal if group=='all' or (group=='occupied' and r['occupancy']>0)
                    or (group=='empty' and r['occupancy']<=0) or r['state']==group]
            e=sorted(abs(r[variant]-r['target']) for r in subset)
            if len(e)>=20:radii[group]=e[min(len(e)-1,int(np.ceil((len(e)+1)*.95))-1)]
        result[variant]=errors(test,variant,radii)
    full=result['full_actuators'];comparators=('aged_sensor','linear','state_only')
    passes=bool(full['occupied']['n']>0
        and all(full[group]['mae']<min(result[v][group]['mae'] for v in comparators) for group in ('all','occupied'))
        and full['occupied']['p95']<=min(result[v]['occupied']['p95'] for v in comparators)
        and full['occupied']['coverage'] is not None and full['occupied']['coverage']>=.95
        and full['occupied']['calibration_radius']<1.5)
    summary=dict(protocol=protocol,building=payload['building'],weather=payload['weather'],
        train_samples=len(training),calibration_samples=len(cal),test_samples=len(test),scores=result,
        passes_prediction_gate=passes,controller_changed=False,notebook_exported=False)
    write_json(root/'results.json',summary)
    write_json(root/'models.json',{k:v.parameters() for k,v in models.items()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
