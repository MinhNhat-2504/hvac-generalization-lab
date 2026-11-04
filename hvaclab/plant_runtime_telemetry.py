"""Offline, read-only taps around plant control stages and scorer-meter aggregation."""
from contextlib import contextmanager
import numpy as np
from hvaclab.policies.zone_pi import ZonePIPolicy
from hvaclab.policies.cooling_hold import CoolingHoldPolicy
from hvaclab.policies.heat_delivery import HeatDeliveryPolicy
from hvaclab.policies.phase_online_actuator import PHASES


def snapshot(policy,action):
    return dict(run=[int(action['discrete_action'][c['run_discrete_idx']]) for c in policy.ahu_controllers],
                sat=[float(action['continuous_action'][c['temp_idx']]) for c in policy.ahu_controllers])


@contextmanager
def stage_taps(primary):
    zone=ZonePIPolicy.get_action;hold=CoolingHoldPolicy.get_action;heat=HeatDeliveryPolicy.get_action
    frame={}
    def zone_tap(self,obs):
        action=zone(self,obs)
        if self is primary():frame['parent']=snapshot(self,action)
        return action
    def hold_tap(self,obs):
        watched=self is primary();before=list(self._hold_left) if watched else None
        applied=self.holds_applied if watched else 0;started=self.holds_started if watched else 0
        action=hold(self,obs)
        if watched:
            after=snapshot(self,action);pre=frame['parent']
            kept=[bool(not p and q) for p,q in zip(pre['run'],after['run'])]
            if sum(kept)!=self.holds_applied-applied:raise AssertionError('hold-apply attribution mismatch')
            refreshed=[bool(self.hold_steps>0 and on and sat<0) for on,sat in zip(pre['run'],pre['sat'])]
            newly=[bool(r and left<=0) for r,left in zip(refreshed,before)]
            if sum(newly)!=self.holds_started-started:raise AssertionError('hold-start attribution mismatch')
            frame.update(hold=kept,refreshed=refreshed,new_hold=newly,hold_before=before,
                         hold_after=list(self._hold_left),post_hold=after)
        return action
    def heat_tap(self,obs):
        watched=self is primary();starts=self.heat_starts if watched else 0
        action=heat(self,obs)
        if watched:
            after=snapshot(self,action)
            added=[bool(not p and q) for p,q in zip(frame['post_hold']['run'],after['run'])]
            if sum(added)!=self.heat_starts-starts:raise AssertionError('heat-airflow attribution mismatch')
            frame['heat_override']=added
        return action
    ZonePIPolicy.get_action=zone_tap;CoolingHoldPolicy.get_action=hold_tap;HeatDeliveryPolicy.get_action=heat_tap
    try:yield frame
    finally:ZonePIPolicy.get_action=zone;CoolingHoldPolicy.get_action=hold;HeatDeliveryPolicy.get_action=heat


def metered_watts(info):
    ahu=info.air_handler_reward_infos.values();boiler=info.boiler_reward_infos.values();hp=info.heat_pump_reward_infos.values()
    return dict(blower=sum(r.blower_electrical_energy_rate for r in ahu),
        cooling=sum(r.air_conditioning_electrical_energy_rate for r in info.air_handler_reward_infos.values()),
        pump=sum(r.pump_electrical_energy_rate for r in boiler)+sum(r.pump_electrical_energy_rate for r in hp),
        gas=sum(r.natural_gas_heating_energy_rate for r in info.boiler_reward_infos.values()),
        heatpump=sum(r.electricity_heating_energy_rate for r in info.heat_pump_reward_infos.values()))


def aggregate(series):
    result={}
    for phase in PHASES:
        rows=[r for r in series if r['phase']==phase]
        out=dict(steps=len(rows),ahu_handler_hours=0.,parent_request_handler_hours=0.,hold_handler_hours=0.,
                 heat_override_handler_hours=0.,armed_hold_handler_hours=0.,parent_cooling_handler_hours=0.,parent_warm_handler_hours=0.,pump_hours=0.,pump_without_valve_hours=0.,held_reheat_handler_hours=0.,
                 held_reheat_without_pump_handler_hours=0.,ahu_starts=0,ahu_stops=0,pump_starts=0,pump_stops=0,hold_refreshes=0,new_hold_episodes=0,
                 hold_origin_phase_steps={p:0 for p in PHASES},metered_Wh={k:0. for k in ('blower','cooling','pump','gas','heatpump')})
        for r in rows:
            hours=r['dt_seconds']/3600.
            for key,field in (('ahu_handler_hours','run'),('parent_request_handler_hours','parent_run'),('hold_handler_hours','hold'),('heat_override_handler_hours','heat_override')):
                out[key]+=sum(r[field])*hours
            out['armed_hold_handler_hours']+=sum(n>0 for n in r.get('hold_left',[]))*hours
            out['parent_cooling_handler_hours']+=sum(on and sat<0 for on,sat in zip(r['parent_run'],r.get('parent_sat',[0]*len(r['parent_run']))))*hours
            out['parent_warm_handler_hours']+=sum(on and sat>=0 for on,sat in zip(r['parent_run'],r.get('parent_sat',[0]*len(r['parent_run']))))*hours
            out['pump_hours']+=r['pump']*hours
            out['pump_without_valve_hours']+=int(r['pump'] and r['open_valves']==0)*hours
            out['held_reheat_handler_hours']+=sum(r['held_reheat'])*hours
            out['held_reheat_without_pump_handler_hours']+=sum(r['held_reheat'])*int(not r['pump'])*hours
            out['ahu_starts']+=sum(r['starts']);out['pump_starts']+=r['pump_start']
            out['ahu_stops']+=sum(r.get('stops',[]));out['pump_stops']+=r.get('pump_stop',0)
            out['hold_refreshes']+=sum(r['refreshed']);out['new_hold_episodes']+=sum(r['new_hold'])
            for held,origin in zip(r['hold'],r['hold_origin']):
                if held and origin in out['hold_origin_phase_steps']:out['hold_origin_phase_steps'][origin]+=1
            for k,value in r['watts'].items():out['metered_Wh'][k]+=value*r['dt_seconds']/3600.
        total=out['parent_request_handler_hours']+out['hold_handler_hours']+out['heat_override_handler_hours']
        if not np.isclose(total,out['ahu_handler_hours'],atol=1e-9):raise ValueError('running-reason partition mismatch')
        result[phase]=out
    return result
