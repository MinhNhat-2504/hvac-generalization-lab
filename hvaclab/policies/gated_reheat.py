"""Apply the measured reheat boost only in selected observable conditions."""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class GatedReheatPolicy(CoordinatedHeatPolicy):
    def __init__(self,*args,curve_mode="off",reheat_exponent=.75,cold_full_k=.5,**kwargs):
        if curve_mode not in ("off","all","heating","cold","heating_cold"):
            raise ValueError("unknown reheat-curve gate")
        if not np.isfinite(reheat_exponent) or reheat_exponent <= 0:
            raise ValueError("reheat exponent must be finite and positive")
        if not np.isfinite(cold_full_k) or cold_full_k <= 0:
            raise ValueError("cold blend width must be finite and positive")
        super().__init__(*args,**kwargs)
        self.curve_mode=curve_mode
        self.reheat_exponent=float(reheat_exponent)
        self.cold_full_k=float(cold_full_k)

    def get_action(self,obs):
        action=super().get_action(obs)
        if self.curve_mode=="off" or self.reheat_exponent==1. or not (
                self.zone_pi_enabled and self._pi_active):
            return action
        continuous,discrete=action["continuous_action"],action["discrete_action"]
        low=self._comfort_band[0]
        for ctrl in self.ahu_controllers:
            # This is the final same-step command after cooling hold and heat
            # delivery. It is not a measurement of actual supply-air heat.
            heating_command=(bool(discrete[ctrl["run_discrete_idx"]])
                             and float(continuous[ctrl["temp_idx"]])>0.)
            if self.curve_mode in ("heating","heating_cold") and not heating_command:
                continue
            for vav,obs_idx,indices in ctrl["zones"]:
                valve=indices.get("reheat_valve_percentage_command")
                h=max(0.,-self._zone_u.get(vav,0.))
                if valve is None or h<=0:
                    continue
                weight=1.
                if self.curve_mode in ("cold","heating_cold"):
                    temp=self._denormalize_zone_temp(obs[obs_idx])
                    weight=float(np.clip((low-temp)/self.cold_full_k,0.,1.))
                if weight<=0:
                    continue
                opening=h+weight*(h**self.reheat_exponent-h)
                continuous[valve]=np.float32(2.*opening-1.)
        return action
