"""Change only the heating damper characteristic, retaining the reheat curve."""
import numpy as np

from hvaclab.policies.reheat_curve import ReheatCurvePolicy


class HeatingDamperCurvePolicy(ReheatCurvePolicy):
    def __init__(self,*args,heating_damper_exponent=1.,**kwargs):
        if not np.isfinite(heating_damper_exponent) or heating_damper_exponent<=0:
            raise ValueError('heating damper exponent must be finite and positive')
        super().__init__(*args,**kwargs)
        self.heating_damper_exponent=float(heating_damper_exponent)

    def _command_zones(self,continuous,zones,obs,low,high):
        super()._command_zones(continuous,zones,obs,low,high)
        if self.heating_damper_exponent==1. or not (self.zone_pi_enabled and self._pi_active):
            return
        for vav,_,indices in zones:
            index=indices.get('supply_air_damper_percentage_command')
            u=self._zone_u.get(vav,0.)
            if index is not None and u<0:
                continuous[index]=np.float32(2.*(-u)**self.heating_damper_exponent-1.)
