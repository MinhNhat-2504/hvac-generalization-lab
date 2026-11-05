"""Integrate error outside an interior comfort band, retaining load memory."""
import numpy as np
from hvaclab.policies.flow_equivalent import FlowEquivalentPolicy


class ComfortBandPIPolicy(FlowEquivalentPolicy):
    def __init__(self,*args,band_pi=True,comfort_reserve_k=.5,**kwargs):
        if not np.isfinite(comfort_reserve_k) or comfort_reserve_k<0:
            raise ValueError('comfort reserve must be finite and nonnegative')
        self.band_pi=bool(band_pi);self.comfort_reserve_k=float(comfort_reserve_k)
        super().__init__(*args,**kwargs)
        self._clear_band_diagnostics()

    def _clear_band_diagnostics(self):
        self.band_zone_steps=0;self.inside_band_with_memory_steps=0;self.max_error_reduction_k=0.

    def reset(self):
        super().reset();self._clear_band_diagnostics()

    def _command_zones(self,continuous,zones,obs,low,high):
        half=.5*(high-low)
        if not self.band_pi or not self._pi_active or self.comfort_reserve_k>=half:
            return super()._command_zones(continuous,zones,obs,low,high)
        centre=.5*(low+high);lower=low+self.comfort_reserve_k;upper=high-self.comfort_reserve_k
        std=self.zone_air_temperature_sensor_std;mean=self.zone_air_temperature_sensor_mean
        if not np.isfinite(std) or std<=0:return super()._command_zones(continuous,zones,obs,low,high)
        adjusted=np.array(obs,copy=True,dtype=float)
        for v,index,actions in zones:
            if not actions:continue
            temperature=float(self._denormalize_zone_temp(obs[index]))
            if not np.isfinite(temperature):continue
            error=temperature-float(np.clip(temperature,lower,upper))
            # Only the parent zone-command call receives the equivalent error.
            # AHU aggregation, other commands and caller observation stay raw.
            adjusted[index]=(centre+error-mean)/std
            self.band_zone_steps+=1
            self.inside_band_with_memory_steps+=int(error==0 and abs(self._zone_integral.get(v,0.))>1e-8)
            self.max_error_reduction_k=max(self.max_error_reduction_k,abs(temperature-centre)-abs(error))
        return super()._command_zones(continuous,zones,adjusted,low,high)

    def hold_report(self):
        return dict(super().hold_report(),band_zone_steps=self.band_zone_steps,
                    inside_band_with_memory_steps=self.inside_band_with_memory_steps,
                    max_error_reduction_k=self.max_error_reduction_k)
