"""Experimental causal delay compensation for the zone PI loop only.

The estimator forecasts obs[t+1]-obs[t] from obs through t and action[t-1].
It learns a previous forecast's target only when the new observation arrives.
The aggregate plant loops continue to read the original observations.
"""
import numpy as np

from hvaclab.policies.coordinated_heat import CoordinatedHeatPolicy


class DelayCompensatedPolicy(CoordinatedHeatPolicy):
    def __init__(self, *args, prediction_blend=0.5, prediction_limit_k=1.0,
                 prediction_warmup=288, **kwargs):
        if not 0 <= prediction_blend <= 1 or prediction_limit_k <= 0 or prediction_warmup < 0:
            raise ValueError("invalid prediction blend, bound or warmup")
        super().__init__(*args, **kwargs)
        self.prediction_blend = float(prediction_blend)
        self.prediction_limit_k = float(prediction_limit_k)
        self.prediction_warmup = int(prediction_warmup)
        self._prediction_zones = [(vav, index, indices, ctrl)
                                  for ctrl in self.ahu_controllers
                                  for vav, index, indices in ctrl["zones"]]
        self._prediction_positions = {vav: i for i, (vav, _, _, _) in enumerate(self._prediction_zones)}
        n = len(self._prediction_zones)
        self._weights = np.zeros((n, 12))
        self._covariance = np.repeat(np.eye(12)[None], n, axis=0)
        self._correction_k = np.zeros(n)

    def reset(self):
        super().reset()
        self._previous_features = None
        self._previous_temperatures = None
        self._previous_action = None
        self._prediction_steps = 0
        if hasattr(self, "_weights"):
            self._weights.fill(0)
            self._covariance[:] = np.eye(12)
            self._correction_k.fill(0)

    def _prediction_features(self, temperatures):
        cont = self._previous_action["continuous_action"]
        disc = self._previous_action["discrete_action"]
        trend = temperatures - self._previous_temperatures
        centre = .5 * sum(self.comfort_temp_window)
        def value(array, index):
            return 0.0 if index is None else float(array[index])
        return np.asarray([
            [1., (temperatures[i] - centre) / 10., trend[i],
             value(cont, indices.get("supply_air_damper_percentage_command")),
             value(cont, indices.get("reheat_valve_percentage_command")),
             value(disc, ctrl["run_discrete_idx"]), value(cont, ctrl["temp_idx"]),
             value(cont, ctrl.get("pressure_idx")), value(cont, ctrl.get("recirc_idx")),
             value(disc, self.water_discrete_idx), value(cont, self.water_temp_idx),
             value(cont, self.water_pressure_idx)]
            for i, (_, _, indices, ctrl) in enumerate(self._prediction_zones)
        ], dtype=float).reshape(-1, 12)

    def get_action(self, obs):
        temperatures = np.asarray([self._denormalize_zone_temp(obs[index])
                                   for _, index, _, _ in self._prediction_zones])
        self._correction_k.fill(0)
        finite = np.all(np.isfinite(temperatures))
        if finite and self._previous_features is not None:
            x = self._previous_features
            target = temperatures - self._previous_temperatures
            prediction = np.einsum("zi,zi->z", self._weights, x)
            projected = np.einsum("zij,zj->zi", self._covariance, x)
            gain = projected / (1. + np.einsum("zi,zi->z", x, projected))[:, None]
            self._weights += gain * (target - prediction)[:, None]
            self._covariance -= np.einsum("zi,zj->zij", gain, projected)
        features = None
        if finite and self._previous_action is not None and self._previous_temperatures is not None:
            features = self._prediction_features(temperatures)
            prediction = np.einsum("zi,zi->z", self._weights, features)
            if self._prediction_steps >= self.prediction_warmup:
                self._correction_k = self.prediction_blend * np.clip(
                    np.nan_to_num(prediction, nan=0., posinf=0., neginf=0.),
                    -self.prediction_limit_k, self.prediction_limit_k)
        action = super().get_action(obs)
        self._previous_features = features
        self._previous_temperatures = temperatures if finite else None
        self._previous_action = {k: np.array(v, copy=True) for k, v in action.items()}
        self._prediction_steps += 1
        return action

    def _command_zones(self, continuous, zones, obs, low, high):
        if (not self._pi_active or not self.prediction_blend
                or not np.any(self._correction_k)):
            return super()._command_zones(continuous, zones, obs, low, high)
        corrected = np.array(obs, dtype=float, copy=True)
        for vav, index, _ in zones:
            position = self._prediction_positions[vav]
            corrected[index] += self._correction_k[position] / self.zone_air_temperature_sensor_std
        return super()._command_zones(continuous, zones, corrected, low, high)
