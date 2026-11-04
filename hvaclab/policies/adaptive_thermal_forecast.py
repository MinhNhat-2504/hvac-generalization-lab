"""Calibrated forecast selection by public occupancy and fan-command state.

Selection error is not a closed-loop reward or safety guarantee. Missing
calibration groups fall back to the unchanged sensor.
"""
import numpy as np

CANDIDATES=('aged_sensor','linear','state_only','commands_only','full_actuators')


class AdaptiveThermalSelector:
    def __init__(self,minimum_samples=100,minimum_model_gain=.05):
        self.minimum_samples=int(minimum_samples);self.minimum_model_gain=float(minimum_model_gain)
        self.selected={};self.radii={};self.counts={}

    @staticmethod
    def group(row):return row['state']+(' occupied' if row['occupancy']>0 else ' empty')

    def fit(self,rows):
        groups={}
        for row in rows:
            if np.all(np.isfinite([row['target']]+[row[k] for k in CANDIDATES])):
                groups.setdefault(self.group(row),[]).append(row)
        self.selected.clear();self.radii.clear();self.counts.clear()
        for group,records in groups.items():
            self.counts[group]=len(records)
            if len(records)<self.minimum_samples:continue
            error={k:float(np.mean([abs(r[k]-r['target']) for r in records])) for k in CANDIDATES}
            baseline=min(CANDIDATES[:2],key=lambda k:error[k])
            best=min(CANDIDATES,key=lambda k:error[k])
            if best not in CANDIDATES[:2] and error[best]>(1-self.minimum_model_gain)*error[baseline]:best=baseline
            self.selected[group]=best
            residuals=sorted(abs(r[best]-r['target']) for r in records)
            self.radii[group]=residuals[min(len(residuals)-1,int(np.ceil((len(residuals)+1)*.95))-1)]
        return self

    def predict(self,row):
        group=self.group(row);chosen=self.selected.get(group,'aged_sensor')
        point=float(row[chosen]);radius=self.radii.get(group)
        if not np.isfinite(point):return None
        return dict(temperature=point,chosen=chosen,interval=None if radius is None else (point-radius,point+radius))
