"""Finalize matched comfort attribution and append local paper without promotion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/flow_comfort_diagnosis')


def finalize():
    from hvaclab.flow_comfort_review import main as review
    review();r=json.loads((ROOT/'verified_review.json').read_text())
    paper=Path('reports/paper/updates_2026_10_27.tex');s=paper.read_text(encoding='utf-8')
    marker='% flow_comfort_diagnosis_'+r['protocol_sha256']
    if marker not in s:
        lines=['',marker,r'\subsection{Matched full-episode comfort regression attribution}',
               'The complete 50-scenario ledger reconciles candidate thermal',
               'occupancy-weighted piecewise regret with the scorer. Phase and',
               'hot/cold candidate levels do not by themselves attribute a delta.',
               'Fresh V10 controls target the largest positive full-episode',
               'comfort change in each cohort and reproduce cached reward terms.',
               'Matched dates, zones, setpoints and occupant exposures are checked.']
        for split,row in r['cohort']['summary'].items():
            lines.append(split+f": {row['positive']} comfort regressions and {row['negative']} improvements; "
                         f"gross positive mean {row['gross_regression_mean']:.9f}, gross negative mean {row['gross_improvement_mean']:.9f}, "
                         f"net comfort change {row['net_delta_mean']:.9f}.")
        for case,row in r['matched_cases'].items():
            lines.append(r'\paragraph{'+case.replace('_',r'\_')+'}')
            lines.append(f"Full comfort change is {row['comfort_delta']:+.9f}.")
            for event,first in row['first_events'].items():
                lines.append(event.replace('_',r'\_')+': first at step '+str(first['step'])+', '+first['clock']+'.')
            for zone,delta in row['zone_deltas'][:5]:
                lines.append(zone.replace('_',r'\_')+f': comfort contribution change {delta:+.9f}.')
            days=row['days'];hot=sum(v['delta_hot'] for v in days.values());cold=sum(v['delta_cold'] for v in days.values())
            lines.append(f"Occupied hot contribution delta {hot:+.9f}; cold delta {cold:+.9f}.")
        probe=r['openloop_probe']
        lines += [r'\paragraph{Fixed-command replay diagnostic.}',
                  'A separate 600-step Phoenix reference was reproduced exactly',
                  'by replaying its saved full command sequence, and its public',
                  'temperature prefix matched the new full V10 trace exactly.',
                  'Replaying those same commands with only the pressure/damper',
                  'allocation transformed gives maximum observed-temperature',
                  f"difference {probe['transformed_openloop_max_temperature_difference']:.9f} K, versus "+
                  f"{probe['closedloop_max_temperature_difference']:.6f} K for the saved closed-loop transform over that prefix.",
                  'This finite-horizon single-case intervention supports feedback',
                  'amplification of tiny allocation differences. It does not identify',
                  'which controller nonlinearity is dominant, nor prove a remedy.',
                  'A guard that only reacts after a changed fan run state may be late.']
        lines+=['First-divergence ordering is an association, not identification',
                'of a causal physical mechanism. The previous pointwise airflow',
                'audit compares with the parent request on the current trajectory,',
                'not with an independently evolving V10 building. No revised',
                'controller, notebook or relaxed comfort gate is promoted here.']
        paper.write_text(s+'\n'.join(lines)+'\n',encoding='utf-8')
    build=(ROOT/'paper_build').resolve();build.mkdir(parents=True,exist_ok=True)
    with (ROOT/'paper_final.log').open('w') as log:
        for _ in range(2):subprocess.run([shutil.which('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'main.tex'],cwd='reports/paper',stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(build/'main.pdf','reports/paper/main.pdf')
    context=Path('PROJECT_CONTEXT.md');old=context.read_text(encoding='utf-8');cm='flow_comfort_final_'+r['protocol_sha256']
    if cm not in old:
        lines=['','- Comfort attribution complete50 candidate ledger +2 matched full V10 controls.']
        for case,row in r['matched_cases'].items():
            lines.append('  '+case+' delta='+str(row['comfort_delta'])+' first events='+json.dumps(row['first_events']))
        lines+=['  Paper appended/PDF rebuilt. Temporal ordering not causal identification. '+cm]
        context.write_text(old+'\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'finalization.json',dict(status='complete',completed=2,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=r['protocol_sha256'],analysis_sha256=r['analysis_sha256'],
        paper_sha256=digest('reports/paper/main.pdf'),paper_source_sha256=digest(paper),report_sha256=digest('reports/FLOW_COMFORT_DIAGNOSIS.md')))
    print('COMFORT_DIAGNOSIS_PAPER_FINALIZED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with RunLock(ROOT/'finalizer.lock'):
        if args.watch:
            while True:
                try:state=json.loads((ROOT/'status.json').read_text())
                except (OSError,ValueError):state={}
                if state.get('status')=='complete':break
                time.sleep(15)
        elif json.loads((ROOT/'status.json').read_text()).get('status')!='complete':raise RuntimeError('matched traces not complete')
        finalize()


if __name__=='__main__':main()
