"""Idempotent local paper/report completion for read-only plant telemetry."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/plant_runtime_short')


def finalize():
    from hvaclab.plant_runtime_review import main as review
    review();result=json.loads((ROOT/'verified_review.json').read_text());fingerprint=digest(ROOT/'protocol.json')
    marker='% plant_runtime_verified_'+fingerprint;paper=Path('reports/paper/updates_2026_10_27.tex');source=paper.read_text(encoding='utf-8')
    if marker not in source:
        lines=['',marker,'\\subsection{Verified plant runtime diagnosis, 5 November}',
            'All four instrumented 2016-step rollouts reproduced the prior whole-prefix',
            'and phase reward terms. The shadow matched 4032 V10 commands. Actual',
            'device run states matched commands, and running-reason classifications',
            'and hold counters reconciled. AHU hours below are summed machine-hours',
            'over handlers, whereas pump hours are elapsed single-system hours.']
        for name,case in result['cases'].items():
            off=case['phase_comparison']['offhours'];a=off['control'];b=off['feedback'];delta=off['delta'];meters=off['meter_delta_Wh']
            lines.append('\\paragraph{'+name.replace('_','\\_')+'}')
            lines.append(f"Offhours AHU hours were {a['ahu_handler_hours']:.4f} for V10 and {b['ahu_handler_hours']:.4f} with feedback; actual hold-override hours were {a['hold_handler_hours']:.4f} and {b['hold_handler_hours']:.4f}. Pump hours were {a['pump_hours']:.4f} and {b['pump_hours']:.4f}.")
            lines.append(f"Inherited-request hours changed by {delta['parent_request_handler_hours']:+.4f}, hold-override hours by {delta['hold_handler_hours']:+.4f}, and heating-airflow override hours by {delta['heat_override_handler_hours']:+.4f}.")
            lines.append('Metered offhours energy changes in kWh (feedback minus V10): '+', '.join(k+' '+format(v/1000.,'+.6f') for k,v in meters.items())+'.')
            parts=case['offhours_delta_by_command_state']
            lines.append('Normalized weighted energy-plus-carbon changes were '+format(parts['identical_run_commands']['weighted_energy_carbon_delta'],'+.8f')+' on steps with identical AHU/pump run vectors and '+format(parts['changed_run_commands']['weighted_energy_carbon_delta'],'+.8f')+' on steps with changed run vectors.')
            lines.append('Additional running handler-steps were classified as '+str(case['offhours_extra_on_reason']).replace('_','\\_')+'.')
        lines+=['These are read-only associations on selected short dev trajectories.',
            'They do not establish that a hold is unnecessary or that changing it',
            'would preserve comfort. Identical binary run vectors may still have',
            'different airflow, pressure, valve settings, thermal state and billed',
            'load. Raw energy units and normalized tariff/carbon contributions are',
            'reported separately. No hold rule, control policy or submission changed.']
        paper.write_text(source+'\n'.join(lines)+'\n',encoding='utf-8')
    compiler=shutil.which('pdflatex')
    if not compiler:raise RuntimeError('pdflatex unavailable')
    build=(ROOT/'paper_build').resolve();build.mkdir(parents=True,exist_ok=True)
    with (ROOT/'paper_final.log').open('w') as log:
        for _ in range(2):subprocess.run([compiler,'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'main.tex'],cwd='reports/paper',stdout=log,stderr=subprocess.STDOUT,check=True)
    target=Path('reports/paper/main.pdf');temp=target.with_suffix('.pdf.plant.tmp');shutil.copyfile(build/'main.pdf',temp);temp.replace(target)
    context=Path('PROJECT_CONTEXT.md');old=context.read_text(encoding='utf-8');cm='plant_runtime_final_'+fingerprint
    if cm not in old:
        lines=['',f'- 2026-11-05: plant runtime diagnosis4/4 verified; old rewards/phase totals',
            '  reproduced and4032 V10 commands exact. Report reports/PLANT_RUNTIME_REVIEW.md.',
            '  Actual states/counters/run-reason partition checked; no policy/hold change.']
        for name,case in result['cases'].items():
            off=case['phase_comparison']['offhours'];lines.append('  '+name+' offhours runtime delta '+json.dumps(off['delta'],sort_keys=True))
            lines.append('  offhours meter deltaWh '+json.dumps(off['meter_delta_Wh'],sort_keys=True))
        lines+=['  Paper appended/rebuilt locally. Associations are not evidence to remove',
                '  hold without a comfort-tested intervention. '+cm]
        context.write_text(old+'\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'finalization.json',dict(status='complete',protocol_sha256=fingerprint,report_sha256=digest('reports/PLANT_RUNTIME_REVIEW.md'),paper_sha256=digest(target),paper_source_sha256=digest(paper)))
    print('PLANT_REPORT_AND_PDF_FINALIZED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    with RunLock(ROOT/'finalizer.lock'):
        if args.watch:
            while True:
                try:state=json.loads((ROOT/'status.json').read_text())
                except (OSError,ValueError):state={}
                if state.get('status')=='complete':break
                time.sleep(15)
        elif json.loads((ROOT/'status.json').read_text()).get('status')!='complete':raise RuntimeError('study not complete')
        finalize()


if __name__=='__main__':main()
