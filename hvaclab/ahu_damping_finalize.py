"""Append verified factorial damping results and rebuild the local paper."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/ahu_damping_short')


def finalize():
    from hvaclab.ahu_damping_review import main as review
    review();r=json.loads((ROOT/'verified_review.json').read_text())
    paper=Path('reports/paper/updates_2026_10_27.tex');s=paper.read_text(encoding='utf-8')
    marker='% ahu_damping_verified_'+r['protocol_sha256']
    if marker not in s:
        lines=['',marker,r'\subsection{Verified bounded AHU damping factorial screen}',
               'Twelve matched 4032-step rollouts compared V10, allocation only,',
               'damping only and both interventions. The two control trajectories',
               'reproduced the previously recorded prefixes on all three cases.',
               f"The V10 shadow matched {r['exact_v10_shadow_steps']} actions exactly.",
               r'\begin{tabular}{lrrrr}',r'Arm & Loss & Comfort & Energy & Carbon \\ \hline']
        for arm,row in r['means'].items():lines.append(arm.replace('_',r'\_')+' & '+' & '.join(f'{row[k]:.8f}' for k in ('loss','comfort','energy','carbon'))+r' \\')
        lines.append(r'\end{tabular}')
        g=r['gate'];lines.append(f"Combined loss changes versus V10 and allocation-only are {g['change_vs_v10_pct']:+.4f} and {g['change_vs_flow_pct']:+.4f} percent. "
            f"Mean comfort gate passed: {g['mean_comfort_no_worse_vs_both']}. Morning guard passed: {g['morning_safe_vs_v10']}. Overall short gate passed: {g['passes_short_gate']}.")
        for case,row in r['sensitivity'].items():
            a,b=row['unfiltered'],row['filtered']
            lines.append(r'\paragraph{'+case.replace('_',r'\_')+'}')
            lines.append(f"Allocation-induced maximum temperature difference without and with filtering is {a['max_temperature_difference']:.6f} and {b['max_temperature_difference']:.6f} K; "
                f"mean stepwise maximum differences are {a['mean_max_temperature_difference']:.6f} and {b['mean_max_temperature_difference']:.6f} K. "
                f"AHU run mismatch counts are {a['ahu_run_mismatch_steps']} and {b['ahu_run_mismatch_steps']}.")
            combined=r['cases']['flow_smooth'][case]
            lines.append(f"Combined candidate full-prefix loss {combined['candidate']['loss']:.8f}; V10 {combined['v10']['loss']:.8f}. "
                f"Filtered handler steps: {combined['diagnostics']['ahu_filtered_steps']}; failed daily morning means: {len(combined['morning_vs_v10']['failed_mornings'])}.")
        lines+=['Sensitivity values compare independently evolving trajectories,',
                'not pointwise predictor error. Filtering can alter thermal state',
                'and energy demand despite unchanged zone-PI equations. These',
                'selected reused two-week prefixes do not establish full-cohort',
                'generalization or a hidden-score gain. No automatic full sweep',
                'or notebook is generated. Previous results remain preserved.']
        paper.write_text(s+'\n'.join(lines)+'\n',encoding='utf-8')
    build=(ROOT/'paper_build').resolve();build.mkdir(parents=True,exist_ok=True)
    with (ROOT/'paper_final.log').open('w') as log:
        for _ in range(2):subprocess.run([shutil.which('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'main.tex'],cwd='reports/paper',stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(build/'main.pdf','reports/paper/main.pdf')
    p=Path('PROJECT_CONTEXT.md');old=p.read_text(encoding='utf-8');cm='ahu_damping_final_'+r['protocol_sha256']
    if cm not in old:p.write_text(old+'\n- AHU damping factorial12/12 prefixes verified; gate='+json.dumps(r['gate'])+'\n  Paper/PDF appended; no full sweep/notebook. '+cm+'\n',encoding='utf-8')
    write_json(ROOT/'finalization.json',dict(status='complete',completed=12,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=r['protocol_sha256'],paper_sha256=digest('reports/paper/main.pdf'),paper_source_sha256=digest(paper),
        report_sha256=digest('reports/AHU_DAMPING_REVIEW.md'),decision=r['gate']))
    print('AHU_DAMPING_REPORT_AND_PAPER_FINALIZED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with RunLock(ROOT/'finalizer.lock'):
        if args.watch:
            while True:
                try:state=json.loads((ROOT/'status.json').read_text())
                except (OSError,ValueError):state={}
                if state.get('status')=='complete':break
                time.sleep(15)
        elif json.loads((ROOT/'status.json').read_text()).get('status')!='complete':raise RuntimeError('damping cohort not complete')
        finalize()


if __name__=='__main__':main()
