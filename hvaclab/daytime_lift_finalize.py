"""Append verified morning recovery allocation results without promoting a prefix winner."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/daytime_lift_short')


def finalize():
    from hvaclab.daytime_lift_review import main as review
    review();r=json.loads((ROOT/'verified_review.json').read_text())
    paper=Path('reports/paper/updates_2026_10_27.tex');s=paper.read_text(encoding='utf-8')
    marker='% daytime_lift_verified_'+r['protocol_sha256']
    if marker not in s:
        lines=['',marker,r'\subsection{Verified public-input morning recovery allocation screen}',
               'Sixteen fixed 4032-step rollouts compared V10, original thermal',
               'allocation, two-hour and four-hour morning recovery protection.',
               f"All {r['exact_v10_shadow_steps']} V10 shadow commands matched.",
               r'\begin{tabular}{lrrrr}',r'Arm & Loss & Comfort & Energy & Carbon \\ \hline']
        for arm,row in r['means'].items():lines.append(arm.replace('_',r'\_')+' & '+' & '.join(f'{row[k]:.8f}' for k in ('loss','comfort','energy','carbon'))+r' \\')
        lines.append(r'\end{tabular}')
        for arm,g in r['gates'].items():
            lines.append(r'\paragraph{'+arm.replace('_',r'\_')+'}')
            lines.append(f"Mean loss changed by {g['change_vs_v10_pct']:+.4f} percent versus V10. "
                         f"Mean comfort gate passed: {g['mean_comfort_no_worse']}. Morning guard: {g['morning_safe']}. "
                         f"Predeclared five-percent material-gain gate: {g['material5pct_loss_gain']}. Overall short gate: {g['passes_short_gate']}.")
            for case,row in r['cases'][arm].items():
                lines.append(case.replace('_',r'\_')+f": loss {row['v10']['loss']:.8f} to {row['candidate']['loss']:.8f}; "
                    f"comfort {row['v10']['comfort']:.8f} to {row['candidate']['comfort']:.8f}; "
                    f"{len(row['morning_vs_v10']['failed_mornings'])} daily morning regressions; "
                    f"{row['diagnostics']['lift_handler_steps']} retained thermal-allocation steps; {row['diagnostics']['comfort_guard_steps']} temperature-triggered fallback decisions (disabled in these arms).")
        lines+=['No automatic full sweep or submission notebook follows from',
                'these selected reused two-week prefixes. The cooling allocation',
                'equation preserves estimated current-state cooling based on',
                'public observations. Delayed sensing and implicit within-step',
                'dynamics can still alter actual loads and comfort. Previous',
                'measurements remain preserved. No hidden-score claim is made.']
        paper.write_text(s+'\n'.join(lines)+'\n',encoding='utf-8')
    build=(ROOT/'paper_build').resolve();build.mkdir(parents=True,exist_ok=True)
    with (ROOT/'paper_final.log').open('w') as log:
        for _ in range(2):subprocess.run([shutil.which('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'main.tex'],cwd='reports/paper',stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copyfile(build/'main.pdf','reports/paper/main.pdf')
    p=Path('PROJECT_CONTEXT.md');old=p.read_text(encoding='utf-8');cm='daytime_lift_final_'+r['protocol_sha256']
    if cm not in old:p.write_text(old+'\n- Guarded lift allocation16/16 prefixes verified; gates='+json.dumps(r['gates'])+'\n  Paper/PDF appended; no full sweep/notebook. '+cm+'\n',encoding='utf-8')
    write_json(ROOT/'finalization.json',dict(status='complete',completed=16,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=r['protocol_sha256'],paper_sha256=digest('reports/paper/main.pdf'),paper_source_sha256=digest(paper),
        report_sha256=digest('reports/DAYTIME_LIFT_REVIEW.md'),decision=r['gates']))
    print('DAYTIME_LIFT_REPORT_AND_PAPER_FINALIZED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with RunLock(ROOT/'finalizer.lock'):
        if args.watch:
            while True:
                try:state=json.loads((ROOT/'status.json').read_text())
                except (OSError,ValueError):state={}
                if state.get('status')=='complete':break
                time.sleep(15)
        elif json.loads((ROOT/'status.json').read_text()).get('status')!='complete':raise RuntimeError('band PI cohort not complete')
        finalize()


if __name__=='__main__':main()
