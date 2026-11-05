"""Finalize complete-cohort results, optionally package, then append local paper."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from hvaclab.coverage import digest,write_json
from hvaclab.pi_calibration import RunLock

ROOT=Path('experiments/runs/flow_equivalent_full')


def finalize():
    from hvaclab.flow_equivalent_full_review import main as review
    review();result=json.loads((ROOT/'verified_review.json').read_text())
    notebook=None
    if result['eligible_for_packaging']:
        with (ROOT/'package.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','hvaclab.flow_equivalent_package'],stdout=log,stderr=subprocess.STDOUT,check=True)
        notebook=Path('experiments/submission/flow_equivalent_full/submission.ipynb')
        validation=json.loads(notebook.with_name('validation.json').read_text())
        if digest(notebook)!=validation['notebook_sha256']:raise ValueError('notebook artifact hash changed')
        result.update(notebook_exported=True,submission_notebook=notebook.as_posix())
        write_json(ROOT/'verified_review.json',result)
    paper=Path('reports/paper/updates_2026_10_27.tex');source=paper.read_text(encoding='utf-8')
    marker='% flow_equivalent_full_verified_'+digest(ROOT/'protocol.json')
    if marker not in source:
        lines=['',marker,r'\subsection{Complete flow-equivalent training and development validation}',
               'All 30 training and 20 dev candidate episodes completed 8640 steps.',
               'The earlier failed short comfort gate is retained as historical',
               'evidence; full measurement was explicitly requested afterward.',
               r'\begin{tabular}{lrrrr}',r'Cohort & Cases & V10 & Candidate & Change (percent) \\ \hline']
        for split,row in result['statistics'].items():
            lines.append(f"{split} & {row['n']} & {row['v10']['loss']:.8f} & {row['candidate']['loss']:.8f} & {row['change_pct']:+.4f}"+r' \\')
        lines.append(r'\end{tabular}')
        for split,row in result['statistics'].items():
            lines.append(split+f": comfort {row['v10']['comfort']:.8f} to {row['candidate']['comfort']:.8f}; "
                         f"{row['improved']} improved and {row['regressed']} regressed scenarios.")
            for case in row['comparisons'][:3]:
                lines.append((case['building']+'/'+case['weather']).replace('_',r'\_')+
                             f": loss {case['v10']:.8f} to {case['candidate']:.8f}.")
        lines.append(f"Full mean gate passed: {result['full_mean_gate']}. Full paired morning cases: {result['paired_full_morning_cases']} of 50. "
                     f"All-50 morning gate passed: {result['all50_morning_gate']}. Notebook exported after standalone parity: {result['notebook_exported']}.")
        lines+=['Remaining baseline morning traces are conditional on passing both',
                'full mean gates. Missing morning controls never count as a pass.',
                'These are local training/dev results, with no hidden-score claim.']
        paper.write_text(source+'\n'.join(lines)+'\n',encoding='utf-8')
    build=(ROOT/'paper_build').resolve();build.mkdir(parents=True,exist_ok=True)
    with (ROOT/'paper_final.log').open('w') as log:
        for _ in range(2):subprocess.run([shutil.which('pdflatex'),'-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory',str(build),'main.tex'],cwd='reports/paper',stdout=log,stderr=subprocess.STDOUT,check=True)
    target=Path('reports/paper/main.pdf');shutil.copyfile(build/'main.pdf',target)
    context=Path('PROJECT_CONTEXT.md');old=context.read_text(encoding='utf-8');cm='flow_equivalent_full_final_'+result['protocol_sha256']
    if cm not in old:
        context.write_text(old+'\n- Full flow-equivalent validation complete30train+20dev8640-step episodes.\n  '+
            json.dumps({s:{k:v for k,v in row.items() if k!='comparisons'} for s,row in result['statistics'].items()})+
            '\n  Eligible_for_packaging='+str(result['eligible_for_packaging'])+'; notebook='+str(notebook)+'.\n  Paper appended/PDF rebuilt locally. '+cm+'\n',encoding='utf-8')
    count=json.loads((ROOT/'status.json').read_text())['completed']
    write_json(ROOT/'finalization.json',dict(status='complete',completed=count,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        protocol_sha256=result['protocol_sha256'],paper_sha256=digest(target),paper_source_sha256=digest(paper),
        report_sha256=digest('reports/FLOW_EQUIVALENT_FULL_REVIEW.md'),
        decision=dict(eligible_for_packaging=result['eligible_for_packaging'],submission_notebook=notebook.as_posix() if notebook else None)))
    print('FULL_REPORT_PAPER_AND_DELIVERY_FINALIZED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with RunLock(ROOT/'finalizer.lock'):
        if args.watch:
            while True:
                try:state=json.loads((ROOT/'status.json').read_text())
                except (OSError,ValueError):state={}
                if state.get('status')=='complete':break
                time.sleep(15)
        elif json.loads((ROOT/'status.json').read_text()).get('status')!='complete':raise RuntimeError('full cohorts not complete')
        finalize()


if __name__=='__main__':main()
