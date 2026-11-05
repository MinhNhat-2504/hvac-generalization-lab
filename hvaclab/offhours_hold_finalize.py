"""Append verified hold-refresh results and rebuild the local paper."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from hvaclab.coverage import digest, write_json
from hvaclab.pi_calibration import RunLock

ROOT = Path('experiments/runs/offhours_hold_short')


def finalize():
    from hvaclab.offhours_hold_review import main as review
    review()
    result = json.loads((ROOT / 'verified_review.json').read_text())
    fingerprint = digest(ROOT / 'protocol.json')
    marker = '% offhours_hold_verified_' + fingerprint
    paper = Path('reports/paper/updates_2026_10_27.tex')
    source = paper.read_text(encoding='utf-8')
    if marker not in source:
        lines = ['', marker,
                 r'\subsection{Verified empty-offhours hold-refresh intervention, 5 November}',
                 'Six matched 2016-step rollouts compared V10 with caps on new',
                 'cooling-hold refreshes outside the protected PI window when',
                 'public occupancy was zero. Caps were 6 steps (30 minutes)',
                 'and 24 steps (2 hours), compared with the normal 432 steps.',
                 'Carried timer memory was not trimmed. The control shadow',
                 f"matched {result['exact_v10_shadow_steps']} V10 actions exactly.",
                 r'\begin{tabular}{lrrrr}',
                 r'Arm & Loss & Comfort & Energy & Carbon \\ \hline']
        for arm, row in result['means'].items():
            name = arm.replace('_', r'\_')
            lines.append(name + ' & ' + ' & '.join(f'{row[k]:.8f}' for k in
                         ('loss', 'comfort', 'energy', 'carbon')) + r' \\')
        lines.append(r'\end{tabular}')
        for arm, gate in result['gates'].items():
            lines.append(r'\paragraph{' + arm.replace('_', r'\_') + '}')
            lines.append(f"Mean prefix loss changed by {gate['change_pct']:+.4f} percent. "
                         f"The predeclared short gate passed: {gate['passes_short_gate']}. "
                         f"Mean comfort was no worse: {gate['comfort_no_worse']}. "
                         f"Every paired morning was no worse: {gate['morning_safe_in_both_cases']}.")
            for case, row in result['cases'][arm].items():
                morning = row['morning']
                lines.append(case.replace('_', r'\_') + ': loss ' +
                             f"{row['v10']['loss']:.8f} to {row['candidate']['loss']:.8f}; " +
                             f"{morning['days']} paired mornings, " +
                             f"{len(morning['failed_mornings'])} mean-comfort regressions, " +
                             f"{len(morning['worst_interval_regressions'])} worst-interval regressions. " +
                             f"Capped refreshes: {row['hold_audit']['capped_refreshes']}.")
                for failed in morning['failed_mornings']:
                    lines.append(f"Morning {failed['day']}: weighted comfort "
                                 f"{failed['baseline']:.8f} to {failed['candidate']:.8f}.")
        lines += ['Morning windows cover the first two reward-clock hours after',
                  'the first observed occupied interval on each local date,',
                  'including empty intervals inside those windows. Dates and',
                  'exposures matched. Worst individual occupied intervals are',
                  'reported separately from the daily-mean gate. These two',
                  'previously used dev prefixes do not establish unseen-building',
                  'generalization or a hidden Kaggle score. No automatic full',
                  'sweep or submission notebook was generated.']
        paper.write_text(source + '\n'.join(lines) + '\n', encoding='utf-8')
    compiler = shutil.which('pdflatex')
    if not compiler:
        raise RuntimeError('pdflatex unavailable')
    build = (ROOT / 'paper_build').resolve()
    build.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'paper_final.log').open('w') as log:
        for _ in range(2):
            subprocess.run([compiler, '-no-shell-escape', '-interaction=nonstopmode',
                            '-halt-on-error', '-output-directory', str(build), 'main.tex'],
                           cwd='reports/paper', stdout=log, stderr=subprocess.STDOUT, check=True)
    target = Path('reports/paper/main.pdf')
    temporary = target.with_suffix('.pdf.offhours.tmp')
    shutil.copyfile(build / 'main.pdf', temporary)
    temporary.replace(target)
    context = Path('PROJECT_CONTEXT.md')
    old = context.read_text(encoding='utf-8')
    cm = 'offhours_hold_final_' + fingerprint
    if cm not in old:
        context.write_text(old + '\n- 2026-11-05: offhours hold-refresh6/24 study6/6 verified; '
                           + str(result['exact_v10_shadow_steps']) + ' exact V10 actions.\n'
                           + '  Gates: ' + json.dumps(result['gates'], sort_keys=True) + '\n'
                           + '  Report reports/OFFHOURS_HOLD_REFRESH_REVIEW.md; paper appended/PDF rebuilt.\n'
                           + '  Two reused dev prefixes only; no full sweep or notebook. ' + cm + '\n',
                           encoding='utf-8')
    write_json(ROOT / 'finalization.json', dict(
        status='complete', protocol_sha256=fingerprint,
        report_sha256=digest('reports/OFFHOURS_HOLD_REFRESH_REVIEW.md'),
        paper_sha256=digest(target), paper_source_sha256=digest(paper)))
    print('OFFHOURS_REPORT_AND_PDF_FINALIZED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    with RunLock(ROOT / 'finalizer.lock'):
        if args.watch:
            while True:
                try:
                    state = json.loads((ROOT / 'status.json').read_text())
                except (OSError, ValueError):
                    state = {}
                if state.get('status') == 'complete':
                    break
                time.sleep(15)
        elif json.loads((ROOT / 'status.json').read_text()).get('status') != 'complete':
            raise RuntimeError('study not complete')
        finalize()


if __name__ == '__main__':
    main()
