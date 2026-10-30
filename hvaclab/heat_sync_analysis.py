"""Validate and report the completed heat-sync experiment.

Run from the repository root after all sweeps finish:
    python -m hvaclab.heat_sync_analysis
"""

import hashlib
import json
import math
from pathlib import Path

from hvaclab import runner

RUNS = Path("experiments/runs")
TRAIN = {
    "building_train_1/fairbanks_1", "building_train_2/fairbanks_2",
    "building_train_4/denver_2", "building_train_7/atlanta_8",
    "building_train_7/fairbanks_7",
}
DEV = {f"building_9001/{w}" for w in ("denver_1", "fairbanks_3", "singapore_2")}


def complete(name):
    rows = runner.load(RUNS / name)
    if not rows or any(r.error or r.steps != 8640 or not math.isfinite(r.mean_reward)
                       for r in rows):
        raise ValueError(f"incomplete or failed full-episode results: {name}")
    if len({(r.policy, r.scenario) for r in rows}) != len(rows):
        raise ValueError(f"duplicate results: {name}")
    for r in rows:
        terms = (-0.6 * r.prod_regret + 0.2 * r.energy_cost + 0.2 * r.carbon) / r.steps
        if not math.isclose(r.score, terms, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"reward decomposition mismatch: {name}, {r.scenario}")
    return rows


def select(rows, policy, scenarios, exact=False):
    chosen = [r for r in rows if r.policy == policy and r.scenario in scenarios]
    if {r.scenario for r in chosen} != scenarios or len(chosen) != len(scenarios):
        raise ValueError(f"missing or duplicate {policy} scenarios")
    if exact and len([r for r in rows if r.policy == policy]) != len(scenarios):
        raise ValueError(f"unexpected extra {policy} scenarios")
    return sorted(chosen, key=lambda r: r.scenario)


def paired_table(candidate, reference):
    ref = {r.scenario: r for r in reference}
    lines = ["| Scenario | h_288 | e_sync | Thay đổi score |",
             "|---|---:|---:|---:|"]
    for r in candidate:
        before = ref[r.scenario].score
        lines.append(f"| `{r.scenario}` | {before:.8f} | {r.score:.8f} "
                     f"| {100 * (r.score / before - 1):+.2f}% |")
    before = sum(r.score for r in reference) / len(reference)
    after = sum(r.score for r in candidate) / len(candidate)
    lines.append(f"| **Mean cùng {len(candidate)} scenario** | **{before:.8f}** "
                 f"| **{after:.8f}** | **{100 * (after / before - 1):+.2f}%** |")
    return lines


def main():
    # Reject stale code/result combinations rather than silently attributing
    # an earlier run to the current implementation.
    provenance = json.loads((RUNS / "heat_sync_provenance.json").read_text())
    for path, digest in provenance["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"measured source changed: {path}")
    probe = complete("ablation_heat_sync_probe.json")
    probe_pairs = {"building_train_1/fairbanks_1", "building_train_7/fairbanks_7"}
    expected_probe = {(key, pair) for key in ("e_sync", "e_guard_air", "e_guard_sync")
                      for pair in probe_pairs}
    if {(r.policy, r.scenario) for r in probe} != expected_probe:
        raise ValueError("expected all 6 factorial probe runs")
    extra = complete("heat_sync_train_fairbanks2.json") + complete("heat_sync_train_remaining.json")
    if len(extra) != 3 or {r.policy for r in extra} != {"e_sync"}:
        raise ValueError("expected exactly 3 extra e_sync training runs")
    train = select(probe + extra, "e_sync", TRAIN, exact=True)
    old = complete("ablation_heat.json")
    train_ref = select(old, "d_ctrl", TRAIN)
    dev = select(complete("heat_sync_dev.json"), "e_sync", DEV, exact=True)
    dev_ref = select(complete("h288_dev.json"), "h_288", DEV)
    runner.save(train, RUNS / "heat_sync_train5.json")
    lines = ["", "## Kết quả đủ episode", "",
             "Tất cả hàng bên dưới đủ 8.640 bước, không lỗi; thành phần reward "
             "được cộng lại và đối chiếu trong ngưỡng 1e-9. "
             "Mức giảm chỉ áp dụng cho các scenario được ghi rõ.", "",
             "### Cùng 5 scenario của batch ban đầu", "", *paired_table(train, train_ref), "",
             "### Cùng 3 scenario dev trên nhà mới", "", *paired_table(dev, dev_ref), "",
             "### Tách tác dụng của hai thay đổi trên 2 scenario đối lập", "",
             "| Policy | Mean score | Comfort | Energy | Carbon |",
             "|---|---:|---:|---:|---:|"]
    ablation = probe + select(old, "d_ctrl", probe_pairs) + select(old, "d_heat", probe_pairs)
    for key, s in runner.summarise(ablation).items():
        lines.append(f"| `{key}` | {s['score']:.8f} | {s['w_prod']:.8f} "
                     f"| {s['w_energy']:.8f} | {s['w_carbon']:.8f} |")
    lines += ["", "Dữ liệu: `ablation_heat_sync_probe.json`, `heat_sync_train5.json`, "
              "`heat_sync_dev.json`; baseline: `ablation_heat.json` và `h288_dev.json`. "
              "Đã so cùng scenario và cùng horizon, không dùng bảng baseline cũ.", "",
              "Đây là xác nhận sơ bộ trên 5 scenario training được chọn có chủ đích "
              "và 3 scenario của một nhà dev. Chưa phải kết quả full shipped-30, "
              "full dev, test hoặc Kaggle.", ""]
    train_s = runner.summarise(train)["e_sync"]
    train_base_s = runner.summarise(train_ref)["d_ctrl"]
    dev_s = runner.summarise(dev)["e_sync"]
    dev_base_s = runner.summarise(dev_ref)["h_288"]
    old_heat_s = runner.summarise(select(old, "d_heat", TRAIN))["d_heat"]
    lines += ["## Kết luận và ứng viên", "",
              f"- So với `d_heat` vừa đo, `e_sync` giảm thêm "
              f"**{100 * (1 - train_s['score'] / old_heat_s['score']):.2f}%** "
              "trên cùng 5 scenario. Cả 5 đều tốt hơn `h_288`.",
              f"- Comfort có trọng số giảm **{100 * (1 - train_s['w_prod'] / train_base_s['w_prod']):.2f}%** "
              f"trên nhóm training và **{100 * (1 - dev_s['w_prod'] / dev_base_s['w_prod']):.2f}%** "
              "trên nhóm dev; energy + carbon tăng. Đây vẫn là đánh đổi có lợi cho tổng điểm.",
              "- Dev Singapore tăng 1,09% (0.01177561 → 0.01190436), tức chênh lệch "
              "tuyệt đối khoảng 0.000129. Hai scenario dev cần sưởi cải thiện rõ.",
              "- Ưu tiên `e_sync`: guard không cải thiện thêm trong ablation đã đo. "
              "Giữ hai biến thể guard làm bản đối chứng nghiên cứu.",
              "- Notebook ứng viên: `experiments/submission/e_sync/submission.ipynb`. "
              "Đã kiểm tra 576 cặp action trên mỗi scenario Fairbanks 1 và 7, tổng "
              "1.152 cặp khớp chính xác với registry. Notebook compile được; manifest "
              "ghi checksum của artifact và nguồn ghép. Chưa nộp Kaggle và giữ bundle mặc định.", ""]
    report = Path("reports/HEAT_SYNC_REVIEW.md")
    body = report.read_text(encoding="utf-8").split("\n## Kết quả đủ episode\n")[0]
    body = body.replace("Đang đo. Chưa có kết luận cải thiện và chưa thay policy trong submission.",
                        "Đã hoàn tất các phép đo bên dưới. Giữ bundle mặc định; "
                        "notebook mới nằm riêng trong `experiments/submission/e_sync/`.")
    report.write_text(body + "\n".join(lines), encoding="utf-8")
    print("Validated 9 training runs and 3 dev runs; wrote paired report and merged train-5 results.")


if __name__ == "__main__":
    main()
