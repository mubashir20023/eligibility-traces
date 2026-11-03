"""Correct the divergence statement in the R36-R37 report section (all 23 diverged runs are QRC with lambda >= 0.9,
which QRC-ET was never given, so they do not show that QRC-ET is more robust). Backs up the report first.
Usage (repo root, ~/wbenv):  ~/wbenv/bin/python wb_fix_divergence_text.py [--dry-run]
"""
import argparse, re
from wb_reorganize import REPORT_URL, block_text

NOTE = ("**Divergence (corrected).** A run counts as diverged if the logged Q(S_t, A_t) (scaled reward units; healthy runs "
        "≈ 1-3; one sample every 1,000 steps) is ever non-finite or |Q| > 10⁴ — a fixed cut-off, not a derived bound. All 23 "
        "diverged tuning runs are **QRC with λ = 0.9 or 0.95** (|Q| reached 10⁴-10¹⁸, one NaN, between 270k and 480k steps, "
        "returns mostly collapsed). QRC-ET was only given λ ≤ 0.8, and at λ = 0.24 and 0.8 neither method diverged, so the "
        "divergence counts do **not** show that QRC-ET is more robust.")

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); args = ap.parse_args()
import wandb_workspaces.reports.v2 as wr
url = REPORT_URL.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]
rep = wr.Report.from_url(url)
bl = list(rep.blocks)
changed = 0
for i, b in enumerate(bl):
    t = block_text(b)
    if type(b).__name__ == "MarkdownBlock" and "QRC-ET was more robust:" in t:
        b.text = re.sub(r"QRC-ET was more robust: [^.]*\.", "The divergence counts are explained by λ (see the corrected note below).", t)
        changed += 1
    if type(b).__name__ == "MarkdownBlock" and t.startswith("QRC-ET's best λ was 0.24"):
        if not any(NOTE[:40] in block_text(x) for x in bl):
            bl.insert(i + 1, wr.MarkdownBlock(text=NOTE)); changed += 1
        break
print(f"{changed} change(s)")
if args.dry_run or not changed:
    raise SystemExit
backup = wr.Report.from_url(url); backup.title += " (backup before divergence fix)"; backup.save(clone=True)
print("Backup:", backup.url)
rep.blocks = bl
rep.save()
print("Updated report:", rep.url)
