"""Make every table in the report landscape and fully visible (no scrolling), and keep plots side by side.

For every table panel (MediaBrowser): full report width and a height that fits all its rows.
For every pair of line plots: two half-width panels on one row.
The report is switched to the wide ("fluid") layout. A backup copy is saved first.

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_fix_layout.py --dry-run     # show the new sizes, change nothing
    ~/wbenv/bin/python wb_fix_layout.py               # back up, then apply
    ~/wbenv/bin/python wb_fix_layout.py --extra 2     # if a table still scrolls: add 2 more height units to every table
"""
import argparse

REPORT_URL = ("https://forge.coreweave.com/mubbashirahmed/qrc-et-algo3-minatar/reports/"
              "QRC-vs-Experimental-QRC-ET-%7C-Five-Game-Screen-and-Breakout-Follow-up--VmlldzoxODExNTc2Nw==")

# number of data rows in each table (titles from the report); used to size the panel height
ROWS = {
    "All Five Games | 1M Steps": 5,
    "All Five Games | Individual Seed Details": 10,
    "Asterix | Seed 601": 2, "Breakout | Seed 601": 2, "Freeway | Seed 601": 2,
    "Seaquest | Seed 601": 2, "SpaceInvaders | Seed 601": 2,
    "Breakout | The Gap Persists at 3M Steps": 3,
    "Breakout | 3M | Seed 601 Metrics": 2,
    "Learning Efficiency, Diagnostics and Cost": 5,
    "What Changed and What Remains Different": 9,
    "Progress and Next Decision": 5,
    "Freeway | 15 Fresh Seeds": 5,
    "Freeway | Score by Learning Rate": 3,
    "Multi-Chain | Value Error (lower is better)": 8,
    "Θ Target and Learning Rate | m=32, σ=1": 6,
    "Reproduction | R28-R31": 4,
}
LONG_TEXT = {"What Changed and What Remains Different", "Progress and Next Decision"}   # wrapped cells need more room


def table_height(title, extra):
    rows = ROWS.get(title, 8)
    per_row = 1.6 if title in LONG_TEXT else 1.0
    return int(round(3 + rows * per_row)) + extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]
    import wandb_workspaces.reports.v2 as wr

    if not args.dry_run and not args.no_backup:
        backup = wr.Report.from_url(args.url)
        backup.title = backup.title + " (backup before layout fix)"
        backup.save(clone=True)
        print("Backup:", backup.url)

    rep = wr.Report.from_url(args.url)
    rep.width = "fluid"
    for b in rep.blocks:
        if type(b).__name__ != "PanelGrid":
            continue
        tables = [p for p in b.panels if type(p).__name__ == "MediaBrowser"]
        plots = [p for p in b.panels if type(p).__name__ != "MediaBrowser"]
        y = 0
        for p in tables:
            h = table_height(p.title, args.extra)
            p.layout = wr.Layout(x=0, y=y, w=24, h=h)
            p.num_columns = 1
            y += h
            print(f"table  w=24 h={h:<3} {p.title}")
        for i, p in enumerate(plots):
            p.layout = wr.Layout(x=12 * (i % 2), y=y + 9 * (i // 2), w=12, h=9)
            print(f"plot   w=12 h=9   {getattr(p, 'title', '')}")
    if args.dry_run:
        return
    rep.save()
    print("Updated report:", rep.url)


if __name__ == "__main__":
    main()
