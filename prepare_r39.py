"""Pick the representation sources for R39: for each game, the best-scoring R37 seed of the best QRC run and of the
best QRC-ET run (final score = last 10% of 5M steps). Writes r39_sources.txt:  game source path_to_cleanrl_model
Usage (from stream-rep-rl, after R37):  python prepare_r39.py
"""
import csv, glob, os, re
import numpy as np

lines = []
for game in ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]:
    for src in ("qrc", "et"):
        best = None
        for p in glob.glob(f"logs/**/{game}-v1__R37_{src}_best_lr*__*/training_data.csv", recursive=True):
            rows = list(csv.DictReader(open(p)))
            st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
            if st[-1] < 0.99 * 5_000_000:
                continue
            score = float(ret[st > 4.5e6].mean())
            d = os.path.dirname(p)
            model = os.path.join(d, os.path.basename(d).split("__")[1] + ".cleanrl_model")
            if os.path.exists(model) and (best is None or score > best[0]):
                best = (score, model, re.search(r"__(\d+)$", d).group(1))
        if best is None:
            print(f"!! no finished R37 {src} model for {game}")
            continue
        print(f"{game:14s} {src:4s} seed {best[2]}  score {best[0]:7.2f}  {best[1]}")
        lines.append(f"{game} {src} {best[1]}")
open("r39_sources.txt", "w").write("\n".join(lines) + "\n")
print(f"\nWrote r39_sources.txt ({len(lines)} lines; R39 needs 10)")
