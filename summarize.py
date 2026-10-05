import os
import glob
import pandas as pd

rows = []

for algo in ["qrc", "qrc-et"]:
    files = glob.glob(f"logs/MinAtar/*__{algo}__*/training_data.csv")

    for f in files:
        df = pd.read_csv(f)

        last = df.iloc[-1]

        run = os.path.basename(os.path.dirname(f))

        env = run.split("__")[0]
        seed = run.split("__")[-1]

        rows.append({
            "env": env,
            "algo": algo,
            "seed": seed,
            "avg_return": last["avg_return"]
        })

df = pd.DataFrame(rows)

summary = (
    df.groupby(["env", "algo"])["avg_return"]
      .agg(["mean", "std", "count"])
      .reset_index()
)

print(summary)

summary.to_csv("summary_results.csv", index=False)
