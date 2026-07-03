import os
import pandas as pd

LOG_ROOT = "logs/MinAtar"
OUT_DIR = "plot_data"

os.makedirs(OUT_DIR, exist_ok=True)

data = {}

for folder in os.listdir(LOG_ROOT):
    folder_path = os.path.join(LOG_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    try:
        env, algo, seed = folder.split("__")
    except ValueError:
        continue

    csv_file = os.path.join(folder_path, "training_data.csv")

    if not os.path.exists(csv_file):
        continue

    df = pd.read_csv(csv_file)

    if "global_step" not in df.columns or "avg_return" not in df.columns:
        continue

    key = (env, algo)

    if key not in data:
        data[key] = {}

    data[key][f"seed{seed}"] = df[["global_step", "avg_return"]].copy()

for (env, algo), seeds in data.items():

    longest_seed = max(
        seeds.values(),
        key=lambda x: len(x)
    )

    merged = pd.DataFrame()
    merged["step"] = longest_seed["global_step"]

    for seed_name, seed_df in sorted(seeds.items()):
        merged = merged.merge(
            seed_df.rename(
                columns={
                    "global_step": "step",
                    "avg_return": seed_name
                }
            ),
            on="step",
            how="left"
        )

    out_file = os.path.join(
        OUT_DIR,
        f"minatar_{algo}_{env}.csv"
    )

    merged.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")

print("\nDone.")