"""Upload the PC-side evidence (every earlier study folder: protocols, readouts, JSON/CSV results, raw arrays,
reports) to W&B as one versioned artifact, so the report's evidence index points at files that are stored in W&B.

Run on your PC (Windows), from anywhere, in the Python environment you used for publish_wandb.py:
    pip install wandb
    wandb login
    python wb_upload_evidence.py --root "C:\\Users\\Mubashir\\stream-rep-rl\\experiments"
Add --skip-raw to leave out large raw arrays (*.npz, *.tgz) if the upload is too big.
"""
import argparse, os
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"C:\Users\Mubashir\stream-rep-rl\experiments")
ap.add_argument("--entity", default="mubbashirahmed")
ap.add_argument("--project", default="qrc-et-algo3-minatar")
ap.add_argument("--skip-raw", action="store_true")
args = ap.parse_args()

run = wandb.init(entity=args.entity, project=args.project, job_type="evidence-upload",
                 name="EVIDENCE | PC experiments folder (all earlier studies)", config={"experiment": "EVIDENCE", "root": args.root})
art = wandb.Artifact("qrc-et-evidence-pc-experiments", type="dataset",
                     description="Every earlier QRC/ET study folder from the PC: protocols, readouts, results, audits, raw data")
n, size = 0, 0
for dirpath, _, files in os.walk(args.root):
    if any(part in (".git", "__pycache__", "wandb") for part in dirpath.split(os.sep)):
        continue
    for f in files:
        if args.skip_raw and f.endswith((".npz", ".tgz", ".tar.gz", ".pt", ".pkl")):
            continue
        path = os.path.join(dirpath, f)
        art.add_file(path, name=os.path.relpath(path, args.root).replace(os.sep, "/"))
        n += 1
        size += os.path.getsize(path)
print(f"adding {n} files, {size / 1e6:.1f} MB")
run.log_artifact(art)
run.finish()
print("Done: artifact qrc-et-evidence-pc-experiments")
