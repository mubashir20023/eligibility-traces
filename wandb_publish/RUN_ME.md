# Publish everything to W&B: step by step

Files in this folder:
- `wb_publish_runs.py`: every R27-R31 run → W&B (offline, then synced), plus tables, figures, code and results artifacts
- `wb_build_report.py`: builds the report (new, or appended to an existing one)
- `QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md`: your master index; the report includes every section of it
- `wb_upload_evidence.py`: run on your PC, uploads the PC `experiments` folder (all earlier studies)
- `tabular_checks/`: the local numpy checks, goes to `experiments/tabular_checks/` in the repo

## A. Cluster (SSH window), from /fred/oz511/muahmed/code/stream-rep-rl
1. Put the files in place (after copying the folder over with scp):
       cp wandb_publish/wb_publish_runs.py wandb_publish/wb_build_report.py wandb_publish/QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md .
       mkdir -p experiments && cp -r wandb_publish/tabular_checks experiments/
       sed -i 's/\r$//' wb_publish_runs.py wb_build_report.py experiments/tabular_checks/*.py
2. A separate Python environment with the current W&B (the conda env has an old W&B that predates forge.coreweave.com):
       module -q load mamba && conda activate streamrl
       python -m venv ~/wbenv
       ~/wbenv/bin/pip install -q wandb wandb-workspaces numpy scipy matplotlib
3. Log in (API key from your W&B user settings → API keys):
       ~/wbenv/bin/wandb login
4. Check what will be published, then write all runs (no network needed; a few minutes):
       ~/wbenv/bin/python wb_publish_runs.py --list | head -3
       srun --time=01:00:00 --cpus-per-task=2 --mem=8G ~/wbenv/bin/python wb_publish_runs.py
5. Upload (on the login node, needs internet):
       cd wandb_upload && ~/wbenv/bin/wandb sync --sync-all && cd ..
6. Build the report (prints its URL):
       ~/wbenv/bin/python wb_build_report.py
   or append to an existing report instead:
       ~/wbenv/bin/python wb_build_report.py --append-url "<report URL>"
7. Save everything in git:
       git add wb_publish_runs.py wb_build_report.py QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md experiments/tabular_checks slurm/R31_theta_fix.sh summarize_r31.py
       git commit -m "W&B publishing scripts, tabular checks, R31" && git push mine cluster-r29

## B. PC (PowerShell): earlier studies
       pip install wandb
       wandb login
       python wb_upload_evidence.py --root "C:\Users\Mubashir\stream-rep-rl\experiments"
   (add --skip-raw if it is too large)
