import pandas as pd
import wandb

wandb.init(
    project="qrc-eligibility-traces",
    entity="mubbashirahmed",
    name="MinAtar-5M-summary"
)

summary = pd.read_csv("summary_results.csv")

table = wandb.Table(dataframe=summary)

wandb.log({
    "results_table": table
})

wandb.finish()