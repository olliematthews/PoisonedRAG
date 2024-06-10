import pickle
from pathlib import Path
import pandas as pd
import asyncio
from src.models import create_model
import tqdm
import json
from src.danger_identification import identify_dangerous_async

CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")
combined = True
experiment_name = "final_4"

results_dir = EXPERIMENT_DIR / experiment_name

with open(results_dir / "config.json", "r") as fd:
    experiment_config = json.load(fd)

context_df = pd.read_pickle(results_dir / "context.p")
question_df = pd.read_pickle(results_dir / "questions.p")

contexts_expanded = pd.concat(
    {column: context_df[column] for column in context_df.columns}
)
contexts_expanded.index.names = ["Context type", "qid"]

query_df = pd.DataFrame(contexts_expanded.rename("contexts"))
query_df["question"] = question_df.loc[contexts_expanded.index.get_level_values("qid")][
    "question"
].to_list()


llm = create_model(f"model_configs/{experiment_config['model']}_config.json")


async def run_all_queries(query_df, llm, use_combined):
    sem = asyncio.Semaphore(10)
    print("Starting queries")

    # prog = tqdm.tqdm(total = len(iter_results))
    async def run_query(row):
        async with sem:
            return await identify_dangerous_async(
                row["contexts"], row["question"], llm, use_combined
            )

    # tqdm.asyncio.gather
    return await tqdm.asyncio.tqdm_asyncio.gather(
        *[run_query(row) for _, row in query_df.iterrows()]
    )


results = pd.DataFrame(
    asyncio.run(run_all_queries(query_df, llm, combined)), index=contexts_expanded.index
)

results.to_pickle(results_dir / f"danger_eval_p{'comb' if combined else 'sep'}.p")
