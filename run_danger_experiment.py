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

experiment_name = "varying_n"

results_dir = EXPERIMENT_DIR / experiment_name

with open(results_dir / "config.json", "r") as fd:
    experiment_config = json.load(fd)

context_df = pd.read_pickle(results_dir / "context.p").iloc[:5]

all_queries = pd.concat({column: context_df[column] for column in context_df.columns})
all_queries.index.names = ["Context type", "qid"]

llm = create_model(f"model_configs/{experiment_config['model']}_config.json")


async def run_all_queries(all_queries, llm):
    sem = asyncio.Semaphore(10)
    print("Starting queries")

    # prog = tqdm.tqdm(total = len(iter_results))
    async def run_query(query):
        async with sem:
            return await identify_dangerous_async(query, llm)

    # tqdm.asyncio.gather
    return await tqdm.asyncio.tqdm_asyncio.gather(
        *[run_query(query) for query in all_queries]
    )


results = pd.DataFrame(
    asyncio.run(run_all_queries(all_queries, llm)), index=all_queries.index
)

results.to_pickle(results_dir / "danger_results.p")
