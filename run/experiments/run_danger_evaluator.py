import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import tqdm.asyncio

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defense.danger_identification.danger_identification import (
    identify_dangerous_async,
)
from poisoned_rag_defense.models import create_model
from run.experiments.utils import (
    load_experiment_config,
    load_questions_context,
    save_df,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--combined",
        action="store_false",
        help="Whether to use the combined or seperate evaluator approach",
    )

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    combined = args.combined

    experiment_config = load_experiment_config(args.experiment_name)
    question_df, context_df = load_questions_context(args.experiment_name)

    contexts_expanded = pd.concat(
        {column: context_df[column] for column in context_df.columns}
    )
    contexts_expanded.index.names = ["Context type", "qid"]

    query_df = pd.DataFrame(contexts_expanded.rename("contexts"))
    query_df["question"] = question_df.loc[
        contexts_expanded.index.get_level_values("qid")
    ]["question"].to_list()

    llm = create_model(f"model_configs/{experiment_config['model']}_config.json")

    async def run_all_queries(query_df, llm, use_combined):
        sem = asyncio.Semaphore(10)
        print("Starting queries")

        async def run_query(row):
            async with sem:
                return await identify_dangerous_async(
                    row["contexts"], row["question"], llm, use_combined
                )

        return await tqdm.asyncio.tqdm_asyncio.gather(
            *[run_query(row) for _, row in query_df.iterrows()]
        )

    results = pd.DataFrame(
        asyncio.run(run_all_queries(query_df, llm, combined)),
        index=contexts_expanded.index,
    )

    save_df(
        results, args.experiment_name, f"danger_eval_p{'comb' if combined else 'sep'}.p"
    )


if __name__ == "__main__":
    main()
