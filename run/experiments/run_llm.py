import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import tqdm.asyncio

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defense.models import create_model
from poisoned_rag_defense.prompts.prompts import PROMPT_TEMPLATES, wrap_prompt
from poisoned_rag_defense.utils import run_cot_query_with_reprompt
from run.experiments.utils import (
    load_experiment_config,
    load_questions_context,
    save_df,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    experiment_config = load_experiment_config(args.experiment_name)
    question_df, context_df = load_questions_context(args.experiment_name)

    def to_query_str(row, context_col, prompt_type):
        if context_col is None:
            return wrap_prompt(row["question"], None, prompt_type)
        else:
            return wrap_prompt(row["question"], row[context_col], prompt_type)

    assert all(
        [
            c[0] in experiment_config["retriever_configs"]
            for c in experiment_config["experiments"]
        ]
    ), "Unexpected context config"
    all_queries = pd.concat(
        {
            (context, prompt_type): context_df.join(question_df).apply(
                to_query_str, args=(context, prompt_type), axis=1
            )
            for context, prompt_type in experiment_config["experiments"]
        }
    )

    if experiment_config["do_no_context"]:
        no_context_queries = question_df.apply(
            to_query_str, args=(None, "refined"), axis=1
        )

        no_context_queries.index = pd.MultiIndex.from_tuples(
            [("No context", "refined", qid) for qid in no_context_queries.index],
        )

        all_queries = pd.concat([no_context_queries, all_queries])

    all_queries.index.names = ["Context type", "Prompt type", "qid"]
    llm = create_model(f"model_configs/{experiment_config['model']}_config.json")

    async def run_all_queries(iter_results):
        """Run all of the queries in parallel"""

        # Use a semaphore to limit concurrent queries
        sem = asyncio.Semaphore(10)
        print("Starting queries")

        async def run_query(prompt_type, query):
            ret = {}
            async with sem:
                return (
                    run_cot_query_with_reprompt(query, llm, 20)
                    if prompt_type == "cot"
                    else {"output": await llm.aquery(query, 20)}
                )

        return await tqdm.asyncio.tqdm_asyncio.gather(
            *[
                run_query(prompt_type, iter_result)
                for prompt_type, iter_result in zip(
                    iter_results.index.get_level_values("Prompt type"), iter_results
                )
            ]
        )

    results = pd.DataFrame(
        asyncio.run(run_all_queries(all_queries)), index=all_queries.index
    )

    save_df(results, args.experiment_name, "llm_outputs.p")


if __name__ == "__main__":
    main()
