from pathlib import Path
import pandas as pd
import asyncio
from poisoned_rag_defense.models import create_model
import tqdm.asyncio
import json
from poisoned_rag_defense.danger_identification import identify_dangerous_async
import argparse


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

    EXPERIMENT_DIR = Path("./results/experiments")

    results_dir = EXPERIMENT_DIR / args.experiment_name
    combined = args.combined

    try:
        with open(results_dir / "config.json", "r") as fd:
            experiment_config = json.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find config for experiment {args.experiment_name}. Have you run 'initialise_experiment.py' for that experiment?"
        ) from e

    try:
        question_df = pd.read_pickle(results_dir / "questions.p")
        context_df = pd.read_pickle(results_dir / "context.p")
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find question and context dfs for experiment {args.experiment_name}. Have you run 'generate_contexts.py' for that experiment?"
        ) from e

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
        asyncio.run(run_all_queries(query_df, llm, combined)),
        index=contexts_expanded.index,
    )

    results.to_pickle(results_dir / f"danger_eval_p{'comb' if combined else 'sep'}.p")


if __name__ == "__main__":
    main()
