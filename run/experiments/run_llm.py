from pathlib import Path
import pandas as pd
from poisoned_rag_defense.prompts.prompts import wrap_prompt, get_prompts
import asyncio
from poisoned_rag_defense.models import create_model
import tqdm.asyncio
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Retriever and BEIR datasets
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args


def main():
    args = parse_args()

    EXPERIMENT_DIR = Path("./results/experiments")

    results_dir = EXPERIMENT_DIR / args.experiment_name

    try:
        with open(results_dir / "config.json", "r") as fd:
            experiment_config = json.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find config for experiment {args.experiment_name}. Have you run 'initialise_experiment.py' for that experiment?"
        ) from e

    try:
        questions_df = pd.read_pickle(results_dir / "questions.p")
        context_df = pd.read_pickle(results_dir / "context.p")
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find question and context dfs for experiment {args.experiment_name}. Have you run 'generate_contexts.py' for that experiment?"
        ) from e

    prompt_templates = {
        prompt_type: get_prompts(prompt_type)
        for prompt_type in ["original", "refined", "cot"]
    }

    def to_query_str(row, context_col, prompt_templates):
        if context_col is None:
            return wrap_prompt(row["question"], None, prompt_templates)
        else:
            return wrap_prompt(row["question"], row[context_col], prompt_templates)

    assert all(
        [
            c[0] in experiment_config["context_configs"]
            for c in experiment_config["experiments"]
        ]
    ), "Unexpected context config"
    all_queries = pd.concat(
        {
            (context, prompt_type): context_df.join(questions_df).apply(
                to_query_str, args=(context, prompt_templates[prompt_type]), axis=1
            )
            for context, prompt_type in experiment_config["experiments"]
        }
    )

    if experiment_config["do_no_context"]:
        no_context_queries = questions_df.apply(
            to_query_str, args=(None, prompt_templates["refined"]), axis=1
        )

        no_context_queries.index = pd.MultiIndex.from_tuples(
            [("No context", "refined", qid) for qid in no_context_queries.index],
        )

        all_queries = pd.concat([no_context_queries, all_queries])

    all_queries.index.names = ["Context type", "Prompt type", "qid"]
    llm = create_model(f"model_configs/{experiment_config['model']}_config.json")

    async def run_all_queries(iter_results):
        sem = asyncio.Semaphore(10)
        print("Starting queries")

        # prog = tqdm.tqdm(total = len(iter_results))
        async def run_query(prompt_type, query):
            ret = {}
            async with sem:
                # tqdm.
                response = await llm.aquery(query, 20)
                ret["output"] = response

                if prompt_type == "cot" and "Answer:" not in response:
                    ret["initial_output"] = response
                    follow_up_prompt = query + response + "\nAnswer:"
                    ret["follow_up_prompt"] = follow_up_prompt
                    follow_up_response = await llm.aquery(follow_up_prompt)
                    ret["output"] = follow_up_response
                else:
                    ret["output"] = response
            return ret

        # tqdm.asyncio.gather
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

    results.to_pickle(results_dir / "llm_outputs.p")


if __name__ == "__main__":
    main()
