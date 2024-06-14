import asyncio
import sys
from pathlib import Path

import pandas as pd
import tqdm.asyncio

main_dir_path = str(Path(__file__).parent.parent.parent)
if main_dir_path not in sys.path:
    sys.path.append(main_dir_path)

from poisoned_rag_defence.logger import logger
from poisoned_rag_defence.models import create_model
from poisoned_rag_defence.prompts.prompts import wrap_prompt
from run.experiments.experiment import Experiment
from run.experiments.utils import experiment_name_parse_args


def run_llm():
    """Runs the llm part of the pipeline.

    Must be run after the retriever.
    Uses the `experiments` part of the experiment config to run the LLM over each
    question for each combination of retriever and LLM prompt.

    Saves the results to an llm_outputs dataframe.
    """
    args = experiment_name_parse_args()

    experiment = Experiment(args.experiment_name)

    def to_query_str(row, context_col, prompt_type):
        if context_col is None:
            return wrap_prompt(row["question"], None, prompt_type)
        else:
            return wrap_prompt(row["question"], row[context_col], prompt_type)

    assert all(
        [
            c[0] in experiment.config["retriever_configs"]
            for c in experiment.config["experiments"]
        ]
    ), "Unexpected context config"
    all_queries = pd.concat(
        {
            (context, prompt_type): experiment.context_df.join(
                experiment.question_df
            ).apply(to_query_str, args=(context, prompt_type), axis=1)
            for context, prompt_type in experiment.config["experiments"]
        }
    )

    if experiment.config["do_no_context"]:
        no_context_queries = experiment.question_df.apply(
            to_query_str, args=(None, "refined"), axis=1
        )

        no_context_queries.index = pd.MultiIndex.from_tuples(
            [("No context", "refined", qid) for qid in no_context_queries.index],
        )

        all_queries = pd.concat([no_context_queries, all_queries])

    all_queries.index.names = ["Context type", "Prompt type", "qid"]

    llm = create_model(f"model_configs/{experiment.config['model']}_config.json")

    async def run_all_queries(iter_results):
        """Run all of the queries in parallel"""

        # Use a semaphore to limit concurrent queries
        sem = asyncio.Semaphore(10)
        logger.info("Starting queries")

        async def run_query(prompt_type, query):
            async with sem:
                return (
                    await llm.aquery_cot_with_reprompt(query, 20)
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

    experiment.save_df(results, "llm_outputs")


if __name__ == "__main__":
    run_llm()
