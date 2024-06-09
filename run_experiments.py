import pickle
from pathlib import Path
import pandas as pd
from src.prompts.prompts import wrap_prompt, get_prompts
import asyncio
from src.models import create_model
import tqdm

CACHE_DIR = Path("./.cache")
EXPERIMENT_DIR = Path("./results/experiments")

experiment_name = "gpt3.5_final"
experiment_config = {
    "context_configs": {
        "Context no poisoning": (5, 5, False, False),
        "Context with poisoning": (5, 5, True, False),
        "Context with poisoning and gt": (5, 5, True, True),
    },
    "model": "gpt3.5",
    "prompt_types": ["original", "refined", "cot"],
}

results_dir = EXPERIMENT_DIR / experiment_name
results_dir.mkdir(exist_ok=True, parents=True)

with open(CACHE_DIR / "RETRIEVAL_DUMP-nq-test.p", "rb") as fd:
    test_cases, corpus = pickle.load(fd)


def get_context(row, n_contexts, n_adv, poison, input_gt=False):
    if n_contexts is None:
        return None
    contexts = []

    ds_contexts = sorted(
        row["dataset_contexts"].items(), key=lambda x: x[1], reverse=True
    )
    contexts = [
        {"context": corpus[key]["text"], "score": score}
        for key, score in ds_contexts[:n_contexts]
    ]

    if poison:
        contexts += row["attack_contexts"][:n_adv]
        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)[:n_contexts]

    if input_gt:
        contexts = [
            {"context": corpus[key]["text"], "score": None} for key in row["gt_ids"]
        ] + contexts
        contexts = contexts[:n_contexts]

    return [c["context"] for c in contexts]


questions_df = pd.DataFrame(test_cases)
questions_df.set_index("qid", inplace=True)

questions_df.to_pickle(results_dir / "questions.p")

context_df = pd.concat(
    {
        col: questions_df.apply(get_context, args=args_, axis=1)
        for col, args_ in experiment_config["context_configs"].items()
    },
    axis=1,
)

context_df.to_pickle(results_dir / "context_df.p")

prompt_templates = {
    prompt_type: get_prompts(prompt_type)
    for prompt_type in experiment_config["prompt_types"]
}


def to_query_str(row, context_col, prompt_templates):
    if context_col is None:
        return wrap_prompt(row["question"], None, prompt_templates)
    else:
        return wrap_prompt(row["question"], row[context_col], prompt_templates)


all_queries = pd.concat(
    {
        (context, prompt_type): context_df.join(questions_df).apply(
            to_query_str, args=(context, prompt_template), axis=1
        )
        for context in experiment_config["context_configs"]
        for prompt_type, prompt_template in prompt_templates.items()
    }
)


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
    count = 0

    print("Starting queries")

    # prog = tqdm.tqdm(total = len(iter_results))
    async def run_query(prompt_type, query):
        ret = {}
        nonlocal count
        async with sem:
            count += 1
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
