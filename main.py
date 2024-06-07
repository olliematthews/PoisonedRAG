import argparse
import os
import json
import random

import tqdm.asyncio
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from src.prompts.prompts import wrap_prompt, get_prompts
import torch
from pathlib import Path
import pickle
import asyncio
from typing import Optional
import tqdm


CACHE_DIR = Path("./.cache")
CACHE_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="test")

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument(
        "--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--orig_beir_results",
        type=str,
        default=None,
        help="Eval results of eval_model on the original beir eval_dataset",
    )

    # LLM settings
    parser.add_argument("--model_config_path", default=None, type=str)
    parser.add_argument("--model_name", type=str, default="palm2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_truth", type=str, default="False")
    parser.add_argument("--gpu_id", type=int, default=0)

    # attack
    parser.add_argument("--attack_method", type=str, default="LM_targeted")
    parser.add_argument(
        "--adv_per_query",
        type=int,
        default=5,
        help="The number of adv texts for each target query.",
    )
    parser.add_argument(
        "--score_function", type=str, default="dot", choices=["dot", "cos_sim"]
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of target queries",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument(
        "--name", type=str, default=None, help="Name of log and result."
    )
    parser.add_argument(
        "--prompt", type=str, default="original", help="The prompt to use."
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    # torch.cuda.set_device(args.gpu_id)
    device = "cpu"
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f"model_configs/{args.model_name}_config.json"

    cache_file = CACHE_DIR / f"{args.eval_dataset}-{args.split}.p"
    if not cache_file.exists():
        # load target queries and answers
        if args.eval_dataset == "msmarco":
            corpus, queries, qrels = load_beir_datasets("msmarco", "train")
        else:
            corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        with open(cache_file, "wb") as fd:
            pickle.dump((corpus, queries, qrels), fd)
    else:
        with open(cache_file, "rb") as fd:
            corpus, queries, qrels = pickle.load(fd)

    dataset_questions = load_json(f"results/target_queries/{args.eval_dataset}.json")

    random.shuffle(dataset_questions)
    dataset_questions = dataset_questions[: args.M]

    # load BEIR top_k results
    if args.orig_beir_results is None:
        print(
            f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}"
        )
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == "test":
            args.orig_beir_results = (
                f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"
            )
        elif args.split == "dev":
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == "cos_sim":
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(
            args.orig_beir_results
        ), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_results from {args.orig_beir_results}.")
    with open(args.orig_beir_results, "r") as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print("Total samples:", len(results))

    # Load retrieval models
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)
    attacker = Attacker(
        args, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb
    )

    llm = create_model(args.model_config_path)

    target_queries = []
    for question_info in dataset_questions:
        top1_idx = list(results[question_info["id"]].keys())[0]
        top1_score = results[question_info["id"]][top1_idx]
        target_queries.append(
            {
                "query": question_info["question"],
                "top1_score": top1_score,
                "id": question_info["id"],
            }
        )

    prompts = get_prompts(args.prompt)
    adv_text_groups = attacker.get_attack(target_queries)
    adv_text_list = sum(adv_text_groups, [])  # convert 2D array to 1D array

    adv_input = tokenizer(
        adv_text_list, padding=True, truncation=True, return_tensors="pt"
    )
    adv_input = {key: value for key, value in adv_input.items()}
    with torch.no_grad():
        adv_embs = get_emb(c_model, adv_input)

    ret_sublist = []

    iter_results = []
    for question_info, adv_text_group in zip(dataset_questions, adv_text_groups):
        question = question_info["question"]
        qid = question_info["id"]

        gt_ids = list(qrels[qid].keys())

        # Pass in the query with no context to get an idea of what the model "knows"
        query_prompt = wrap_prompt(question, None, prompts)
        # iter_results.append(
        #     {
        #         "question_id": qid,
        #         "query_type": "No context",
        #         "input_prompt": query_prompt,
        #     }
        # )

        # Test it with context, but no poisoning
        topk_idx = list(results[question_info["id"]].keys())[: args.top_k]
        topk_results = [
            {
                "score": results[question_info["id"]][idx],
                "context": corpus[idx]["text"],
            }
            for idx in topk_idx
        ]

        topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]

        query_prompt = wrap_prompt(question, topk_contents, prompts)

        # iter_results.append(
        #     {
        #         "question_id": qid,
        #         "query_type": "Context no poisoning",
        #         "input_prompt": query_prompt,
        #     }
        # )

        # Test it with context, with poisoning
        query_input = tokenizer(
            question, padding=True, truncation=True, return_tensors="pt"
        )
        query_input = {key: value for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        for j in range(len(adv_text_list)):
            adv_emb = adv_embs[j, :].unsqueeze(0)
            # similarity
            if args.score_function == "dot":
                adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
            elif args.score_function == "cos_sim":
                adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()

            topk_results.append({"score": adv_sim, "context": adv_text_list[j]})

        topk_results = sorted(
            topk_results, key=lambda x: float(x["score"]), reverse=True
        )
        topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
        # tracking the num of adv_text in topk
        adv_text_set = set(adv_text_group)

        cnt_from_adv = sum([i in adv_text_set for i in topk_contents])
        ret_sublist.append(cnt_from_adv)
        query_prompt = wrap_prompt(question, topk_contents, prompts)

        injected_adv = [i for i in topk_contents if i in adv_text_set]
        iter_results.append(
            {
                "question_id": qid,
                "query_type": "Context with poisoning",
                "provided_context": topk_contents,
                "injected_adv": injected_adv,
                "input_prompt": query_prompt,
            }
        )

        # Test with the correct context as as well as poisoned answers
        gts = [corpus[id]["text"] for id in gt_ids]
        topk_contents = gts + topk_contents

        query_prompt = wrap_prompt(question, topk_contents, prompts)
        iter_results.append(
            {
                "question_id": qid,
                "query_type": "Context with gt and poisoning",
                "provided_context": topk_contents,
                "injected_adv": injected_adv,
                "input_prompt": query_prompt,
            }
        )

    if not llm.is_async:
        for iter_result in zip(iter_results):
            response = llm.query(iter_result["input_prompt"])
            iter_result["output"] = response
    else:

        async def run_all_queries(iter_results):
            sem = asyncio.Semaphore(10)
            count = 0

            print("Starting queries")

            # prog = tqdm.tqdm(total = len(iter_results))
            async def run_query(iter_result):
                nonlocal count
                async with sem:
                    count += 1
                    # tqdm.
                    response = await llm.aquery(iter_result["input_prompt"], 10)
                    iter_result["output"] = response

                    if (
                        args.prompt == "cot"
                        and iter_result["query_type"] != "No context"
                        and "Answer:" not in response
                    ):
                        iter_result["initial_output"] = response
                        follow_up_prompt = (
                            iter_result["input_prompt"] + response + "\nAnswer:"
                        )
                        iter_result["follow_up_prompt"] = follow_up_prompt
                        follow_up_response = await llm.aquery(follow_up_prompt)
                        iter_result["output"] = follow_up_response
                    else:
                        iter_result["output"] = response
                    # if not count % 10:
                    #     print(f"Progress: {count / total * 100}%")

            # tqdm.asyncio.gather
            await tqdm.asyncio.tqdm_asyncio.gather(
                *[run_query(iter_result) for iter_result in iter_results]
            )

        asyncio.run(run_all_queries(iter_results))

    data = {
        "prompts": prompts,
        "args": args.__dict__,
        "results": iter_results,
    }
    save_results(data, args.name)


if __name__ == "__main__":
    main()
