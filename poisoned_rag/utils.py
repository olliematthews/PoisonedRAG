import sys, os
from .contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
from collections import defaultdict
import random
import torch
from transformers import AutoTokenizer
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .models import GPT

base_experiment_dir = Path("results", "experiments")
base_experiment_dir.mkdir(exist_ok=True)

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
}


def contriever_get_emb(model, input):
    return model(**input)


def dpr_get_emb(model, input):
    return model(**input).pooler_output


def ance_get_emb(model, input):
    input.pop("token_type_ids", None)
    return model(input)["sentence_embedding"]


def load_models(model_code):
    assert (
        model_code in model_code_to_qmodel_name
        and model_code in model_code_to_cmodel_name
    ), f"Model code {model_code} not supported!"
    if "contriever" in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert (
            model_code_to_cmodel_name[model_code]
            == model_code_to_qmodel_name[model_code]
        )
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif "ance" in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert (
            model_code_to_cmodel_name[model_code]
            == model_code_to_qmodel_name[model_code]
        )
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError

    return model, c_model, tokenizer, get_emb


def load_beir_datasets(dataset_name, split):
    assert dataset_name in ["nq", "msmarco", "hotpotqa"]
    if dataset_name == "msmarco":
        split = "train"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name
    )
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if "-train" in data_path:
        split = "train"
    corpus, queries, qrels = data.load(split=split)

    return corpus, queries, qrels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_next_name(prefix, dir):
    past_prefix_matches = [
        d.stem[len(prefix) :] for d in dir.iterdir() if d.stem.startswith(prefix)
    ]
    top_index = 0
    for d in past_prefix_matches:
        try:
            index = int(d)
        except ValueError:
            continue
        if index >= top_index:
            top_index = index + 1
    return f"{prefix}{top_index}"


def save_results(results, experiment_name=None):
    if experiment_name is None:
        experiment_name = get_next_name("experiment_", base_experiment_dir)
    save_dir = base_experiment_dir / experiment_name
    save_dir.mkdir(exist_ok=True)
    results_file = get_next_name("results_", save_dir) + ".json"
    with open(save_dir / results_file, "w") as f:
        json.dump(results, f)


def load_results(file_name):
    with open(os.path.join("results", file_name)) as file:
        results = json.load(file)
    return results


def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f)


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_str(s):
    try:
        s = str(s)
    except:
        print("Error: the output cannot be converted to a string")
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.

    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.

    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(
        2 * precision * recall, precision + recall, where=(precision + recall) != 0
    )

    return f1_scores


async def run_cot_query_with_reprompt(
    prompt: str, llm: GPT, llm_backoff: int
) -> dict[str, any]:
    """Run a CoT query, and reprompt the llm if it does not include the answer.

    Args:
        prompt (str): _description_
        llm (GPT): _description_
        llm_backoff (int): _description_

    Returns:
        dict[str, any]: _description_
    """
    response = await llm.aquery(prompt, llm_backoff)

    ret = {"prompt": prompt, "initial_response": response}
    ret_split = response.split("Answer:")
    if len(ret_split) == 1:
        # The LLM did not reply with an answer - reprompt it
        follow_up_prompt = prompt + response + "\nAnswer:"
        ret["follow_up_prompt"] = follow_up_prompt
        output = await llm.aquery(follow_up_prompt)

        ret["output"] = output
    elif len(ret_split) == 2:
        # Answer was in the correct format. Just return the result
        ret["output"] = ret_split[1]
    elif len(ret_split) != 2:
        print(f"Incorrect response format!!! Got {response}")
        ret["output"] = ""
    return ret
