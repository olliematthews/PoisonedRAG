from pathlib import Path
import pandas as pd
import json


def is_correct(row):
    return row["correct answer"].lower() in row["output"].lower()


def is_incorrect(row):
    return row["incorrect answer"].lower() in row["output"].lower()


cdir = Path(__file__).parent

with open(cdir / "experiment_2" / "results_0.json") as fd:
    data_0 = json.load(fd)
with open(cdir / "experiment_2" / "results_1.json") as fd:
    data_1 = json.load(fd)
with open(cdir / "experiment_2" / "results_2.json") as fd:
    data_2 = json.load(fd)


dataset_questions = pd.read_json(
    f"results/target_queries/{data_0['args']['eval_dataset']}.json"
)
dataset_questions.set_index("id", inplace=True)

results_original_prompt = pd.DataFrame(data_0["results"]).join(
    dataset_questions, on="question_id"
)
results_guarded_prompt = pd.DataFrame(data_1["results"]).join(
    dataset_questions, on="question_id"
)
results_gpt4_prompt = pd.DataFrame(data_2["results"]).join(
    dataset_questions, on="question_id"
)

results_original_prompt["correct"] = results_original_prompt.apply(is_correct, axis=1)
results_original_prompt["poisoned"] = results_original_prompt.apply(
    is_incorrect, axis=1
)

results_guarded_prompt["correct"] = results_guarded_prompt.apply(is_correct, axis=1)
results_guarded_prompt["poisoned"] = results_guarded_prompt.apply(is_incorrect, axis=1)

results_gpt4_prompt["correct"] = results_gpt4_prompt.apply(is_correct, axis=1)
results_gpt4_prompt["poisoned"] = results_gpt4_prompt.apply(is_incorrect, axis=1)


print()
