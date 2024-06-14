import json
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path("./results/experiments")

### FOR SAVING / LOADING RESULTS


def load_experiment_config(experiment_name: str) -> dict[str, any]:
    results_dir = EXPERIMENT_DIR / experiment_name
    try:
        with open(results_dir / "config.json", "r") as fd:
            return json.load(fd)
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find config for experiment {experiment_name}. Have you run 'initialise_experiment_set.py' for that experiment?"
        ) from e


def load_questions_context(experiment_name: str) -> tuple[pd.DataFrame]:
    results_dir = EXPERIMENT_DIR / experiment_name
    try:
        question_df = pd.read_pickle(results_dir / "questions.p")
        context_df = pd.read_pickle(results_dir / "context.p")
        return question_df, context_df
    except FileNotFoundError as e:
        raise Exception(
            f"Unable to find question and context dfs for experiment {experiment_name}. Have you run 'run_retriever.py' for that experiment?"
        ) from e


def save_df(df: pd.DataFrame, experiment_name: str, save_name: str):
    results_dir = EXPERIMENT_DIR / experiment_name
    df.to_pickle(results_dir / save_name)
