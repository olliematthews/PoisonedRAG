import json

from .GPT import GPT


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == "gpt":
        model = GPT(config)
    else:
        raise ValueError(f"ERROR: Unsupported provider {provider}")
    return model
