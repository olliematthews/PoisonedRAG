from . import prompt_cot, prompt_original, prompt_refined

PROMPT_TEMPLATES = {
    "original": {
        "context": prompt_original.PROMPT_W_CONTEXT,
        "no context": prompt_original.PROMPT_WO_CONTEXT,
    },
    "refined": {
        "context": prompt_refined.PROMPT_W_CONTEXT,
        "no context": prompt_refined.PROMPT_WO_CONTEXT,
    },
    "cot": {
        "context": prompt_cot.PROMPT_W_CONTEXT,
        "no context": prompt_cot.PROMPT_WO_CONTEXT,
    },
}


def wrap_prompt(question, context, prompt_type) -> str:
    if context is None:
        return PROMPT_TEMPLATES[prompt_type]["no context"].replace(
            "[question]", question
        )
    if isinstance(context, list):
        context = "\n".join(context)

    return (
        PROMPT_TEMPLATES[prompt_type]["context"]
        .replace("[question]", question)
        .replace("[context]", context)
    )
