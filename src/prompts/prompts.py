from . import prompt_cot, prompt_original, prompt_refined


def get_prompts(prompt: str):
    match prompt:
        case "original":
            context_prompt = prompt_original.PROMPT_W_CONTEXT
            no_context_prompt = prompt_original.PROMPT_WO_CONTEXT
        case "refined":
            context_prompt = prompt_refined.PROMPT_W_CONTEXT
            no_context_prompt = prompt_refined.PROMPT_WO_CONTEXT
        case "cot":
            context_prompt = prompt_cot.PROMPT_W_CONTEXT
            no_context_prompt = prompt_cot.PROMPT_WO_CONTEXT
    return {"context": context_prompt, "no context": no_context_prompt}


def wrap_prompt(question, context, prompts) -> str:

    if context is None:
        return prompts["no context"].replace("[question]", question)
    if isinstance(context, list):
        context = "\n".join(context)

    return (
        prompts["context"].replace("[question]", question).replace("[context]", context)
    )
