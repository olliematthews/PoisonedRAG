import json

from ..models import GPT
from .combined_prompts import PROMPTS as COMBINED_PROMPTS
from .separate_prompts import PROMPTS as SEPARATE_PROMPTS

PROMPT_SETS = {"combined": COMBINED_PROMPTS, "separate": SEPARATE_PROMPTS}


async def identify_dangerous_async(
    contexts: list[str], question: str, llm: GPT, use_combined: bool = True
) -> dict[str, any]:
    """Identify is set of contexts are dangerous, given a question.

    Args:
        contexts (list[str]): the set of contexts
        question (str): the question
        llm (GPT): the LLM to use for evaluation
        use_combined (bool, optional): if true, use a single combined LLM prompt. Else individual LLM calls for each threat type. Defaults to True.

    Returns:
        dict[str, any]: information from the LLM call
    """
    if use_combined:
        prompt_set = PROMPT_SETS["combined"]
    else:
        prompt_set = PROMPT_SETS["separate"]

    dangerous = False
    rets = {}
    for prompt_type, prompt_template in prompt_set.items():
        prompt = prompt_template.replace(
            "[TEXTS]", json.dumps(contexts, indent=4)
        ).replace("[QUESTION]", question)

        ret = await llm.aquery_cot_with_reprompt(prompt, 10)

        # We just look for the word 'dangerous' in the response
        dangerous |= "dangerous" in ret["output"].lower()
        rets[prompt_type] = ret
    return {"dangerous": dangerous, "prompt_results": rets}
