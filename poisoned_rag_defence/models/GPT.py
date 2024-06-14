from asyncio import sleep

from openai import AsyncOpenAI, RateLimitError

from ..logger import logger
from .model import Model


class GPT(Model):
    def __init__(self, config: dict):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.is_async = True
        self.aclient = AsyncOpenAI()

    def query(self, msg: str) -> str:
        """Query the model.

        Args:
            msg (str): the message to pass to the model

        Returns:
            str: the LLM's response
        """
        raise NotImplementedError("Only async implemented")

    async def aquery(self, msg, backoff=10) -> str:
        """Query the model asynchronously.

        If the openai api returns an error indicating you have hit your limits,
        will retry after a backoff time.

        Args:
            msg (str): the message to pass to the model
            backoff (int): the seconds to wait before trying again if a rate
              limit is reached

        Returns:
            str: the LLM's response
        """
        while True:
            try:
                completion = await self.aclient.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": msg},
                    ],
                )
                response = completion.choices[0].message.content
                break

            except RateLimitError as e:
                logger.debug(f"Hit rate limit. Will try again in {backoff}s: {e}")
                response = ""
                await sleep(backoff)

        return response

    async def aquery_cot_with_reprompt(
        self, msg: str, llm_backoff: int
    ) -> dict[str, any]:
        """Run a CoT query, and reprompt the llm if it does not include the answer.

        Args:
            msg (str): the message to pass to the model
            llm_backoff (int): the seconds to wait before trying again if a rate
              limit is reached

        Returns:
            dict[str, any]: _description_
        """
        response = await self.aquery(msg, llm_backoff)

        ret = {"prompt": msg, "initial_response": response}
        ret_split = response.split("Answer:")
        if len(ret_split) == 1:
            # The LLM did not reply with an answer - reprompt it
            follow_up_prompt = msg + response + "\nAnswer:"
            ret["follow_up_prompt"] = follow_up_prompt
            output = await self.aquery(follow_up_prompt)

            ret["output"] = output
        elif len(ret_split) == 2:
            # Answer was in the correct format. Just return the result
            ret["output"] = ret_split[1]
        elif len(ret_split) != 2:
            logger.warning(f"Incorrect response format!!! Got {response}")
            ret["output"] = ""
        return ret
