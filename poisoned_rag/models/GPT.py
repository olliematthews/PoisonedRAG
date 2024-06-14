from openai import AsyncOpenAI, OpenAI
from .Model import Model
from pathlib import Path
from asyncio import sleep


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.is_async = True
        self.aclient = AsyncOpenAI()
        self.client = OpenAI()

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
            )
            response = completion.choices[0].message.content

        except Exception as e:
            print(e)
            response = ""

        return response

    async def aquery(self, msg, backoff=10):
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

            except Exception as e:
                print(e)
                response = ""
                await sleep(backoff)

        return response
