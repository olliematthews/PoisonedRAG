from openai import OpenAI
from .Model import Model
from pathlib import Path

key = (Path.home() / Path("pers", "pr-key.txt")).read_text()

class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=key)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response