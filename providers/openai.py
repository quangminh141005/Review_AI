import os
from openai import OpenAI
from dotenv import load_dotenv

from providers.base import BaseProvider

load_dotenv()


class OpenaiProvider(BaseProvider):
    def __init__(self, model):
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL")

    def generate(
            self, messages: list[dict[str, str]], 
            system_prompt = ""
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
        )

        return response.output_text