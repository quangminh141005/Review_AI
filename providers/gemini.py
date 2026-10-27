from google import genai
from google.genai import types

from providers.base import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client()


    def generate(self, prompt: str, system_prompt: str = "") -> str:
        response = self.client.model.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        return response.text
    
