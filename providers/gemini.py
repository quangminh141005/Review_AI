import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from providers.base import BaseProvider, ToolCall, LLMResponse
from tools.schemas.gemini_schemas import gemini_tools

load_dotenv()

"""
The same note as openai (in ./openai.py)
"""

class GeminiProvider(BaseProvider):
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL")
        self.client = genai.Client()


    def convert_to_gemini(self, messages):
        convert = []

        # pull and change the model name from assistant 2 model
        for message_content in messages:
            role = message_content["role"]
            
            if role == "assistant":
                role = "model"

            convert.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part(text=str(message_content["content"]))
                    ]
                )
            ) 

        return convert

    def generate(self, messages: list[dict[str, str]], system_prompt: str = "") -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.convert_to_gemini(messages),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        tool_call = []

        if response.function_calls:
            for item in response.fuction_calls:
                tool_calls.append(
                    ToolCall(
                        id=item.id,
                        name=item.name,
                        arguments=json.load
                    )
                )


        return response.text
    
