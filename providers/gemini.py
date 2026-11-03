import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

from providers.base import BaseProvider, ToolCall, LLMResponse
from tools.schemas.gemini_schemas import gemini_tools

load_dotenv()

"""
The same note as openai (in ./openai.py)
- provide gemini thought signature?
"""

class GeminiProvider(BaseProvider):
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL")
        self.client = genai.Client()


    def _convert_messages(self, messages):
        converted = [] 

        for message in messages:

            parts = [] # for store function call

            # Normal text message
            if message.content and message.role != "tool":
                parts.append(
                    types.Part.from_text(text=message.content)
                )

            # assistance tool calls
            for call in message.tool_calls:
                parts.append(
                    types.Part.from_function_call(
                        name=call.name,
                        args=call.arguments,
                    )
                )

            # convert tool's results
            if message.role == "tool":
                try:
                    result = json.loads(message.content)
                except (json.JSONDecodeError, TypeError):
                    result = {"result": message.content}

                if not isinstance(result, dict):
                    result = {"result": result}

                parts.append(
                    type.Part.from_function_response(
                        name=message.tool_name,
                        response=result,
                    )
                )

            if not parts:
                continue

            role = ("model" if message.role == "assistant" else "user")

            converted.append(
                type.Content(
                    role=role,
                    parts=parts,
                )
            )

            return converted

    def generate(self, messages: list[dict[str, str]], system_prompt: str = "") -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._convert_messages(messages),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        tool_calls = []

        if response.function_calls:
            for item in response.fuction_calls:
                tool_calls.append(
                    ToolCall(
                        id=item.id,
                        name=item.name,
                        arguments=item.args,
                    )
                )


        return LLMResponse(
            text=response.text or None,
            tool_calls=tool_calls,
            raw_response=response,
        )
    
