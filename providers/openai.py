import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from providers.base import BaseProvider, ToolCall, LLMResponse, Messages
from tools.schemas.openai_schemas import openai_tools # add this in the .create arguments when testing

load_dotenv()

"""
The role of a provider (this is applicable for both gemini, openai-compatible and future provider as well)
-  a tool(interface) for the genetation class
- wrap around the return message with something that the agent can use for every single cases
- return a same datatype

A provider when generate a response:
1. take a message contain history and the lastest query, the system prompt and a toollist
2. Check if therer is the query need tool or not, put in a loop to check if the tools status still exist
3. execute the tool (function)
4. return the final answer

Uses the Chat Completions API: the schemas in tools/schemas/openai_schemas.py
are in Chat Completions format, and most "OpenAI-compatible" servers only
implement this endpoint.
"""

class OpenaiProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL")


    def _convert_messages(
        self,
        messages: list[Messages],
        system_prompt: str = "",
    ) -> list[dict]:
        converted = []

        # OpenAI takes the system prompt as the first message
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})

        for message in messages:

            # convert tool's results
            if message.role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                })
                continue

            item = {
                "role": message.role,
                "content": message.content or None,
            }

            # assistant tool calls (arguments must be a JSON string for OpenAI)
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]

            converted.append(item)

        return converted


    def generate(
        self,
        messages: list[Messages],
        system_prompt: str = "",
        tools=None,
    ) -> LLMResponse:

        kwargs = {}
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages, system_prompt),
            **kwargs,
        )

        message = response.choices[0].message

        tool_calls = []

        for item in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=item.id,
                    name=item.function.name,
                    arguments=json.loads(item.function.arguments or "{}"),
                )
            )

        return LLMResponse(
            text=message.content or None,
            tool_calls=tool_calls,
            raw_response=response,
        )
