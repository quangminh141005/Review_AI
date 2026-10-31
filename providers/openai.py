import os
from openai import OpenAI
from dotenv import load_dotenv

from providers.base import BaseProvider, ToolCall, LLMResponse
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
"""

class OpenaiProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL")

    def generate(
        self, 
        messages: list[dict[str, str]], 
        system_prompt = "",
        tools=None,
    ) -> str:
        
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=messages,
            tools=tools or  [],
        )

        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.load(item.arguments),
                    )
                )

        return LLMResponse(
            text=response.output_text or None,
            tool_calls=tool_calls,
            raw_response=response
        )

