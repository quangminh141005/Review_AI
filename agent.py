from providers.base import BaseProvider
from tools.registry import ToolRegistry


"""
The similar thing of llm response of any provider is that 
there is a variable for function call or not (if i set automation
for function calling to false)
"""

class Agent:
    def __init__(self, provider: BaseProvider, tool_registry: ToolRegistry):
        self.provider = provider
        self.tool_registry = tool_registry

    def run(self, messages, system_prompt="", tools=None):
        # Loop through 10 times - maximum number of thoughts to reach final answers
        for _ in range(10):
            response = self.provider.generate(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Check if there is variable tool_call exist, if no return
            if not response.tool_calls:
                return response.text

            for call in response.tool_calls:
                result = self.tool_registry.execute(
                    call.name,
                    call.arguments,
                )

                # add model tools call value after execute
                messages.append({
                    "type": "function_call_output",
                    "call_id": "" # fill this later,
                    "output": json.dumps(result),
                })
