from typing import Any

class BaseProvider:
    """Base class for other LLM's SDK to inherit"""
    def generate(self, 
        messages: list[dict[str, str]], 
        system_prompt: str = ""
    ) -> str:
        raise NotImplementedError

    def generate_with_tools(
        self, 
        prompt: str, 
        tools_schemas: list[dict[str, Any]], 
        system_prompt: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError