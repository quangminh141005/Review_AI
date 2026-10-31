from abc import ABC, abstractmethod
from dataclass import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: int | None
    name: str
    arguments: dict[str, Any]

@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None


class BaseProvider:
    """Base class for other LLM's SDK to inherit"""
    # def generate(self, 
    #     messages: list[dict[str, str]], 
    #     system_prompt: str = ""
    # ) -> str:
    #     raise NotImplementedError

    # def generate_with_tools(
    #     self, 
    #     prompt: str, 
    #     tools_schemas: list[dict[str, Any]], 
    #     system_prompt: str = "",
    # ) -> dict[str, Any]:
    #     raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        tools: list[str] | None = None,            
    ) -> LLMResponse: 
        pass


    