from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]

@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None

@dataclass
class Messages:
    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


class BaseProvider(ABC):
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

    def _convert_messages(
        self,
        messages: list[Messages],
    ) -> Any:
        return NotImplementedError("Message conversion must be implemented by the provider") 

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        tools: list[str, Any] | None = None,            
    ) -> LLMResponse: 
        pass


    