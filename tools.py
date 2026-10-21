from google.genai import types

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

available_tools = {
    "add": add,
    "multiply": multiply,
}

add_schema = types.FunctionDeclaration(
    name="add",
    description="Add two number",
    parameters={
        "type":"object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number",
            },
            "b": {
                "type": "number",
                "description": "The second number",
            },
        },
        "required": ["a", "b"],
    },
)

multiply_schema = types.FunctionDeclaration(
    name="multiply",
    description="Multiply two number",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number",
            },
            "b": {
                "type": "number",
                "description": "The second number",
            },
        },
        "required": ["a", "b"],
    },
)
