import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI


# OPENAI CHAT COMPLETIONS SCHEMAS
openai_add_schema = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
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
        }
    }
}

openai_multiply_schema = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "multiply two number",
        "parameters": {
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
        }
    }
}

openai_get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time on the machine running the program",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

openai_read_file_schema = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the content of a file in the data directory",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The file name the user wants to read",
                },
            },
            "required": ["filename"],
        },
    },
}

openai_get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather data for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city the user wants to check the weather for",
                },
            },
            "required": ["city"],
        },
    },
}



openai_tools = [
    openai_add_schema,
    openai_get_current_time_schema,
    openai_get_weather_schema,
    openai_multiply_schema,
    openai_read_file_schema,
]

