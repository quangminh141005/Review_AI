import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google.genai import types



# GEMINI TOOLs SCHEMA
gemini_add_schema = types.FunctionDeclaration(
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

gemini_multiply_schema = types.FunctionDeclaration(
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

gemini_get_current_time_schema = types.FunctionDeclaration(
    name="get_current_time",
    description="Get the current time of the user",
)

gemini_read_file_schema = types.FunctionDeclaration(
    name="read_file",
    description="Read the content of a file",
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The file name the user want to read",
            },
        },
        "required": ["filename"],
    },
)

gemini_get_weather_schema = types.FunctionDeclaration(
    name="get_weather",
    description="Get the weather data from a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city the user want to check the weather data",
            },
        },
        "required": ["city"],
    },
)



# TOOLS LIST
gemini_tools = [
    gemini_add_schema,
    gemini_get_current_time_schema,
    gemini_get_weather_schema,
    gemini_multiply_schema,
    gemini_read_file_schema,
]