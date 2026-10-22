import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

DATA_DIR = (Path(__file__).resolve().parent / "data")
DATA_DIR.mkdir(exist_ok=True)


def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def get_current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def read_file(filename: str):
    """Read a file ONLY in the data dir"""
    try:
        path = (DATA_DIR / filename).resolve()

        if not path.is_relative_to(DATA_DIR):
            return "Error: Can not access the file"

        if not path.is_file():
            return "Error: File not exist"

        return Path(path).read_text(encoding="utf-8")
    
    except(OSError, UnicodeError) as e:
        return f"Error reading file: {e}"


def get_weather(city: str) -> dict | str:
    """return weather data from openWeather API"""
    api_key = os.getenv("OPEN_WEATHER_API")

    if not api_key:
        return ("Error: the OpenWeather API doesn't work")

    try:
        # convert geo to cordinate
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={
                "q": city,
                "limit": 1,
                "appid": api_key,
            },
            timeout=10
        )

        geo_response.raise_for_status() # check if the http request have any error

        location = geo_response.json()

        if not location:
            return f"Error: city not founded: {city}"

        location =location[0]

        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": location["lat"],
                "lon": location["lon"],
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return {
            "city": location["name"],
            "country": location["country"],
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "humidity_percent": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed_mps": data["wind"]["speed"],
        }
    
    except (requests.RequestException, ValueError, IndexError, KeyError) as e:
        return f"Error: {str(e)}"


available_tools = {
    "add": add,
    "multiply": multiply,
    "get_current_time": get_current_time,
    "read_file": read_file,
    "get_weather": get_weather,
}


# TOOL SCHEMA
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

get_current_time_schema = types.FunctionDeclaration(
    name="get_current_time",
    description="Get the current time of the user",
)

read_file_schema = types.FunctionDeclaration(
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

get_weather_schema = types.FunctionDeclaration(
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
