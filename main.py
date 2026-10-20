import os
from google import genai
from google.genai import types
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("MODEL")
client = genai.Client()
provider = os.getenv("PROVIDER")

messages = []

while True: 
    user_input = input("User: ").strip() # take the user query

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    messages.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
    )

    response = client.models.generate_content( # call gemeni api
        model=model,
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction="You are a puppy, you can only say what a puppy can say."
        )
    )

    messages.append(response.candidates[0].content)

    print("Bot: ", response.text)