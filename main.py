import os
from google import genai
from google.genai import types
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

provider = os.getenv("PROVIDER")

if provider=="gemini":
    model = os.getenv("GEMINI_MODEL")
    client = genai.Client()
elif provider=="mistral":
    model = os.getenv("MISTRAL_MODEL")
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
else:
    raise ValueError(f"Unsupported model provider: {provider}")




messages = []


def generated_response(client, provider, model, messages):
    if provider == "gemini":
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction="You are a puppy, you can only say what a puppy can say."
            )
        )
        response_text = response.text

    if provider == "mistral":
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a puppy, you can only say what a puppy can say.",
                },
                *messages,
            ]
        )
        response_text = response.choices[0].message.content

    return response_text




def main():
    while True: 
        user_input = input("User: ").strip() # take the user query

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        response_text = generated_response(client, provider, model, messages)

        messages.append({
            "role": "assistant",
            "content": response_text
        })

        print("Bot: ", response_text)

if __name__ == "__main__":
    main()