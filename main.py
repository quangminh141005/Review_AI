import os
from google import genai
from google.genai import types
from mistralai import Mistral
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



system_content = "You are a puppy, you can only say what a puppy can say."
messages = []


# since the default message is for mistral, change for gemini
def convert_to_gemini(messages):
    convert = []

    # pull and change the model name from assistant 2 model
    for message_content in messages:
        role = message_content["role"]
        
        if role == "assistant":
            role = "model"

        convert.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(text=message_content["content"])
                ]
            )
        ) 

    return convert

def generated_response(client, provider, model, messages):
    if provider == "gemini":
        messages = convert_to_gemini(messages)
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_content
            )
        )
        response_text = response.text

    if provider == "mistral":
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
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