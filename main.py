import os
from google import genai
from google.genai import types
from mistralai import Mistral
from dotenv import load_dotenv

from tools import add_schema, multiply_schema, available_tools

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


# Message define
system_content = "You are gangster"
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
                    types.Part(text=str(message_content["content"]))
                ]
            )
        ) 

    return convert

def generated_response(client, provider, model, messages):
    
    # GEMINI
    if provider == "gemini":

        messages = convert_to_gemini(messages)


        for _ in range(10):
        
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_content,
                    tools=[
                        types.Tool(
                            function_declarations=[
                            add_schema,
                            multiply_schema,
                            ]
                        ),
                    ],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )

            # if does not request any more function, return
            if not response.function_calls:
                return response.text

            messages.append(response.candidates[0].content)


            # execute, save and give back to the llm by wrap in types.
            tool_results = []

            for function_calls in response.function_calls:
                tool_name = function_calls.name
                arguments = function_calls.args

                print(f"LLM use tool {tool_name}")

                result = available_tools[tool_name](**arguments)


                # Convert into a gemini function respond
                tool_results.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

            messages.append(
                types.Content(
                    role="user",
                    parts=tool_results
                )
            )

        raise RuntimeError("Maximum tool interations exceeded")

            

            


    # MISTRAL
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