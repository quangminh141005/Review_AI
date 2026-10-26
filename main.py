import os
import time
import json
from google import genai
from datetime import datetime
from google.genai import types
from mistralai import Mistral
from dotenv import load_dotenv

from tools.functions import available_tools
from tools.gemini_schemas import gemini_tools
from tools.open_ai_schemas import openai_tools

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

def generate_response(client, provider, model, messages, system_content):

    step = 0
    trace_logs = []
    user_query = messages[-1]["content"]
    
    # GEMINI
    if provider == "gemini":

        messages = convert_to_gemini(messages)

        # thought - action - observation
        for _ in range(10):

            step += 1

            step_start_time = time.time()
            
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_content,
                    tools=[
                        types.Tool(
                            function_declarations=gemini_tools
                        ),
                    ],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )

            latency_ms = round((time.time() - step_start_time) * 1000, 2)

            # if does not request any more function, return the text message
            if not response.function_calls:

                # add trace for the final answer
                trace_logs.append({
                    "step": step,
                    "query": user_query,
                    "action_type": "FINAL_ANSWER",
                    "output": response.text,
                    "latency_ms": latency_ms
                })

                return response.text, step, trace_logs

            messages.append(response.candidates[0].content)


            # execute, save and give back to the llm by wrap in types.
            tool_results = []

            for function_calls in response.function_calls:

                tool_name = function_calls.name
                arguments = function_calls.args

                print(f"LLM use tool {tool_name}")


                tool_start = time.perf_counter()
                result = available_tools[tool_name](**arguments)
                tool_latency_ms = round((time.perf_counter() - tool_start) * 1000, 2)

                # Convert into a gemini function respond
                tool_results.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                )

                trace_logs.append({
                    "step": step,
                    "query": user_query,
                    "action_type": "TOOL_EXECUTION",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "observation": result,
                    "latency_ms": tool_latency_ms
                })

            messages.append( # append the function result to the message, if enough will return the message to the user, it not do return for another tool, this is reAct i think
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




if __name__ == "__main__":

    # define the system prompt and initialize message holder - a list
    messages = []
    system_content = """
    You are batman with access to tools.

    For each request:
    - Decide whether a tool is needed
    - Decide the approprivate tool and provide its required arguments
    - If the action depends on the results of the previous tool, wait for the result
    - After receiveing a tool result, check whether it answers the request.
    - If more information needed, call another tool.
    - Ask for clarification if the required information is missing.
    - Never invent tool results.
    - When you have enough information, give a final answer
    """

    # Create log code for log file
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/trace_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"

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

        response_text, number_of_thoughts, logs = generate_response(client, provider, model, messages, system_content)

        # save the log in each step
        with open(log_path, "a", encoding="utf-8") as file:
            for entry in logs:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        messages.append({
            "role": "assistant",
            "content": response_text
        })

        print(f"Bro has thinked {number_of_thoughts} times")
        print("Bot: ", response_text)

