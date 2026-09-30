from openai import OpenAI
from decouple import config
from datetime import datetime
import json

client = OpenAI(
    api_key=config("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# -------------- Here two Tools - mainly python function ---------------


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str):
    allowed_chars = set("0123456789+-*/(). ")

    if not set(expression) <= allowed_chars:
        raise ValueError("Only numbers and arithmetic operators are allowed.")

    try:
        return eval(expression, {"__builtins__": None}, {})
    except Exception:
        raise ValueError("Invalid arithmetic expression.")


# ----------- Tool Schemas - Instructions/ID Cards for LLM ---------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type":"object", 
                          "properties":{}, 
                          "required":[]
                        }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": ("Calculate an arithmatic expression.No guess."
                           "Only numbers and arithmetic operators are allowed."),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": ("Arithmetic expression using numbers, "
                                        "+, -, *, /, parentheses and spaces.")
                    }
                },
                "required": ["expression"],
            }
        }
    }
]

# ----------------- Available Functions ------------------

AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculator": calculator 
}

# ----------------- Agent Conversation ------------------- 

messages = [
    {
        "role":"system",
        "content": "You are an Assistant.Give the correct answer by using tool if needed"
    },
    {
        "role":"user",
        "content": "If someone start a journy Right now.After 1.5 hours he reached a place called Btoanical Garden.what time is the sow on Her clock?"  
    }
]

# ----------------- First LLM Call ------------------- 

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=messages,
    tools=tools
)

msg = response.choices[0].message

# ----------------- Tool Calling -------------------

if msg.tool_calls:
    messages.append(msg)

    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        print(f"LLM request: {fn_name}({fn_args})")

        result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

        messages.append(
            {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content": str(result)
            }
        )

# -----------------  Second LLM Call -------------------

    final = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=messages,
        tools=tools
    )
    print("\nAssistant", final.choices[0].message.content)

else:
    print("General:", msg.content)