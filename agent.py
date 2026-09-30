import os
import json

from decouple import config
from openai import OpenAI   


# ============================================================
# Configuration
# ============================================================

client = OpenAI(
    api_key=config("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

TARGET_FOLDER = "notes"

# Make sure the target folder exists
os.makedirs(TARGET_FOLDER, exist_ok=True)


# ============================================================
# Agent Tools
# ============================================================

def safe_path(filename):
    """
    Security check:
    Only allow files inside the TARGET_FOLDER.
    """

    if not filename:
        return None

    # Prevent absolute paths
    if os.path.isabs(filename):
        return None

    # Prevent directory traversal
    normalized = os.path.normpath(filename)

    if normalized.startswith("..") or normalized == "..":
        return None

    path = os.path.join(TARGET_FOLDER, normalized)

    # Final protection: make sure the resolved path
    # is still inside TARGET_FOLDER
    target_dir = os.path.abspath(TARGET_FOLDER)
    full_path = os.path.abspath(path)

    if not (
        full_path == target_dir
        or full_path.startswith(target_dir + os.sep)
    ):
        return None

    return full_path


def list_files():
    """
    List all files inside the notes folder.
    """

    try:
        files = os.listdir(TARGET_FOLDER)

        if not files:
            return "The notes folder is empty."

        return "\n".join(files)

    except Exception as e:
        return f"Error listing files: {e}"


def read_file(filename):
    """
    Read the contents of a file inside the notes folder.
    """

    path = safe_path(filename)

    if path is None:
        return "Only files inside the notes folder can be read."

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Limit very large files
        return content[:3000]

    except FileNotFoundError:
        return (
            f"File '{filename}' was not found. "
            "Use list_files to see the available files."
        )

    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filename, content):
    """
    Create or overwrite a file inside the notes folder.
    """

    path = safe_path(filename)

    if path is None:
        return "Only files inside the notes folder can be written."

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File '{filename}' was written successfully ({len(content)} characters)."

    except Exception as e:
        return f"Error writing file: {e}"


# ============================================================
# Tool Definitions for the LLM
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List all files available inside the notes folder. "
                "Use this when you need to know which files exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a specific file inside the notes folder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file, for example 'todo.txt'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file inside the notes folder "
                "with the provided content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The file name, for example 'report.md'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content that should be written to the file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
]


AVAILABLE_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
}


# ============================================================
# Agent System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a development AI agent that works only with files inside the notes folder.

The user will give you a goal.

You must complete the goal step by step using the available tools.

Rules:

1. Do not guess information that can be obtained from files.
2. Use tools when you need information or need to perform an action.
3. Work step by step:
   - Decide what information/action is needed.
   - Call the appropriate tool.
   - Inspect the tool result.
   - Decide the next action.
4. Do not perform unnecessary tool calls.
5. Never access files outside the notes folder.
6. When the goal has been completely achieved, stop using tools and give a short final response in English describing what you did.
7. If a file named report.md is being created, do not read report.md afterward unless the user explicitly asks you to.
"""


# ============================================================
# Agent Loop
# ============================================================

MAX_STEPS = 15

def run_agent(goal, max_steps= MAX_STEPS):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": goal,
        },
    ]

    for step in range(1, max_steps + 1):

        # ====================================================
        # THINK
        # Ask the LLM what to do next
        # ====================================================

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # ====================================================
        # FINISH
        # No tool call means the agent considers the task done
        # ====================================================

        if not msg.tool_calls:

            print("\nAgent Final Response:\n")
            print(msg.content)

            return

        # Add the assistant's tool-call message to conversation
        messages.append(msg)

        # ====================================================
        # ACT
        # Execute each requested tool
        # ====================================================

        for tool_call in msg.tool_calls:

            function_name = tool_call.function.name

            try:
                function_args = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError as e:

                result = f"Invalid tool arguments: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

                continue

            print(
                f"\n[Step {step}] "
                f"{function_name}({function_args})"
            )

            function = AVAILABLE_FUNCTIONS.get(function_name)

            if function is None:

                result = f"Unknown function: {function_name}"

            else:

                try:
                    # IMPORTANT:
                    # Pass JSON arguments as keyword arguments.
                    result = function(**function_args)

                except Exception as e:
                    result = f"Tool execution error: {e}"

            # =================================================
            # OBSERVE
            # Give the tool result back to the LLM
            # =================================================

            print("\nTool Result:")
            print(result[:1000])

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # ====================================================
        # LOOP
        # The LLM receives the tool result and decides next step
        # ====================================================

    print(
        f"\nAgent stopped after reaching the maximum "
        f"limit of {max_steps} steps."
    )


# ============================================================
# Run Agent
# ============================================================

if __name__ == "__main__":

    run_agent(
        """
        Read every file in the notes folder.

        Create a 2-3 line summary for each file.

        Then organize all summaries clearly and save them
        into a file named report.md.

        Do not read report.md after creating it.
        """
    )