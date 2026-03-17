import argparse
import json
import os
import subprocess
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    }
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=TOOLS,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []

        assistant_message = {
            "role": "assistant",
            "content": getattr(message, "content", None),
        }
        if len(tool_calls) > 0:
            assistant_message["tool_calls"] = []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                if not function:
                    raise RuntimeError("tool call missing function")
                assistant_message["tool_calls"].append(
                    {
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", "function"),
                        "function": {
                            "name": getattr(function, "name", None),
                            "arguments": getattr(function, "arguments", "{}"),
                        },
                    }
                )
        messages.append(assistant_message)

        if len(tool_calls) == 0:
            print(message.content)
            return

        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if not function:
                raise RuntimeError("tool call missing function")

            function_name = getattr(function, "name", None)
            function_args = getattr(function, "arguments", "{}")

            try:
                parsed_args = json.loads(function_args)
            except json.JSONDecodeError as e:
                raise RuntimeError("tool call arguments are not valid JSON") from e

            if function_name == "Read":
                file_path = parsed_args.get("file_path")
                if not file_path or not isinstance(file_path, str):
                    raise RuntimeError("Read requires a string file_path")

                with open(file_path, "r", encoding="utf-8") as f:
                    tool_result = f.read()
            elif function_name == "Write":
                file_path = parsed_args.get("file_path")
                content = parsed_args.get("content")
                if not file_path or not isinstance(file_path, str):
                    raise RuntimeError("Write requires a string file_path")
                if content is None or not isinstance(content, str):
                    raise RuntimeError("Write requires a string content")

                file_dir = os.path.dirname(file_path)
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                tool_result = f"Wrote file: {file_path}"
            elif function_name == "Bash":
                command = parsed_args.get("command")
                if not command or not isinstance(command, str):
                    raise RuntimeError("Bash requires a string command")

                completed = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                stdout_text = completed.stdout or ""
                stderr_text = completed.stderr or ""
                output = f"{stdout_text}{stderr_text}".strip()

                if completed.returncode != 0:
                    if output:
                        tool_result = (
                            f"Command failed with exit code {completed.returncode}:\n{output}"
                        )
                    else:
                        tool_result = f"Command failed with exit code {completed.returncode}"
                else:
                    tool_result = output
            else:
                raise RuntimeError(f"unsupported tool: {function_name}")

            tool_call_id = getattr(tool_call, "id", None)
            if not tool_call_id:
                raise RuntimeError("tool call missing id")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result,
                }
            )


if __name__ == "__main__":
    main()
