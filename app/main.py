import argparse
import json
import os
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
    }
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=TOOLS,
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    message = chat.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if tool_calls and len(tool_calls) > 0:
        tool_call = tool_calls[0]
        function = getattr(tool_call, "function", None)

        if not function:
            raise RuntimeError("tool call missing function")

        function_name = getattr(function, "name", None)
        function_args = getattr(function, "arguments", "{}")

        if function_name != "Read":
            raise RuntimeError(f"unsupported tool: {function_name}")

        try:
            parsed_args = json.loads(function_args)
        except json.JSONDecodeError as e:
            raise RuntimeError("tool call arguments are not valid JSON") from e

        file_path = parsed_args.get("file_path")
        if not file_path or not isinstance(file_path, str):
            raise RuntimeError("Read requires a string file_path")

        with open(file_path, "r", encoding="utf-8") as f:
            file_contents = f.read()

        print(file_contents)
        return

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    print(message.content)


if __name__ == "__main__":
    main()
