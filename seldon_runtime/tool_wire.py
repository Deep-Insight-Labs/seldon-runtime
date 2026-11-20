"""A single forced function tool is an output format, never an executable action."""

import json
import re

from jsonschema import Draft202012Validator


def output_tool(request):
    """Validate the bounded tool wire contract and return its function definition."""
    tools = request.get("tools")
    if tools is None:
        if request.get("tool_choice") is not None or request.get("parallel_tool_calls") is not None:
            raise ValueError("output_tool_required")
        return None
    if request.get("response_format") is not None:
        raise ValueError("conflicting_output_formats")
    if not isinstance(tools, list) or len(tools) != 1:
        raise ValueError("single_output_tool_required")
    tool = tools[0]
    if (
        not isinstance(tool, dict)
        or set(tool) != {"type", "function"}
        or tool["type"] != "function"
    ):
        raise ValueError("invalid_output_tool")
    function = tool["function"]
    if (
        not isinstance(function, dict)
        or set(function) - {"name", "description", "parameters", "strict"}
        or not isinstance(function.get("name"), str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", function["name"])
        or ("description" in function and not isinstance(function["description"], str))
        or ("strict" in function and type(function["strict"]) is not bool)
        or not isinstance(function.get("parameters"), dict)
        or function["parameters"].get("type") != "object"
    ):
        raise ValueError("invalid_output_tool")
    choice = request.get("tool_choice")
    named = {"type": "function", "function": {"name": function["name"]}}
    if choice != "required" and choice != named:
        raise ValueError("forced_output_tool_required")
    if "parallel_tool_calls" in request and request["parallel_tool_calls"] is not False:
        raise ValueError("parallel_output_tools_unsupported")
    return function


def request_schema(request):
    tool = output_tool(request)
    return (
        tool["parameters"]
        if tool
        else (request.get("response_format") or {}).get("json_schema", {}).get("schema")
    )


def decode_output(raw, request):
    """Keep legacy content decoding; tool results must match the registered name/schema."""
    from .contracts import canonical

    tool = output_tool(request)
    choice = raw["choices"][0]
    message = choice["message"]
    if tool is None:
        output = json.loads(message["content"])
    else:
        calls = message.get("tool_calls")
        if (
            len(raw["choices"]) != 1
            or choice.get("finish_reason") != "tool_calls"
            or message.get("content") not in (None, "")
            or not isinstance(calls, list)
            or len(calls) != 1
            or not isinstance(calls[0], dict)
            or calls[0].get("type") != "function"
            or not isinstance(calls[0].get("function"), dict)
            or calls[0]["function"].get("name") != tool["name"]
            or not isinstance(calls[0]["function"].get("arguments"), str)
        ):
            raise ValueError("invalid_provider_output")
        output = json.loads(calls[0]["function"]["arguments"])
        if not Draft202012Validator(tool["parameters"]).is_valid(output):
            raise ValueError("invalid_provider_output")
    canonical(output)
    return output


def completion_choice(request, output, *, call_id):
    """Wrap validated execution output in the original Function response format."""
    from .contracts import canonical

    tool = output_tool(request)
    content = canonical(output)
    if tool is None:
        return {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }
    if not Draft202012Validator(tool["parameters"]).is_valid(output):
        raise ValueError("output_schema")
    return {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": content},
                }
            ],
        },
    }
