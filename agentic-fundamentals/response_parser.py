"""
response_parser.py — parse a raw Muse Glimmer assistant turn into its channels.

Muse Glimmer emits channel-scoped messages. This module turns that raw text into
(reasoning, tool_calls, final_content). It has no model and no GPU dependency,
so you can run it — and its checks — anywhere:

    python response_parser.py

Faithful to the vLLM `muse_glimmer_tool_parser` channel-scoping rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------------------
# Muse Glimmer emits channel-scoped messages:
#     ... to=self<|message|>  <reasoning>            <|eom|>
#     ... to=<tool>.<fn><|message|> <atem:function_calls>...</atem:function_calls> <|eom|>
#     ... to=user<|message|> <final answer>          <|eot|>
# An <atem:invoke> echoed inside a to=self or to=user span must NOT be parsed as a
# real tool call, so we strip those spans before scanning for invokes.
# --------------------------------------------------------------------------------------

_REASONING_RE = re.compile(r"to=self<\|message\|>(.*?)<\|eom\|>", re.DOTALL)
_CONTENT_RE = re.compile(r"to=user<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|$)", re.DOTALL)
_STRIP_REASONING_RE = re.compile(r"to=self<\|message\|>.*?<\|eom\|>", re.DOTALL)
_STRIP_CONTENT_RE = re.compile(r"to=user<\|message\|>.*?(?=<\|eot\|>|<\|eom\|>|$)", re.DOTALL)
_INVOKE_RE = re.compile(r"(<atem:invoke\b.*?</atem:invoke>)", re.DOTALL)
_NAME_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)


def _decode_value(raw: str) -> Any:
    """JSON-decode a parameter value when possible, else keep the raw string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ParsedTurn:
    reasoning: str | None
    tool_calls: list[ToolCall]
    final_content: str | None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class MuseGlimmerATEMParser:
    """Parse a raw Muse Glimmer assistant turn into reasoning / tool_calls / final_content."""

    @staticmethod
    def _strip_channels(text: str) -> str:
        text = _STRIP_REASONING_RE.sub("", text)
        text = _STRIP_CONTENT_RE.sub("", text)
        return text

    @classmethod
    def parse(cls, text: str) -> ParsedTurn:
        reasoning_m = _REASONING_RE.search(text)
        reasoning = reasoning_m.group(1).strip() if reasoning_m else None

        scoped = cls._strip_channels(text)
        tool_calls: list[ToolCall] = []
        for invoke in _INVOKE_RE.findall(scoped):
            name_m = _NAME_RE.search(invoke)
            if not name_m:
                continue
            args = {pm.group("key"): _decode_value(pm.group("value"))
                    for pm in _PARAM_RE.finditer(invoke)}
            tool_calls.append(ToolCall(name=name_m.group(1), arguments=args))

        content_m = _CONTENT_RE.search(text)
        if content_m:
            final_content = content_m.group(1).strip() or None
        elif "to=self<|message|>" not in text and "<|message|>" not in text:
            final_content = text.strip() or None
        else:
            final_content = None

        return ParsedTurn(reasoning=reasoning, tool_calls=tool_calls, final_content=final_content)


# --------------------------------------------------------------------------------------
# Offline checks — no model, no GPU. Run: python response_parser.py
# --------------------------------------------------------------------------------------
def _check(name: str, cond: bool) -> None:
    print(("PASS" if cond else "FAIL"), name)
    assert cond, name


def _main() -> None:
    # 1. A reasoning block + a single tool call (the exact shape Muse Glimmer produces).
    raw1 = (
        ' to=self<|message|>The user wants the weather in Paris. I will call get_weather.<|eom|>'
        '<|start|>assistant to=get_weather<|message|><atem:function_calls>\n'
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="units">celsius</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eot|>'
    )
    t1 = MuseGlimmerATEMParser.parse(raw1)
    _check("reasoning captured", bool(t1.reasoning) and "Paris" in (t1.reasoning or ""))
    _check("one tool call", len(t1.tool_calls) == 1)
    _check("tool name", t1.tool_calls[0].name == "get_weather")
    _check("tool args", t1.tool_calls[0].arguments == {"city": "Paris", "units": "celsius"})
    _check("no final content yet", t1.final_content is None)

    # 2. A final answer channel (to=user) with no tool call.
    raw2 = 'to=user<|message|>The weather in Paris is 14C and rainy.<|eot|>'
    t2 = MuseGlimmerATEMParser.parse(raw2)
    _check("no tool calls on final", not t2.has_tool_calls)
    _check("final content extracted", t2.final_content == "The weather in Paris is 14C and rainy.")

    # 3. Channel scoping: an <atem:invoke> echoed inside a to=user span is NOT a real call.
    raw3 = (
        'to=user<|message|>Here is how you would call it: '
        '<atem:invoke name="get_weather"><atem:parameter name="city">X</atem:parameter></atem:invoke>'
        '<|eot|>'
    )
    t3 = MuseGlimmerATEMParser.parse(raw3)
    _check("echoed invoke inside to=user is ignored", not t3.has_tool_calls)

    # 4. Plain content with no channel tags at all.
    raw4 = "Just a plain answer."
    t4 = MuseGlimmerATEMParser.parse(raw4)
    _check("plain content passthrough", t4.final_content == "Just a plain answer.")

    # 5. Parallel tool calls in one turn, separated by <|eom|>.
    raw5 = (
        ' to=self<|message|>I need both cities, so I will call the tool twice.<|eom|>'
        '<|start|>assistant to=get_weather<|message|><atem:function_calls>\n'
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eom|>'
        '<|start|>assistant to=get_weather<|message|><atem:function_calls>\n'
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eot|>'
    )
    t5 = MuseGlimmerATEMParser.parse(raw5)
    _check("two parallel tool calls survive <|eom|>", len(t5.tool_calls) == 2)
    _check("both cities parsed",
           [tc.arguments["city"] for tc in t5.tool_calls] == ["Paris", "Tokyo"])

    # 6. Structured (JSON) parameter values are decoded, scalars stay strings.
    raw6 = (
        '<|start|>assistant to=write_report<|message|><atem:function_calls>\n'
        '<atem:invoke name="write_report">\n'
        '<atem:parameter name="sections">["intro", "body"]</atem:parameter>\n'
        '<atem:parameter name="limit">3</atem:parameter>\n'
        '<atem:parameter name="title">Q3 Summary</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls><|eot|>'
    )
    t6 = MuseGlimmerATEMParser.parse(raw6)
    _check("json list decoded", t6.tool_calls[0].arguments["sections"] == ["intro", "body"])
    _check("json int decoded", t6.tool_calls[0].arguments["limit"] == 3)
    _check("plain string kept raw", t6.tool_calls[0].arguments["title"] == "Q3 Summary")

    print("\nAll parser checks passed.")


if __name__ == "__main__":
    _main()
