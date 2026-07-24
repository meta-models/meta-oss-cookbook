"""
agent_loop.py — a pure-Python agent loop for Muse Glimmer, using HuggingFace Transformers.

No vLLM, no server, no API keys. This is the reference implementation of the Muse Glimmer
agent loop: plan -> call tool -> execute -> feed result back -> self-correct,
until the model emits a final `to=user` answer.

It contains:
  * MuseGlimmerAgent   — loads the HF checkpoint and runs the tool-use loop.
  * Tool        — a single registered tool and its JSON schema.
  * ToolRegistry — a @tool decorator plus dispatch for defining local tools.

Turn parsing lives next door in `response_parser.py`.

Run `python run_agent.py` for a worked example.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from response_parser import MuseGlimmerATEMParser


# --------------------------------------------------------------------------------------
# Tool registry — define local Python functions as agent tools
# --------------------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def to_openai_schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, description: str, parameters: dict) -> Callable:
        """Decorator: register a function as a tool with an explicit JSON schema."""
        def deco(fn: Callable) -> Callable:
            self._tools[fn.__name__] = Tool(fn.__name__, description, parameters, fn)
            return fn
        return deco

    def register(self, name: str, description: str, parameters: dict, fn: Callable) -> None:
        self._tools[name] = Tool(name, description, parameters, fn)

    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        # Muse Glimmer may namespace as "tool.fn"; accept the bare function name too.
        key = name.split(".")[-1] if name not in self._tools else name
        if key not in self._tools:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = self._tools[key].fn(**arguments)
        except Exception as e:  # surface errors so the model can self-correct
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------
@dataclass
class MuseGlimmerAgent:
    model_path: str
    tools: ToolRegistry
    system_prompt: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    max_steps: int = 6
    device_map: str = "auto"
    verbose: bool = True

    tokenizer: Any = field(init=False, default=None)
    model: Any = field(init=False, default=None)
    _eos_ids: list[int] = field(init=False, default_factory=list)

    def load(self) -> "MuseGlimmerAgent":
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map=self.device_map)
        self.model.eval()
        # Correct Muse Glimmer stop tokens: <|eot|> and <|end_of_text|>. NEVER <|eom|>.
        self._eos_ids = [self.tokenizer.convert_tokens_to_ids("<|eot|>"),
                         self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        return self

    def _generate(self, messages: list[dict]) -> str:
        enc = self.tokenizer.apply_chat_template(
            messages, tools=self.tools.schemas(), add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(self.model.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, eos_token_id=self._eos_ids)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        gen = out[0][enc["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=False)

    def run(self, user_message: str) -> str:
        """Run the full agent loop until a final answer. Returns the final content."""
        messages: list[dict] = []
        sys = self.system_prompt
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": user_message})

        for step in range(1, self.max_steps + 1):
            raw = self._generate(messages)
            turn = MuseGlimmerATEMParser.parse(raw)

            if self.verbose:
                print(f"\n=== step {step} ===")
                if turn.reasoning:
                    print(f"[reasoning] {turn.reasoning[:300]}")
                for tc in turn.tool_calls:
                    print(f"[tool call] {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})")

            if not turn.has_tool_calls:
                final = turn.final_content or ""
                if self.verbose:
                    print(f"[final] {final[:500]}")
                return final

            # Record the assistant's tool-call turn, then execute each tool and
            # feed the results back so the model can continue / self-correct.
            # NOTE: the Muse Glimmer chat template requires `arguments` to be a dict
            # (mapping) — a JSON string cannot be parsed in the HF jinja sandbox.
            tool_calls_payload = [
                {"id": f"call_{step}_{i}", "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for i, tc in enumerate(turn.tool_calls)
            ]
            assistant_msg: dict = {"role": "assistant", "content": None,
                                   "tool_calls": tool_calls_payload}
            if turn.reasoning:
                assistant_msg["reasoning_content"] = turn.reasoning
            messages.append(assistant_msg)

            for i, tc in enumerate(turn.tool_calls):
                result = self.tools.call(tc.name, tc.arguments)
                if self.verbose:
                    print(f"[tool result] {result[:300]}")
                messages.append({"role": "tool", "name": tc.name,
                                 "tool_call_id": f"call_{step}_{i}", "content": result})

        return "[agent] max_steps reached without a final answer"
