from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SFT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = SFT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_script("evaluate")
plot_results = _load_script("plot_results")


def test_normalize_text_is_case_punctuation_article_and_space_insensitive() -> None:
    assert evaluate.normalize_text("  The, QUICK\tfox!  ") == "quick fox"
    assert evaluate.normalize_text("A café") == "café"
    assert evaluate.normalize_text("Answer: R_1 = 10 Ω") == "answer r 1 10 ω"


def test_normalized_exact_match_and_token_f1() -> None:
    assert evaluate.normalized_exact_match("The answer is 42.", "answer is 42") == 1.0
    assert evaluate.normalized_exact_match("42", "43") == 0.0
    assert evaluate.token_f1("red red blue", "red blue blue") == 2.0 / 3.0
    assert evaluate.token_f1("", "") == 1.0
    assert evaluate.token_f1("nonempty", "") == 0.0
    assert evaluate.compute_metrics("A cat", "cat") == {
        "normalized_exact_match": 1.0,
        "token_f1": 1.0,
    }


def test_extract_user_response_removes_atem_routing_and_followup_turns() -> None:
    direct = "to=userFirst answer.assistant to=userRepeated answer."
    reasoned = "to=selfPrivate reasoning.assistant to=userFinal answer."
    assert evaluate.extract_user_response(direct) == "First answer."
    assert evaluate.extract_user_response(reasoned) == "Final answer."
    assert evaluate.extract_user_response("Plain answer.") == "Plain answer."


def test_generation_eos_ids_include_muse_end_of_turn() -> None:
    tokenizer = SimpleNamespace(
        eos_token_id=7,
        get_vocab=lambda: {"<|eot|>": 11},
        convert_tokens_to_ids=lambda token: 11,
    )
    assert evaluate._generation_eos_token_ids(tokenizer) == [7, 11]


def _multimodal_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "/images/first.png"},
                {"type": "text", "text": "Compare these."},
                {"type": "image", "path": "/images/second.png"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "/images/third.png"},
                {"type": "text", "text": "Then summarize."},
            ],
        },
    ]


def _image_paths(messages: list[dict[str, object]]) -> list[str]:
    paths = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                path = part.get("path")
                assert isinstance(path, str)
                paths.append(path)
    return paths


def test_remove_image_is_global_deterministic_and_non_mutating() -> None:
    messages = _multimodal_messages()
    transformed = evaluate.remove_image(messages, 1)
    assert _image_paths(transformed) == [
        "/images/first.png",
        "/images/third.png",
    ]
    assert _image_paths(messages) == [
        "/images/first.png",
        "/images/second.png",
        "/images/third.png",
    ]
    assert evaluate.count_images(transformed) == 2


def test_reorder_images_reverses_payloads_but_preserves_slots_and_text() -> None:
    messages = _multimodal_messages()
    transformed = evaluate.reorder_images(messages)
    assert _image_paths(transformed) == [
        "/images/third.png",
        "/images/second.png",
        "/images/first.png",
    ]
    assert transformed[0]["content"][1] == {
        "type": "text",
        "text": "Compare these.",
    }
    assert transformed[1]["content"][1] == {
        "type": "text",
        "text": "Then summarize.",
    }
    assert _image_paths(messages)[0] == "/images/first.png"


def test_select_fixed_examples_is_stable_across_input_order() -> None:
    examples = [
        {"id": "text-1", "messages": []},
        {"id": "image-1", "messages": []},
        {"id": "multi-1", "messages": []},
        {"id": "text-2", "messages": []},
    ]
    selected = evaluate.select_fixed_examples(examples, limit=3, seed=91)
    reversed_selected = evaluate.select_fixed_examples(
        reversed(examples), limit=3, seed=91
    )
    assert [row["id"] for row in selected] == [row["id"] for row in reversed_selected]
    requested = evaluate.select_fixed_examples(examples, ids=["multi-1", "text-1"])
    assert [row["id"] for row in requested] == ["multi-1", "text-1"]


def test_processor_prefers_base_image_processor_to_checkpoint_tokenizer() -> None:
    calls = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(source: str, **_kwargs: object) -> tuple[str, str]:
            calls.append(("processor", source))
            if source == "checkpoint":
                raise OSError("processor was not saved")
            return ("processor", source)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source: str, **_kwargs: object) -> tuple[str, str]:
            calls.append(("tokenizer", source))
            return ("tokenizer", source)

    args = SimpleNamespace(
        processor=None,
        checkpoint="checkpoint",
        model="base",
        revision="revision",
        trust_remote_code=True,
        local_files_only=True,
    )
    transformers = SimpleNamespace(
        AutoProcessor=AutoProcessor,
        AutoTokenizer=AutoTokenizer,
    )
    assert evaluate._load_processor(args, transformers) == ("processor", "base")
    assert calls == [("processor", "checkpoint"), ("processor", "base")]


def test_qlora_reload_uses_bfloat16_model_compute_and_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bfloat16 = object()
    captured = {}

    class BitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["quantization"] = kwargs

    class Model:
        def eval(self) -> None:
            captured["evaluated"] = True

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(source: str, **kwargs: object) -> Model:
            captured["source"] = source
            captured["model"] = kwargs
            return Model()

    class PeftModel:
        @staticmethod
        def from_pretrained(model: Model, adapter: str, *, is_trainable: bool) -> Model:
            captured["adapter"] = (adapter, is_trainable)
            return model

    peft = ModuleType("peft")
    peft.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    args = SimpleNamespace(
        model="base",
        checkpoint=None,
        adapter="adapter",
        revision="revision",
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        attn_implementation=None,
        load_in_4bit=True,
    )
    transformers = SimpleNamespace(
        AutoModelForImageTextToText=AutoModelForImageTextToText,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )
    torch = SimpleNamespace(bfloat16=bfloat16)

    evaluate._load_model(args, transformers, torch)

    assert captured["source"] == "base"
    assert captured["model"]["torch_dtype"] is bfloat16
    assert captured["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": bfloat16,
        "bnb_4bit_quant_storage": bfloat16,
    }
    assert captured["adapter"] == ("adapter", False)
    assert captured["evaluated"] is True


def test_deterministic_evaluation_sets_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    def no_examples(path: Path) -> list[dict[str, object]]:
        raise evaluate.EvaluationError("no examples")

    monkeypatch.setattr(evaluate, "load_jsonl", no_examples)
    with pytest.raises(evaluate.EvaluationError, match="no examples"):
        evaluate.evaluate(SimpleNamespace(examples=Path("missing.jsonl")))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_state_is_strict_and_disables_tf32() -> None:
    calls = []
    torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("manual_seed", seed)),
        use_deterministic_algorithms=lambda enabled: calls.append(
            ("deterministic", enabled)
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(
                allow_tf32=True,
                benchmark=True,
                deterministic=False,
            ),
        ),
    )

    evaluate.set_deterministic_state(torch, 31)

    assert calls == [
        ("manual_seed", 31),
        ("cuda_seed", 31),
        ("deterministic", True),
    ]
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_json_and_csv_outputs_include_required_fields(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "result.csv"
    result = {
        "id": "fixed-1",
        "prompt": "user prompt",
        "reference": "reference",
        "generated": "generated",
        "normalized_exact_match": 0.0,
        "token_f1": 0.5,
        "counterfactual_mode": "original",
        "counterfactual_applied": False,
    }
    evaluate._write_outputs(output_json, output_csv, {"examples": [result]})

    assert json.loads(output_json.read_text(encoding="utf-8"))["examples"] == [result]
    header = output_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:4] == ["id", "prompt", "reference", "generated"]


def test_plot_history_input_aggregation_handles_json_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "history.json"
    json_path.write_text(
        json.dumps(
            {
                "history": [
                    {
                        "_step": 2,
                        "train": {
                            "loss": 0.8,
                            "tokens_per_second": 120,
                        },
                        "eval": {"loss": 0.7},
                    },
                    {
                        "_step": 1,
                        "train": {
                            "loss": 1.0,
                            "tokens_per_second": 100,
                        },
                        "system": {"gpu.0.memoryAllocatedBytes": 24 * 1024**3},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "Step,train/loss,eval/loss,gpu_memory_gb,samples_per_second\n"
        "3,0.6,0.5,25,4.5\n",
        encoding="utf-8",
    )

    json_series = plot_results.aggregate_history(plot_results.load_history(json_path))
    csv_series = plot_results.aggregate_history(plot_results.load_history(csv_path))
    assert json_series["loss"] == [(1.0, 1.0), (2.0, 0.8)]
    assert json_series["validation_loss"] == [(2.0, 0.7)]
    assert json_series["memory"] == [(1.0, 24.0)]
    assert json_series["throughput"] == [(1.0, 100.0), (2.0, 120.0)]
    assert csv_series == {
        "loss": [(3.0, 0.6)],
        "validation_loss": [(3.0, 0.5)],
        "memory": [(3.0, 25.0)],
        "throughput": [(3.0, 4.5)],
    }


def test_plot_evaluation_aggregation_supports_summary_and_examples() -> None:
    metrics = plot_results.aggregate_evaluations(
        [
            (
                "tuned",
                {
                    "examples": [
                        {"normalized_exact_match": 1.0, "token_f1": 0.75},
                        {"normalized_exact_match": 0.0, "token_f1": 0.25},
                    ]
                },
            ),
            (
                "base",
                {
                    "metrics": {
                        "normalized_exact_match": 0.25,
                        "token_f1": 0.4,
                    }
                },
            ),
        ]
    )
    assert metrics == {
        "normalized_exact_match": {"base": 0.25, "tuned": 0.5},
        "token_f1": {"base": 0.4, "tuned": 0.5},
    }


def test_plot_evaluation_aggregation_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicate evaluation label: adapter"):
        plot_results.aggregate_evaluations(
            [
                ("adapter", {"metrics": {"token_f1": 0.1}}),
                ("adapter", {"metrics": {"token_f1": 0.9}}),
            ]
        )
