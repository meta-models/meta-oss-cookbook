#!/usr/bin/env python3
"""Deterministically evaluate Muse Glimmer checkpoints on fixed JSONL examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path


class EvaluationError(RuntimeError):
    """Raised when evaluation input or model configuration is invalid."""


_IMAGE_TYPES: frozenset[str] = frozenset({"image", "image_url", "input_image"})
_ARTICLES_RE: re.Pattern[str] = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply SQuAD-style normalization for exact match and token F1."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        " "
        if character in string.punctuation
        or unicodedata.category(character).startswith("P")
        else character
        for character in text
    )
    text = _ARTICLES_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def compute_metrics(prediction: str, reference: str) -> dict[str, float]:
    return {
        "normalized_exact_match": normalized_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
    }


def extract_user_response(generated: str) -> str:
    """Extract the first user-directed response from decoded ATEM output."""
    text = generated.strip()
    marker = "to=user"
    marker_index = text.find(marker)
    if marker_index >= 0:
        text = text[marker_index + len(marker) :]
    boundaries = [
        index
        for boundary in ("assistant to=", "user to=", "to=self")
        if (index := text.find(boundary)) >= 0
    ]
    if boundaries:
        text = text[: min(boundaries)]
    return text.strip()


def _is_image_part(part: object) -> bool:
    return isinstance(part, dict) and part.get("type") in _IMAGE_TYPES


def _image_positions(
    messages: Sequence[dict[str, object]],
) -> list[tuple[int, int]]:
    positions = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, part in enumerate(content):
            if _is_image_part(part):
                positions.append((message_index, content_index))
    return positions


def count_images(messages: Sequence[dict[str, object]]) -> int:
    return len(_image_positions(messages))


def remove_image(
    messages: Sequence[dict[str, object]], image_index: int
) -> list[dict[str, object]]:
    """Return a deep copy with one globally indexed image content part removed."""
    transformed = deepcopy(list(messages))
    positions = _image_positions(transformed)
    if image_index < 0 or image_index >= len(positions):
        raise EvaluationError(
            f"image index {image_index} is outside [0, {len(positions)})"
        )
    message_index, content_index = positions[image_index]
    content = transformed[message_index]["content"]
    if not isinstance(content, list):
        raise EvaluationError("image position no longer points to list content")
    del content[content_index]
    return transformed


def reorder_images(
    messages: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return a deep copy with image parts reversed while slots stay fixed."""
    transformed = deepcopy(list(messages))
    positions = _image_positions(transformed)
    image_parts = []
    for message_index, content_index in positions:
        content = transformed[message_index]["content"]
        if not isinstance(content, list):
            raise EvaluationError("image position no longer points to list content")
        image_parts.append(deepcopy(content[content_index]))
    for (message_index, content_index), image_part in zip(
        positions, reversed(image_parts), strict=True
    ):
        content = transformed[message_index]["content"]
        if not isinstance(content, list):
            raise EvaluationError("image position no longer points to list content")
        content[content_index] = image_part
    return transformed


def apply_counterfactual(
    messages: Sequence[dict[str, object]],
    mode: str,
    *,
    remove_image_index: int = 0,
) -> list[dict[str, object]]:
    if mode == "original":
        return deepcopy(list(messages))
    if mode == "remove":
        return remove_image(messages, remove_image_index)
    if mode == "reorder":
        return reorder_images(messages)
    raise EvaluationError(f"unsupported counterfactual mode: {mode}")


def stable_example_id(example: dict[str, object]) -> str:
    for key in ("id", "example_id"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
    canonical = json.dumps(
        example, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"generated-{digest}"


def select_fixed_examples(
    examples: Iterable[dict[str, object]],
    *,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int = 17,
) -> list[dict[str, object]]:
    """Select examples independently of JSONL order and attach stable IDs."""
    if limit is not None and limit < 1:
        raise EvaluationError("limit must be positive")

    by_id = {}
    for example in examples:
        example_copy = deepcopy(example)
        example_id = stable_example_id(example_copy)
        if example_id in by_id:
            raise EvaluationError(f"duplicate example id: {example_id}")
        example_copy["id"] = example_id
        by_id[example_id] = example_copy

    if ids is not None:
        if len(set(ids)) != len(ids):
            raise EvaluationError("requested IDs contain duplicates")
        missing = [example_id for example_id in ids if example_id not in by_id]
        if missing:
            raise EvaluationError(f"requested IDs were not found: {', '.join(missing)}")
        selected_ids = list(ids)
        if limit is not None:
            selected_ids = selected_ids[:limit]
    else:
        selected_ids = sorted(
            by_id,
            key=lambda example_id: (
                hashlib.sha256(f"{seed}\0{example_id}".encode()).hexdigest(),
                example_id,
            ),
        )
        if limit is not None:
            selected_ids = selected_ids[:limit]
    return [by_id[example_id] for example_id in selected_ids]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise EvaluationError(f"{path}:{line_number}: expected a JSON object")
            examples.append(value)
    if not examples:
        raise EvaluationError(f"no examples found in {path}")
    return examples


def _messages_from_example(example: dict[str, object]) -> list[dict[str, object]]:
    raw_messages = example.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise EvaluationError(f"example {stable_example_id(example)} has no messages")
    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise EvaluationError("every message must be a JSON object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise EvaluationError("every message must have a non-empty role")
        messages.append(deepcopy(message))
    return messages


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise EvaluationError("message content must be a string or content-part list")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise EvaluationError("message content parts must be JSON objects")
        if part.get("type") in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise EvaluationError(
                    "text content part is missing string field 'text'"
                )
            parts.append(text)
    return "\n".join(parts)


def split_prompt_reference(
    example: dict[str, object],
) -> tuple[list[dict[str, object]], str]:
    messages = _messages_from_example(example)
    if messages[-1].get("role") != "assistant":
        raise EvaluationError(
            f"example {stable_example_id(example)} must end with an assistant reference"
        )
    reference = _content_text(messages[-1].get("content"))
    return messages[:-1], reference


def _image_path(part: dict[str, object], dataset_directory: Path) -> Path:
    value = part.get("path") or part.get("image")
    if value is None and part.get("type") == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url")
        else:
            value = image_url
    if not isinstance(value, str) or not value:
        raise EvaluationError("image content part is missing a local path")
    value = value.removeprefix("file://")
    if "://" in value:
        raise EvaluationError("only local image paths are supported")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = dataset_directory / path
    if not path.is_file():
        raise EvaluationError(f"image does not exist: {path}")
    return path


def _collect_image_paths(
    messages: Sequence[dict[str, object]], dataset_directory: Path
) -> list[Path]:
    paths = []
    for message_index, content_index in _image_positions(messages):
        content = messages[message_index].get("content")
        if not isinstance(content, list):
            raise EvaluationError("image position no longer points to list content")
        part = content[content_index]
        if not isinstance(part, dict):
            raise EvaluationError("image position no longer points to an object")
        paths.append(_image_path(part, dataset_directory))
    return paths


def _selected_ids_sha256(examples: Sequence[dict[str, object]]) -> str:
    joined = "\n".join(stable_example_id(example) for example in examples)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_processor(
    args: argparse.Namespace,
    transformers: object,
    *,
    allow_tokenizer: bool = True,
) -> object:
    sources = []
    if args.processor:
        sources.append(args.processor)
    elif args.checkpoint:
        sources.append(args.checkpoint)
    else:
        sources.append(args.model)
    if args.model not in sources:
        sources.append(args.model)

    errors = []
    common_by_source = []
    for source in sources:
        revision = args.revision if source == args.model else None
        common = {
            "revision": revision,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        common_by_source.append((source, common))
        try:
            return transformers.AutoProcessor.from_pretrained(source, **common)
        except (OSError, RuntimeError, TypeError, ValueError) as processor_error:
            errors.append(f"AutoProcessor({source}): {processor_error}")

    if not allow_tokenizer:
        raise EvaluationError(
            "could not load an image-capable processor:\n" + "\n".join(errors)
        )
    for source, common in common_by_source:
        try:
            return transformers.AutoTokenizer.from_pretrained(source, **common)
        except (OSError, RuntimeError, TypeError, ValueError) as tokenizer_error:
            errors.append(f"AutoTokenizer({source}): {tokenizer_error}")
    raise EvaluationError("could not load processor:\n" + "\n".join(errors))


def _load_model(
    args: argparse.Namespace, transformers: object, torch: object
) -> object:
    source = args.checkpoint or args.model
    model_kwargs = {
        "revision": args.revision if source == args.model else None,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

    errors = []
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(source, **model_kwargs)
            break
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{class_name}: {error}")
    else:
        raise EvaluationError("could not select a model class:\n" + "\n".join(errors))

    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise EvaluationError("PEFT is required to load a LoRA adapter") from error
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    return model


def load_runtime(
    args: argparse.Namespace, *, allow_tokenizer: bool = True
) -> tuple[object, object, object]:
    try:
        import torch
        import transformers
    except ImportError as error:
        raise EvaluationError(
            "evaluation requires torch and transformers; install the recipe environment"
        ) from error
    processor = _load_processor(args, transformers, allow_tokenizer=allow_tokenizer)
    model = _load_model(args, transformers, torch)
    return model, processor, torch


def set_deterministic_state(torch: object, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _decode_target(processor: object) -> object:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer
    if hasattr(processor, "decode"):
        return processor
    raise EvaluationError("processor does not provide a decode method")


def _generation_eos_token_ids(tokenizer: object) -> list[int]:
    token_ids = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        token_ids.append(eos_token_id)
    if (
        hasattr(tokenizer, "get_vocab")
        and hasattr(tokenizer, "convert_tokens_to_ids")
        and "<|eot|>" in tokenizer.get_vocab()
    ):
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot|>")
        if isinstance(eot_token_id, int) and eot_token_id not in token_ids:
            token_ids.append(eot_token_id)
    return token_ids


def generate_one(
    model: object,
    processor: object,
    torch: object,
    messages: Sequence[dict[str, object]],
    *,
    dataset_directory: Path,
    max_new_tokens: int,
    reasoning_strength: str,
    current_date: str | None,
) -> tuple[str, str]:
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_strength": reasoning_strength,
    }
    if current_date is not None:
        template_kwargs["current_date"] = current_date
    try:
        prompt = processor.apply_chat_template(list(messages), **template_kwargs)
    except Exception as error:
        raise EvaluationError(f"processor chat template failed: {error}") from error
    if not isinstance(prompt, str):
        raise EvaluationError("processor chat template did not return text")

    image_paths = _collect_image_paths(messages, dataset_directory)
    images = []
    if image_paths:
        try:
            from PIL import Image
        except ImportError as error:
            raise EvaluationError("Pillow is required for image evaluation") from error
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

    processor_kwargs = {"text": prompt, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    model_device = next(model.parameters()).device
    inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise EvaluationError("processor output is missing input_ids")

    decoder = _decode_target(processor)
    tokenizer = getattr(processor, "tokenizer", processor)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    eos_token_ids = _generation_eos_token_ids(tokenizer)
    if eos_token_ids:
        generation_kwargs["eos_token_id"] = (
            eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
        )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    generated = decoder.decode(generated_ids, skip_special_tokens=True).strip()
    return prompt, generated


def _read_ids(path: Path | None, explicit_ids: Sequence[str]) -> list[str] | None:
    ids = list(explicit_ids)
    if path is not None:
        ids.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return ids or None


def _write_outputs(
    output_json: Path, output_csv: Path, payload: dict[str, object]
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    csv_temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = payload.get("examples")
    if not isinstance(results, list):
        raise EvaluationError("payload examples are missing")
    fieldnames = [
        "id",
        "prompt",
        "reference",
        "generated",
        "generated_raw",
        "normalized_exact_match",
        "token_f1",
        "counterfactual_mode",
        "counterfactual_applied",
    ]
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    json_temporary.replace(output_json)
    csv_temporary.replace(output_csv)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    examples = load_jsonl(args.examples)
    selected = select_fixed_examples(
        examples,
        ids=_read_ids(args.ids_file, args.id),
        limit=args.limit,
        seed=args.selection_seed,
    )
    requires_image_processor = any(
        count_images(split_prompt_reference(example)[0]) > 0 for example in selected
    )
    model, processor, torch = load_runtime(
        args, allow_tokenizer=not requires_image_processor
    )
    set_deterministic_state(torch, args.seed)

    results = []
    for example in selected:
        prompt_messages, reference = split_prompt_reference(example)
        image_count = count_images(prompt_messages)
        counterfactual_applied = args.counterfactual != "original" and image_count >= 2
        if counterfactual_applied:
            prompt_messages = apply_counterfactual(
                prompt_messages,
                args.counterfactual,
                remove_image_index=args.remove_image_index,
            )
        prompt, generated_raw = generate_one(
            model,
            processor,
            torch,
            prompt_messages,
            dataset_directory=args.examples.parent,
            max_new_tokens=args.max_new_tokens,
            reasoning_strength=args.reasoning_strength,
            current_date=args.current_date,
        )
        generated = extract_user_response(generated_raw)
        result = {
            "id": stable_example_id(example),
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
            "generated_raw": generated_raw,
            **compute_metrics(generated, reference),
            "counterfactual_mode": args.counterfactual,
            "counterfactual_applied": counterfactual_applied,
        }
        results.append(result)

    count = len(results)
    metrics = {
        "normalized_exact_match": sum(
            float(result["normalized_exact_match"]) for result in results
        )
        / count,
        "token_f1": sum(float(result["token_f1"]) for result in results) / count,
    }
    if args.adapter:
        model_kind = "qlora" if args.load_in_4bit else "lora"
    elif args.checkpoint:
        model_kind = "full_checkpoint"
    else:
        model_kind = "base"
    payload = {
        "schema_version": 1,
        "label": args.label or model_kind,
        "model": args.model,
        "revision": args.revision,
        "model_kind": model_kind,
        "adapter": args.adapter,
        "checkpoint": args.checkpoint,
        "load_in_4bit": args.load_in_4bit,
        "selection_seed": args.selection_seed,
        "generation_seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_strength": args.reasoning_strength,
        "current_date": args.current_date,
        "counterfactual_mode": args.counterfactual,
        "selected_ids_sha256": _selected_ids_sha256(selected),
        "metrics": metrics,
        "examples": results,
    }
    _write_outputs(args.output_json, args.output_csv, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Base model ID or local path")
    parser.add_argument("--revision", help="Pinned base-model revision")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--adapter", help="LoRA or QLoRA adapter directory")
    source.add_argument("--checkpoint", help="Full-model checkpoint directory")
    parser.add_argument("--processor", help="Optional processor ID or local path")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--label", help="Run label stored in evaluation JSON")
    parser.add_argument("--id", action="append", default=[], help="Fixed example ID")
    parser.add_argument("--ids-file", type=Path, help="One fixed example ID per line")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-strength",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument(
        "--current-date",
        help="Fixed ISO date passed to the chat template for reproducibility",
    )
    parser.add_argument(
        "--counterfactual",
        choices=("original", "remove", "reorder"),
        default="original",
    )
    parser.add_argument("--remove-image-index", type=int, default=0)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Reload the base in NF4 with bf16 compute and quant storage",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device map")
    parser.add_argument("--attn-implementation")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_csv is None:
        args.output_csv = args.output_json.with_suffix(".csv")
    try:
        payload = evaluate(args)
    except EvaluationError as error:
        parser.error(str(error))
    print(json.dumps(payload["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
