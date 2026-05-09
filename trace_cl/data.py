from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import load_json


IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    original_prompt_tokens: int
    kept_prompt_tokens: int
    truncated: bool
    overflow_after_truncation: bool


def load_trace_split(
    data_root: str | Path,
    task_name: str,
    split: str,
    *,
    prompt_field: str = "prompt",
    response_field: str = "answer",
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    path = Path(data_root) / task_name / f"{split}.json"
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(records):
        if prompt_field not in item or response_field not in item:
            raise KeyError(
                f"{path} item {idx} must contain `{prompt_field}` and `{response_field}`"
            )
        normalized.append(
            {
                "prompt": str(item[prompt_field]),
                "answer": str(item[response_field]),
            }
        )
        if max_samples is not None and len(normalized) >= max_samples:
            break
    return normalized


def render_generation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    kwargs = dict(chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def answer_suffix_ids(tokenizer: Any, answer: str) -> list[int]:
    eos_token = tokenizer.eos_token or ""
    suffix = f"{answer}{eos_token}\n"
    return encode_text(tokenizer, suffix)


def truncate_token_ids(
    token_ids: list[int],
    budget: int,
    strategy: str,
) -> tuple[list[int], bool]:
    if budget < 0:
        budget = 0
    if len(token_ids) <= budget:
        return list(token_ids), False
    if budget == 0:
        return [], True
    if strategy == "head":
        return token_ids[:budget], True
    if strategy == "head_tail":
        head = budget // 2
        tail = budget - head
        if tail == 0:
            return token_ids[:head], True
        return token_ids[:head] + token_ids[-tail:], True
    return token_ids[-budget:], True


def encode_trace_example(
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
    truncation: str = "tail",
    chat_template_kwargs: dict[str, Any] | None = None,
    safety_margin: int = 8,
) -> EncodedExample:
    prompt_content_ids = encode_text(tokenizer, prompt)
    prompt_text = render_generation_prompt(
        tokenizer, prompt, chat_template_kwargs=chat_template_kwargs
    )
    prompt_ids = encode_text(tokenizer, prompt_text)
    suffix_ids = answer_suffix_ids(tokenizer, answer)
    input_ids = prompt_ids + suffix_ids

    truncated = False
    kept_prompt_tokens = len(prompt_content_ids)
    if len(input_ids) > max_length:
        empty_prompt_ids = encode_text(
            tokenizer,
            render_generation_prompt(
                tokenizer, "", chat_template_kwargs=chat_template_kwargs
            ),
        )
        budget = max_length - len(empty_prompt_ids) - len(suffix_ids) - safety_margin
        kept_ids, truncated = truncate_token_ids(prompt_content_ids, budget, truncation)

        while True:
            kept_prompt_tokens = len(kept_ids)
            truncated_prompt = decode_ids(tokenizer, kept_ids)
            prompt_ids = encode_text(
                tokenizer,
                render_generation_prompt(
                    tokenizer,
                    truncated_prompt,
                    chat_template_kwargs=chat_template_kwargs,
                ),
            )
            input_ids = prompt_ids + suffix_ids
            if len(input_ids) <= max_length or not kept_ids:
                break
            overflow = len(input_ids) - max_length
            budget = max(0, len(kept_ids) - overflow - safety_margin)
            kept_ids, _ = truncate_token_ids(prompt_content_ids, budget, truncation)

    labels = [IGNORE_INDEX] * len(prompt_ids) + suffix_ids
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        original_prompt_tokens=len(prompt_content_ids),
        kept_prompt_tokens=kept_prompt_tokens,
        truncated=truncated,
        overflow_after_truncation=len(input_ids) > max_length,
    )


class TraceSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: Any,
        *,
        max_length: int,
        truncation: str,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.features: list[dict[str, list[int]]] = []
        self.encoded_stats: list[EncodedExample] = []
        for record in records:
            encoded = encode_trace_example(
                tokenizer,
                record["prompt"],
                record["answer"],
                max_length=max_length,
                truncation=truncation,
                chat_template_kwargs=chat_template_kwargs,
            )
            self.features.append(
                {"input_ids": encoded.input_ids, "labels": encoded.labels}
            )
            self.encoded_stats.append(encoded)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]

    def token_stats(self) -> dict[str, float | int]:
        lengths = [len(feature["input_ids"]) for feature in self.features]
        truncated = [item for item in self.encoded_stats if item.truncated]
        overflow = [item for item in self.encoded_stats if item.overflow_after_truncation]
        kept = [item.kept_prompt_tokens for item in self.encoded_stats]
        original = [item.original_prompt_tokens for item in self.encoded_stats]
        return {
            "num_examples": len(self.features),
            "mean_length": mean(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "truncated_count": len(truncated),
            "truncation_rate": len(truncated) / len(self.features)
            if self.features
            else 0,
            "overflow_after_truncation_count": len(overflow),
            "mean_original_prompt_tokens": mean(original) if original else 0,
            "mean_kept_prompt_tokens": mean(kept) if kept else 0,
        }


class CausalLMPaddedCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        self.padding_side = getattr(tokenizer, "padding_side", "right")

    def __call__(
        self, features: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            labels = list(feature["labels"])
            pad_len = max_len - len(input_ids)
            if self.padding_side == "left":
                batch_input_ids.append([self.pad_token_id] * pad_len + input_ids)
                batch_attention_mask.append([0] * pad_len + [1] * len(input_ids))
                batch_labels.append([IGNORE_INDEX] * pad_len + labels)
            else:
                batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
                batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
                batch_labels.append(labels + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
