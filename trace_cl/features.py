from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .config import chat_template_kwargs
from .data import decode_ids, encode_text, render_generation_prompt, truncate_token_ids


def _example_prompt(example: Any) -> str:
    if isinstance(example, dict):
        return str(example.get("prompt", example.get("input", "")))
    return str(getattr(example, "prompt"))


def _task_max_length(config: dict[str, Any], task: dict[str, Any] | None) -> int:
    anchor_config = config.get("anchor", {})
    if anchor_config.get("max_length") is not None:
        return int(anchor_config["max_length"])
    data_config = config.get("data", {})
    if task is not None:
        return int(task.get("max_length", data_config.get("default_max_length", 2048)))
    return int(data_config.get("default_max_length", 2048))


def _task_truncation(task: dict[str, Any] | None) -> str:
    return str((task or {}).get("truncation", "tail"))


def render_feature_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    max_length: int,
    truncation: str,
    template_kwargs: dict[str, Any],
) -> str:
    prompt_content_ids = encode_text(tokenizer, prompt)
    rendered = render_generation_prompt(
        tokenizer,
        prompt,
        chat_template_kwargs=template_kwargs,
    )
    rendered_ids = encode_text(tokenizer, rendered)
    if len(rendered_ids) <= max_length:
        return rendered

    empty_prompt_ids = encode_text(
        tokenizer,
        render_generation_prompt(tokenizer, "", chat_template_kwargs=template_kwargs),
    )
    budget = max(0, max_length - len(empty_prompt_ids) - 8)
    kept_ids, _ = truncate_token_ids(prompt_content_ids, budget, truncation)
    while True:
        truncated_prompt = decode_ids(tokenizer, kept_ids)
        rendered = render_generation_prompt(
            tokenizer,
            truncated_prompt,
            chat_template_kwargs=template_kwargs,
        )
        rendered_ids = encode_text(tokenizer, rendered)
        if len(rendered_ids) <= max_length or not kept_ids:
            return rendered
        overflow = len(rendered_ids) - max_length
        budget = max(0, len(kept_ids) - overflow - 8)
        kept_ids, _ = truncate_token_ids(prompt_content_ids, budget, truncation)


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    normalized = mode.lower()
    if normalized in {"mean", "mean_pool"}:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        return (hidden_states * mask).sum(dim=1) / denom
    if normalized == "last_token":
        last_indices = attention_mask.size(1) - 1 - attention_mask.flip(1).argmax(dim=1)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]
    raise ValueError(f"Unsupported feature_pooling: {mode}")


def resolve_hidden_layer(layer: int, num_hidden_states: int) -> int:
    index = layer if layer >= 0 else num_hidden_states + layer
    if index < 0 or index >= num_hidden_states:
        raise ValueError(
            f"Feature layer {layer} resolves to {index}, but model returned "
            f"{num_hidden_states} hidden-state tensors."
        )
    return index


def _maybe_disable_adapter(model: torch.nn.Module):
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        return nullcontext()
    try:
        return disable_adapter()
    except TypeError:
        return nullcontext()


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    config: dict[str, Any],
    *,
    task: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not examples:
        return torch.empty(0, 0)

    anchor_config = config.get("anchor", {})
    max_samples = anchor_config.get("max_samples_for_clustering")
    if max_samples is not None and int(max_samples) > 0:
        examples = examples[: int(max_samples)]

    batch_size = int(anchor_config.get("feature_batch_size", 8))
    feature_layer = int(anchor_config.get("feature_layer", 4))
    pooling = str(anchor_config.get("feature_pooling", "mean"))
    max_length = _task_max_length(config, task)
    truncation = _task_truncation(task)
    template_kwargs = chat_template_kwargs(config)

    if device is None:
        device = next(model.parameters()).device

    prompts = [
        render_feature_prompt(
            tokenizer,
            _example_prompt(example),
            max_length=max_length,
            truncation=truncation,
            template_kwargs=template_kwargs,
        )
        for example in examples
    ]

    was_training = model.training
    model.eval()
    chunks: list[torch.Tensor] = []
    with _maybe_disable_adapter(model):
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_index = resolve_hidden_layer(
                feature_layer,
                len(outputs.hidden_states),
            )
            pooled = pool_hidden_states(
                outputs.hidden_states[hidden_index],
                encoded["attention_mask"],
                pooling,
            )
            chunks.append(pooled.detach().float().cpu())

    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)
