from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import torch


LORA_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def is_lora_parameter(name: str) -> bool:
    return (
        "lora_A" in name
        or "lora_B" in name
        or "lora_embedding_A" in name
        or "lora_embedding_B" in name
    )


def get_lora_named_parameters(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
) -> list[tuple[str, torch.nn.Parameter]]:
    params: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if not is_lora_parameter(name):
            continue
        if trainable_only and not parameter.requires_grad:
            continue
        params.append((name, parameter))
    return params


def mark_only_lora_trainable(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_lora_parameter(name))


def parse_layer_id_from_param_name(name: str) -> int | None:
    match = LORA_LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def build_layer_to_lora_params(
    model: torch.nn.Module,
) -> dict[int, list[tuple[str, torch.nn.Parameter]]]:
    grouped: dict[int, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in get_lora_named_parameters(model):
        layer_id = parse_layer_id_from_param_name(name)
        if layer_id is not None:
            grouped[layer_id].append((name, parameter))
    return dict(sorted(grouped.items()))


def zero_grad_except_layers(
    model: torch.nn.Module,
    allowed_layers: set[int],
) -> None:
    for name, parameter in get_lora_named_parameters(model, trainable_only=False):
        if parameter.grad is None:
            continue
        layer_id = parse_layer_id_from_param_name(name)
        if layer_id not in allowed_layers:
            parameter.grad.zero_()


def init_lora_grad_accumulator(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for name, parameter in get_lora_named_parameters(model)
    }


def add_current_grads_to_accumulator(
    model: torch.nn.Module,
    accumulator: dict[str, torch.Tensor],
    *,
    allowed_layers: set[int] | None = None,
    scale: float = 1.0,
) -> None:
    for name, parameter in get_lora_named_parameters(model):
        if parameter.grad is None or name not in accumulator:
            continue
        if allowed_layers is not None:
            layer_id = parse_layer_id_from_param_name(name)
            if layer_id not in allowed_layers:
                continue
        accumulator[name].add_(
            parameter.grad.detach().to(accumulator[name].dtype),
            alpha=float(scale),
        )


def set_model_grads_from_accumulator(
    model: torch.nn.Module,
    accumulator: dict[str, torch.Tensor],
) -> None:
    for name, parameter in get_lora_named_parameters(model):
        grad = accumulator.get(name)
        if grad is None:
            parameter.grad = None
            continue
        parameter.grad = grad.detach().clone().to(
            device=parameter.device,
            dtype=parameter.dtype,
        )


def lora_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for _, parameter in get_lora_named_parameters(model)]


def layer_ids_for_param_names(names: Iterable[str]) -> dict[str, int | None]:
    return {name: parse_layer_id_from_param_name(name) for name in names}
