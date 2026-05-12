from __future__ import annotations

from typing import Any

import torch


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def tensor_batch_only(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}


def compute_layerwise_fisher(
    model: torch.nn.Module,
    anchor_loader: torch.utils.data.DataLoader,
    layer_to_params: dict[int, list[tuple[str, torch.nn.Parameter]]],
    *,
    normalize_by_numel: bool = True,
    device: torch.device | None = None,
) -> dict[int, float]:
    if device is None:
        device = _model_device(model)

    fisher = {int(layer): 0.0 for layer in layer_to_params}
    counts = {
        int(layer): int(sum(parameter.numel() for _, parameter in params))
        for layer, params in layer_to_params.items()
    }
    num_batches = 0
    was_training = model.training
    model.eval()

    for batch in anchor_loader:
        batch = move_batch_to_device(tensor_batch_only(batch), device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss.backward()
        num_batches += 1
        for layer, params in layer_to_params.items():
            value = 0.0
            for _, parameter in params:
                if parameter.grad is None:
                    continue
                value += float(parameter.grad.detach().float().pow(2).sum().cpu())
            fisher[int(layer)] += value

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    if num_batches == 0:
        return fisher
    for layer in fisher:
        fisher[layer] /= num_batches
        if normalize_by_numel:
            fisher[layer] /= max(1, counts.get(layer, 0))
    return fisher


def select_top_layers(fisher: dict[int, float], top_m: int) -> list[int]:
    if top_m <= 0:
        return []
    ranked = sorted(fisher.items(), key=lambda item: (-float(item[1]), int(item[0])))
    return [int(layer) for layer, _ in ranked[: int(top_m)]]


def fisher_json_payload(
    task_id: int,
    fisher: dict[int, float],
    selected_layers: list[int],
) -> dict[str, Any]:
    return {
        "task_id": int(task_id),
        "fisher": {str(layer): float(score) for layer, score in sorted(fisher.items())},
        "selected_layers": [int(layer) for layer in selected_layers],
    }
