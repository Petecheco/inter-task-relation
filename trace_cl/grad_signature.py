from __future__ import annotations

from typing import Any

import torch

from .fisher import move_batch_to_device, tensor_batch_only
from .lora_utils import get_lora_named_parameters


def compute_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**batch, use_cache=False).loss


def flatten_lora_grads(
    model: torch.nn.Module,
    lora_params: list[tuple[str, torch.nn.Parameter]] | None = None,
) -> torch.Tensor:
    params = lora_params if lora_params is not None else get_lora_named_parameters(model)
    chunks: list[torch.Tensor] = []
    for _, parameter in params:
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(parameter.grad.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def compute_grad_signature(
    model: torch.nn.Module,
    batch: dict[str, Any],
    lora_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    *,
    device: torch.device | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    batch = move_batch_to_device(tensor_batch_only(batch), device)
    model.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch)
    loss.backward()
    signature = flatten_lora_grads(model, lora_params)
    model.zero_grad(set_to_none=True)
    if normalize:
        signature = signature / (signature.norm() + 1e-12)
    return signature


def compute_anchor_gradient(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lora_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    *,
    device: torch.device | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    params = lora_params if lora_params is not None else get_lora_named_parameters(model)
    total: torch.Tensor | None = None
    count = 0
    was_training = model.training
    model.eval()
    for batch in dataloader:
        signature = compute_grad_signature(
            model,
            batch,
            params,
            device=device,
            normalize=False,
        )
        total = signature if total is None else total + signature
        count += 1
    if was_training:
        model.train()
    if total is None:
        total = torch.empty(0, dtype=torch.float32)
    else:
        total = total / max(1, count)
    if normalize:
        total = total / (total.norm() + 1e-12)
    return total.cpu()
