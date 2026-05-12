from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import CausalLMPaddedCollator, TraceSFTDataset
from .grad_signature import compute_grad_signature
from .lora_utils import get_lora_named_parameters


def cosine_similarity(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    if u.numel() == 0 or v.numel() == 0 or u.numel() != v.numel():
        return 0.0
    denom = float(u.norm().item() * v.norm().item())
    if denom <= eps:
        return 0.0
    return float(torch.dot(u.float(), v.float()).item() / (denom + eps))


def compute_raw_shared_score(
    g_z: torch.Tensor,
    task_id: int,
    anchor_grads: dict[int, torch.Tensor],
) -> float:
    if not anchor_grads:
        return 0.0

    alignments = {
        int(anchor_task): cosine_similarity(g_z, grad)
        for anchor_task, grad in anchor_grads.items()
    }
    if len(alignments) == 1:
        return float(alignments.get(int(task_id), next(iter(alignments.values()))))

    own_alignment = alignments.get(int(task_id), 0.0)
    other_alignments = [
        value for anchor_task, value in alignments.items() if anchor_task != int(task_id)
    ]
    cross_term = (
        sum(other_alignments) / len(other_alignments) if other_alignments else 0.0
    )
    return float(max(0.0, own_alignment) + cross_term)


def rank_normalize_scores(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []
    if len(raw_scores) == 1:
        return [1.0]

    indexed = sorted(enumerate(raw_scores), key=lambda item: (item[1], item[0]))
    normalized = [0.0] * len(raw_scores)
    denom = len(raw_scores) - 1
    start = 0
    while start < len(indexed):
        end = start + 1
        value = indexed[start][1]
        while end < len(indexed) and indexed[end][1] == value:
            end += 1
        percentile = ((start + end - 1) / 2.0) / denom
        for position in range(start, end):
            normalized[indexed[position][0]] = float(percentile)
        start = end
    return normalized


def split_from_score(score: float, share_ratio: float) -> str:
    return "share" if float(score) >= 1.0 - float(share_ratio) else "specific"


def _example_field(example: Any, field: str, default: Any = None) -> Any:
    if isinstance(example, dict):
        return example.get(field, default)
    return getattr(example, field, default)


def assign_scores_to_examples(
    examples: list[Any],
    raw_scores: list[float],
    *,
    share_ratio: float,
) -> list[Any]:
    scores = rank_normalize_scores(raw_scores)
    for example, raw, score in zip(examples, raw_scores, scores, strict=True):
        if isinstance(example, dict):
            example["raw_score"] = float(raw)
            example["score"] = float(score)
            example["split"] = split_from_score(score, share_ratio)
        else:
            example.raw_score = float(raw)
            example.score = float(score)
            example.split = split_from_score(score, share_ratio)
    return examples


def score_example_dataset(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    dataset: TraceSFTDataset,
    anchor_grads: dict[int, torch.Tensor],
    *,
    share_ratio: float,
    micro_batch_size: int = 1,
    device: torch.device | None = None,
) -> list[Any]:
    if not examples:
        return []
    if len(examples) != len(dataset):
        raise ValueError(
            f"Scoring examples/dataset length mismatch: {len(examples)} vs {len(dataset)}"
        )
    if device is None:
        device = next(model.parameters()).device

    params = get_lora_named_parameters(model)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(micro_batch_size)),
        shuffle=False,
        collate_fn=CausalLMPaddedCollator(tokenizer),
    )
    raw_scores: list[float] = []
    was_training = model.training
    model.eval()
    offset = 0
    for batch in loader:
        batch_size = int(batch["input_ids"].shape[0])
        if batch_size != 1:
            for item_index in range(batch_size):
                item_batch = {
                    key: value[item_index : item_index + 1]
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                signature = compute_grad_signature(
                    model,
                    item_batch,
                    params,
                    device=device,
                    normalize=False,
                )
                source_task = int(_example_field(examples[offset + item_index], "task_id"))
                raw_scores.append(
                    compute_raw_shared_score(signature, source_task, anchor_grads)
                )
        else:
            signature = compute_grad_signature(
                model,
                batch,
                params,
                device=device,
                normalize=False,
            )
            source_task = int(_example_field(examples[offset], "task_id"))
            raw_scores.append(compute_raw_shared_score(signature, source_task, anchor_grads))
        offset += batch_size
    if was_training:
        model.train()
    return assign_scores_to_examples(
        examples,
        raw_scores,
        share_ratio=share_ratio,
    )


def write_scores_jsonl(path: str | Path, examples: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            record = {
                "uid": _example_field(example, "uid"),
                "task_id": int(_example_field(example, "task_id")),
                "task_name": _example_field(example, "task_name"),
                "raw_score": float(_example_field(example, "raw_score", 0.0)),
                "score": float(_example_field(example, "score", 0.0)),
                "split": _example_field(example, "split", "specific"),
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
