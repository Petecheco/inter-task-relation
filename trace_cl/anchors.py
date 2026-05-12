from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def _example_field(example: Any, field: str, default: Any = None) -> Any:
    if isinstance(example, dict):
        return example.get(field, default)
    return getattr(example, field, default)


def _anchor_record(
    example: Any,
    *,
    local_index: int,
    cluster_id: int,
    distance: float | None,
) -> dict[str, Any]:
    return {
        "uid": _example_field(example, "uid"),
        "task_id": _example_field(example, "task_id"),
        "task_name": _example_field(example, "task_name"),
        "sample_index": _example_field(example, "sample_index", local_index),
        "split": _example_field(example, "source_split", "train"),
        "cluster": int(cluster_id),
        "task_local_index": int(local_index),
        "own_centroid_distance": None if distance is None else float(distance),
    }


def _to_numpy(features: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        return features.detach().float().cpu().numpy()
    return np.asarray(features, dtype=np.float32)


def select_anchors(
    examples: list[Any],
    features: torch.Tensor | np.ndarray,
    num_clusters: int,
    anchors_per_cluster: int,
    *,
    seed: int = 42,
) -> list[Any]:
    matrix = _to_numpy(features)
    if len(examples) != matrix.shape[0]:
        raise ValueError(
            f"Anchor examples/features length mismatch: {len(examples)} vs {matrix.shape[0]}"
        )
    if not examples:
        return []

    requested_k = max(1, int(num_clusters))
    per_cluster = max(1, int(anchors_per_cluster))
    if len(examples) <= requested_k:
        return list(examples)

    num_unique = int(np.unique(matrix, axis=0).shape[0])
    effective_k = max(1, min(requested_k, len(examples), num_unique))
    if effective_k == 1:
        center = matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(matrix - center, axis=1)
        order = np.lexsort((np.arange(len(examples)), distances))
        return [examples[int(index)] for index in order[:per_cluster]]

    model = MiniBatchKMeans(
        n_clusters=effective_k,
        random_state=seed,
        n_init=10,
        batch_size=max(32, effective_k * per_cluster * 4),
    )
    labels = model.fit_predict(matrix)
    centers = model.cluster_centers_
    selected: list[Any] = []
    selected_indices: set[int] = set()

    for cluster_id in sorted(np.unique(labels)):
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            continue
        distances = np.linalg.norm(matrix[members] - centers[int(cluster_id)], axis=1)
        order = np.lexsort((members, distances))
        for ordered_index in order[:per_cluster]:
            local_index = int(members[int(ordered_index)])
            if local_index in selected_indices:
                continue
            selected_indices.add(local_index)
            selected.append(examples[local_index])

    return selected


def build_anchor_records(
    examples: list[Any],
    features: torch.Tensor | np.ndarray,
    selected: list[Any],
    num_clusters: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    matrix = _to_numpy(features)
    if not selected:
        return []
    uid_to_index = {
        str(_example_field(example, "uid", index)): index
        for index, example in enumerate(examples)
    }
    if len(examples) <= max(1, int(num_clusters)):
        return [
            _anchor_record(
                example,
                local_index=uid_to_index[str(_example_field(example, "uid", index))],
                cluster_id=index,
                distance=None,
            )
            for index, example in enumerate(selected)
        ]

    effective_k = max(1, min(int(num_clusters), len(examples), int(np.unique(matrix, axis=0).shape[0])))
    model = MiniBatchKMeans(
        n_clusters=effective_k,
        random_state=seed,
        n_init=10,
        batch_size=max(32, effective_k * 8),
    )
    labels = model.fit_predict(matrix)
    centers = model.cluster_centers_
    records: list[dict[str, Any]] = []
    for example in selected:
        uid = str(_example_field(example, "uid"))
        local_index = uid_to_index[uid]
        cluster_id = int(labels[local_index])
        distance = float(np.linalg.norm(matrix[local_index] - centers[cluster_id]))
        records.append(
            _anchor_record(
                example,
                local_index=local_index,
                cluster_id=cluster_id,
                distance=distance,
            )
        )
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
