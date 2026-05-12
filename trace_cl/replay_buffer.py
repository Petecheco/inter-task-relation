from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainExample:
    uid: str
    prompt: str
    answer: str
    task_id: int
    task_name: str
    sample_index: int
    source_split: str = "train"
    raw_score: float = 0.0
    score: float = 0.0
    split: str = "specific"

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        *,
        task_id: int,
        task_name: str,
        sample_index: int,
        source_split: str = "train",
    ) -> "TrainExample":
        return cls(
            uid=f"{task_id}:{source_split}:{sample_index}",
            prompt=str(record["prompt"]),
            answer=str(record["answer"]),
            task_id=int(task_id),
            task_name=str(task_name),
            sample_index=int(sample_index),
            source_split=str(source_split),
        )

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "TrainExample":
        return cls(
            uid=str(record["uid"]),
            prompt=str(record.get("prompt", record.get("input", ""))),
            answer=str(record.get("answer", record.get("output", ""))),
            task_id=int(record["task_id"]),
            task_name=str(record.get("task_name", record.get("task", ""))),
            sample_index=int(record.get("sample_index", 0)),
            source_split=str(record.get("source_split", record.get("split_name", "train"))),
            raw_score=float(record.get("raw_score", 0.0)),
            score=float(record.get("score", 0.0)),
            split=str(record.get("split", "specific")),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def to_trace_record(self) -> dict[str, str]:
        return {"prompt": self.prompt, "answer": self.answer}


def examples_to_records(examples: list[TrainExample]) -> list[dict[str, str]]:
    return [example.to_trace_record() for example in examples]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        share_ratio: float,
        *,
        per_task_balance: bool = True,
    ) -> None:
        self.capacity = max(0, int(capacity))
        self.share_ratio = float(share_ratio)
        self.per_task_balance = bool(per_task_balance)
        self.examples: list[TrainExample] = []

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int,
        share_ratio: float,
        per_task_balance: bool = True,
    ) -> "ReplayBuffer":
        buffer = cls(capacity, share_ratio, per_task_balance=per_task_balance)
        path = Path(path)
        if not path.exists():
            return buffer
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    buffer.examples.append(TrainExample.from_json(json.loads(line)))
        buffer._trim_to_capacity()
        return buffer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for example in self.examples:
                f.write(json.dumps(example.to_json(), ensure_ascii=False))
                f.write("\n")

    def __len__(self) -> int:
        return len(self.examples)

    def by_task(self) -> dict[int, list[TrainExample]]:
        grouped: dict[int, list[TrainExample]] = {}
        for example in self.examples:
            grouped.setdefault(int(example.task_id), []).append(example)
        return dict(sorted(grouped.items()))

    def task_distribution(self) -> dict[int, int]:
        return {task_id: len(examples) for task_id, examples in self.by_task().items()}

    def update_with_task_examples(
        self,
        task_id: int,
        scored_examples: list[TrainExample],
        features: Any | None = None,
    ) -> None:
        del features
        remaining = [item for item in self.examples if item.task_id != int(task_id)]
        self.examples = remaining + list(scored_examples)
        self._trim_to_capacity()

    def replace_examples(self, examples: list[TrainExample]) -> None:
        self.examples = list(examples)
        self._trim_to_capacity()

    def _select_task_examples(
        self,
        examples: list[TrainExample],
        task_capacity: int,
    ) -> list[TrainExample]:
        if task_capacity <= 0:
            return []
        if len(examples) <= task_capacity:
            return sorted(examples, key=lambda item: item.uid)

        shared = [item for item in examples if item.split == "share"]
        specific = [item for item in examples if item.split != "share"]
        share_capacity = int(round(task_capacity * self.share_ratio))
        share_capacity = max(0, min(task_capacity, share_capacity))
        specific_capacity = task_capacity - share_capacity

        selected_shared = sorted(
            shared,
            key=lambda item: (-float(item.score), item.uid),
        )[:share_capacity]
        selected_specific = sorted(
            specific,
            key=lambda item: (float(item.score), item.uid),
        )[:specific_capacity]

        selected = selected_shared + selected_specific
        if len(selected) < task_capacity:
            selected_uids = {item.uid for item in selected}
            overflow = [item for item in examples if item.uid not in selected_uids]
            overflow = sorted(
                overflow,
                key=lambda item: (
                    0 if item.split == "share" else 1,
                    -float(item.score) if item.split == "share" else float(item.score),
                    item.uid,
                ),
            )
            selected.extend(overflow[: task_capacity - len(selected)])
        return sorted(selected, key=lambda item: item.uid)

    def _trim_to_capacity(self) -> None:
        if self.capacity <= 0:
            self.examples = []
            return
        if len(self.examples) <= self.capacity:
            self.examples = sorted(self.examples, key=lambda item: item.uid)
            return

        grouped = self.by_task()
        if not self.per_task_balance:
            shared = sorted(
                [item for item in self.examples if item.split == "share"],
                key=lambda item: (-float(item.score), item.uid),
            )
            specific = sorted(
                [item for item in self.examples if item.split != "share"],
                key=lambda item: (float(item.score), item.uid),
            )
            self.examples = (shared + specific)[: self.capacity]
            self.examples = sorted(self.examples, key=lambda item: item.uid)
            return

        task_ids = sorted(grouped)
        base = self.capacity // len(task_ids)
        remainder = self.capacity % len(task_ids)
        selected: list[TrainExample] = []
        for offset, task_id in enumerate(task_ids):
            task_capacity = base + (1 if offset < remainder else 0)
            selected.extend(self._select_task_examples(grouped[task_id], task_capacity))
        self.examples = sorted(selected[: self.capacity], key=lambda item: item.uid)
