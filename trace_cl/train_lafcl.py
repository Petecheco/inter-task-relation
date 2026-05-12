from __future__ import annotations

import argparse
import gc
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler, set_seed

from .anchors import build_anchor_records, select_anchors, write_jsonl
from .config import (
    chat_template_kwargs,
    load_config,
    run_dir,
    save_json,
    save_yaml,
    stage_name,
    task_specs,
)
from .data import CausalLMPaddedCollator, TraceSFTDataset, load_trace_split
from .features import extract_features
from .fisher import compute_layerwise_fisher, fisher_json_payload, select_top_layers
from .grad_signature import compute_anchor_gradient, compute_loss
from .lora_utils import (
    add_current_grads_to_accumulator,
    build_layer_to_lora_params,
    get_lora_named_parameters,
    init_lora_grad_accumulator,
    lora_parameters,
    mark_only_lora_trainable,
    set_model_grads_from_accumulator,
)
from .replay_buffer import ReplayBuffer, TrainExample, examples_to_records
from .scoring import score_example_dataset, write_scores_jsonl
from .train_seq_lora import load_tokenizer, prepare_model_for_stage


class AdaptiveSFTDataset(Dataset):
    def __init__(
        self,
        examples: list[TrainExample],
        tokenizer: Any,
        config: dict[str, Any],
        task_by_id: dict[int, dict[str, Any]],
    ) -> None:
        self.examples = list(examples)
        self.datasets_by_task: dict[int, TraceSFTDataset] = {}
        self.local_indices: list[tuple[int, int]] = []

        grouped: dict[int, list[TrainExample]] = {}
        for example in self.examples:
            grouped.setdefault(int(example.task_id), []).append(example)

        for task_id, task_examples in grouped.items():
            task = task_by_id[task_id]
            dataset = encode_lafcl_task_dataset(
                config,
                task,
                tokenizer,
                examples_to_records(task_examples),
            )
            self.datasets_by_task[task_id] = dataset

        task_offsets: dict[int, int] = {}
        for example in self.examples:
            task_id = int(example.task_id)
            local_index = task_offsets.get(task_id, 0)
            self.local_indices.append((task_id, local_index))
            task_offsets[task_id] = local_index + 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_id, local_index = self.local_indices[index]
        feature = dict(self.datasets_by_task[task_id][local_index])
        example = self.examples[index]
        feature.update(
            {
                "uid": example.uid,
                "task_id": int(example.task_id),
                "split": example.split,
            }
        )
        return feature


class AdaptiveCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.base = CausalLMPaddedCollator(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_features = [
            {"input_ids": item["input_ids"], "labels": item["labels"]}
            for item in features
        ]
        batch = self.base(tensor_features)
        batch["task_ids"] = torch.tensor(
            [int(item["task_id"]) for item in features],
            dtype=torch.long,
        )
        batch["splits"] = [str(item["split"]) for item in features]
        batch["uids"] = [str(item["uid"]) for item in features]
        return batch


def select_batch_rows(batch: dict[str, Any], indices: list[int]) -> dict[str, torch.Tensor]:
    return {
        key: value[indices]
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key in {"input_ids", "attention_mask", "labels"}
    }


def sample_train_batch(
    current_dataset: AdaptiveSFTDataset,
    replay_dataset: AdaptiveSFTDataset | None,
    batch_size: int,
    replay_ratio: float,
    rng: random.Random,
    collator: AdaptiveCollator,
) -> dict[str, Any]:
    if len(current_dataset) == 0:
        raise ValueError("Current task dataset is empty.")

    replay_available = replay_dataset is not None and len(replay_dataset) > 0
    replay_count = (
        int(round(batch_size * replay_ratio)) if replay_available else 0
    )
    replay_count = max(0, min(batch_size, replay_count))
    current_count = max(1, batch_size - replay_count)
    if current_count + replay_count > batch_size:
        current_count = batch_size - replay_count

    items: list[dict[str, Any]] = []
    for _ in range(current_count):
        items.append(current_dataset[rng.randrange(len(current_dataset))])
    if replay_available and replay_dataset is not None:
        for _ in range(replay_count):
            items.append(replay_dataset[rng.randrange(len(replay_dataset))])
    rng.shuffle(items)
    return collator(items)


def make_examples(
    records: list[dict[str, str]],
    *,
    task_id: int,
    task_name: str,
    source_split: str,
) -> list[TrainExample]:
    return [
        TrainExample.from_record(
            record,
            task_id=task_id,
            task_name=task_name,
            sample_index=index,
            source_split=source_split,
        )
        for index, record in enumerate(records)
    ]


def encode_lafcl_task_dataset(
    config: dict[str, Any],
    task: dict[str, Any],
    tokenizer: Any,
    records: list[dict[str, str]],
) -> TraceSFTDataset:
    data_config = config.get("data", {})
    return TraceSFTDataset(
        records,
        tokenizer,
        max_length=int(
            task.get("max_length", data_config.get("default_max_length", 512))
        ),
        truncation=task.get("truncation", "tail"),
        chat_template_kwargs=chat_template_kwargs(config),
    )


def task_dataset_for_examples(
    config: dict[str, Any],
    task: dict[str, Any],
    tokenizer: Any,
    examples: list[TrainExample],
) -> TraceSFTDataset:
    return encode_lafcl_task_dataset(
        config,
        task,
        tokenizer,
        examples_to_records(examples),
    )


def dataloader_for_examples(
    config: dict[str, Any],
    task: dict[str, Any],
    tokenizer: Any,
    examples: list[TrainExample],
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = task_dataset_for_examples(config, task, tokenizer, examples)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        collate_fn=CausalLMPaddedCollator(tokenizer),
    )


def resolve_device(config: dict[str, Any], raw_device: str | None = None) -> torch.device:
    device_name = raw_device or config.get("train", {}).get("device")
    if device_name:
        return torch.device(str(device_name))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analysis_examples(
    examples: list[TrainExample],
    config: dict[str, Any],
) -> list[TrainExample]:
    max_samples = config.get("anchor", {}).get("max_samples_for_clustering")
    if max_samples is None or int(max_samples) <= 0:
        return examples
    return examples[: int(max_samples)]


def rescore_buffer(
    model: torch.nn.Module,
    tokenizer: Any,
    config: dict[str, Any],
    buffer: ReplayBuffer,
    task_by_id: dict[int, dict[str, Any]],
    anchor_grads: dict[int, torch.Tensor],
    *,
    device: torch.device,
) -> None:
    if len(buffer) == 0:
        return
    scored: list[TrainExample] = []
    micro_batch_size = int(config.get("gradient_signature", {}).get("micro_batch_size", 1))
    for source_task_id, examples in buffer.by_task().items():
        task = task_by_id[source_task_id]
        dataset = task_dataset_for_examples(config, task, tokenizer, examples)
        scored.extend(
            score_example_dataset(
                model,
                tokenizer,
                examples,
                dataset,
                anchor_grads,
                share_ratio=buffer.share_ratio,
                micro_batch_size=micro_batch_size,
                device=device,
            )
        )
    buffer.replace_examples(scored)


def train_one_task_adaptive(
    model: torch.nn.Module,
    tokenizer: Any,
    config: dict[str, Any],
    task_stats: dict[int, dict[str, Any]],
    current_examples: list[TrainExample],
    replay_examples: list[TrainExample],
    task_by_id: dict[int, dict[str, Any]],
    stage_dir: Path,
    *,
    device: torch.device,
    max_steps_override: int | None,
) -> dict[str, Any]:
    train_config = config.get("train", {})
    adaptive_config = config.get("adaptive_training", {})
    batch_size = int(train_config.get("per_device_train_batch_size", 1))
    grad_accum_steps = int(train_config.get("gradient_accumulation_steps", 1))
    replay_ratio = float(adaptive_config.get("replay_ratio", 0.5))
    learning_rate = float(train_config.get("learning_rate", 2e-4))
    weight_decay = float(train_config.get("weight_decay", 0.0))
    max_grad_norm = float(adaptive_config.get("max_grad_norm", train_config.get("max_grad_norm", 1.0)))
    epochs = float(train_config.get("num_train_epochs", 1))
    configured_max_steps = int(train_config.get("max_steps", -1))
    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else configured_max_steps
    )
    if max_steps <= 0:
        current_per_micro = max(1, batch_size - int(round(batch_size * replay_ratio)))
        examples_per_step = max(1, current_per_micro * max(1, grad_accum_steps))
        max_steps = max(1, math.ceil(len(current_examples) / examples_per_step * epochs))

    current_dataset = AdaptiveSFTDataset(current_examples, tokenizer, config, task_by_id)
    replay_dataset = (
        AdaptiveSFTDataset(replay_examples, tokenizer, config, task_by_id)
        if replay_examples
        else None
    )
    collator = AdaptiveCollator(tokenizer)
    rng = random.Random(int(config.get("seed", 42)) + int(current_examples[0].task_id) * 1009)

    optimizer = torch.optim.AdamW(
        lora_parameters(model),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        name=str(train_config.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(max_steps * float(train_config.get("warmup_ratio", 0.03))),
        num_training_steps=max_steps,
    )

    logging_steps = int(train_config.get("logging_steps", 10))
    save_steps = int(train_config.get("save_steps", 0) or 0)
    train_losses: list[float] = []
    shared_seen = 0
    specific_seen = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    lora_params = lora_parameters(model)

    for step in range(1, max_steps + 1):
        accumulator = init_lora_grad_accumulator(model)
        step_loss = 0.0
        step_shared = 0
        step_specific = 0

        for _ in range(max(1, grad_accum_steps)):
            batch = sample_train_batch(
                current_dataset,
                replay_dataset,
                batch_size,
                replay_ratio,
                rng,
                collator,
            )
            actual_batch_size = len(batch["uids"])
            shared_indices = [
                index for index, split in enumerate(batch["splits"]) if split == "share"
            ]
            specific_by_task: dict[int, list[int]] = {}
            for index, split in enumerate(batch["splits"]):
                if split == "share":
                    continue
                task_id = int(batch["task_ids"][index].item())
                specific_by_task.setdefault(task_id, []).append(index)

            if shared_indices:
                model.zero_grad(set_to_none=True)
                sub_batch = {
                    key: value.to(device)
                    for key, value in select_batch_rows(batch, shared_indices).items()
                }
                scale = len(shared_indices) / actual_batch_size / max(1, grad_accum_steps)
                loss = compute_loss(model, sub_batch) * scale
                loss.backward()
                add_current_grads_to_accumulator(model, accumulator)
                model.zero_grad(set_to_none=True)
                step_loss += float(loss.detach().float().cpu())
                step_shared += len(shared_indices)

            for task_id, indices in specific_by_task.items():
                allowed_layers = set(task_stats[int(task_id)]["key_layers"])
                if not allowed_layers:
                    continue
                model.zero_grad(set_to_none=True)
                sub_batch = {
                    key: value.to(device)
                    for key, value in select_batch_rows(batch, indices).items()
                }
                scale = len(indices) / actual_batch_size / max(1, grad_accum_steps)
                loss = compute_loss(model, sub_batch) * scale
                loss.backward()
                add_current_grads_to_accumulator(
                    model,
                    accumulator,
                    allowed_layers=allowed_layers,
                )
                model.zero_grad(set_to_none=True)
                step_loss += float(loss.detach().float().cpu())
                step_specific += len(indices)

        set_model_grads_from_accumulator(model, accumulator)
        if max_grad_norm > 0:
            clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        train_losses.append(step_loss)
        shared_seen += step_shared
        specific_seen += step_specific
        if step == 1 or step == max_steps or (logging_steps > 0 and step % logging_steps == 0):
            print(
                f"[Train] step {step}/{max_steps} loss={step_loss:.6f} "
                f"shared={step_shared} specific={step_specific}",
                flush=True,
            )

        if save_steps > 0 and step % save_steps == 0:
            checkpoint_dir = stage_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))

    return {
        "max_steps": max_steps,
        "mean_loss": sum(train_losses) / len(train_losses) if train_losses else None,
        "shared_examples_seen": shared_seen,
        "specific_examples_seen": specific_seen,
        "replay_examples_available": len(replay_examples),
    }


def train_one_stage(
    config: dict[str, Any],
    task: dict[str, Any],
    task_id: int,
    tokenizer: Any,
    stage_dir: Path,
    previous_adapter: Path | None,
    task_stats: dict[int, dict[str, Any]],
    replay_buffer: ReplayBuffer,
    task_by_id: dict[int, dict[str, Any]],
    *,
    device: torch.device,
    max_train_samples: int | None,
    max_steps_override: int | None,
) -> Path:
    print(f"\n=== LAF-CL task {task_id}: {task['name']} ===", flush=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(
        {
            **config,
            "current_task": {
                "index": task_id,
                "name": task["name"],
                "previous_adapter": str(previous_adapter) if previous_adapter else None,
            },
        },
        stage_dir / "resolved_config.yaml",
    )

    data_config = config.get("data", {})
    train_split = str(data_config.get("train_split", "train"))
    records = load_trace_split(
        data_config["root"],
        task["name"],
        train_split,
        prompt_field=data_config.get("prompt_field", "prompt"),
        response_field=data_config.get("response_field", "answer"),
        max_samples=max_train_samples,
    )
    current_examples = make_examples(
        records,
        task_id=task_id,
        task_name=task["name"],
        source_split=train_split,
    )

    model = prepare_model_for_stage(config, previous_adapter)
    mark_only_lora_trainable(model)
    model.to(device)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    anchor_config = config.get("anchor", {})
    feature_examples = analysis_examples(current_examples, config)
    features = extract_features(
        model,
        tokenizer,
        feature_examples,
        config,
        task=task,
        device=device,
    )
    anchor_examples = select_anchors(
        feature_examples,
        features,
        int(anchor_config.get("num_clusters", 8)),
        int(anchor_config.get("anchors_per_cluster", 4)),
        seed=int(config.get("seed", 42)),
    )
    anchor_records = build_anchor_records(
        feature_examples,
        features,
        anchor_examples,
        int(anchor_config.get("num_clusters", 8)),
        seed=int(config.get("seed", 42)),
    )
    write_jsonl(stage_dir / "anchors.jsonl", anchor_records)
    print(f"[Task {task_id}] selected anchors: {len(anchor_examples)}", flush=True)

    fisher_config = config.get("fisher", {})
    anchor_loader = dataloader_for_examples(
        config,
        task,
        tokenizer,
        anchor_examples,
        batch_size=int(fisher_config.get("micro_batch_size", 1)),
    )
    layer_to_params = build_layer_to_lora_params(model)
    fisher = compute_layerwise_fisher(
        model,
        anchor_loader,
        layer_to_params,
        normalize_by_numel=bool(fisher_config.get("normalize_by_numel", True)),
        device=device,
    )
    key_layers = select_top_layers(
        fisher,
        int(fisher_config.get("top_m_layers", 8)),
    )
    save_json(
        fisher_json_payload(task_id, fisher, key_layers),
        stage_dir / "fisher.json",
    )
    save_json(
        {"task_id": task_id, "selected_layers": key_layers},
        stage_dir / "key_layers.json",
    )
    print(f"[Task {task_id}] top fisher layers: {key_layers}", flush=True)

    anchor_grad = compute_anchor_gradient(
        model,
        anchor_loader,
        get_lora_named_parameters(model),
        device=device,
        normalize=True,
    )
    torch.save(anchor_grad, stage_dir / "anchor_grad.pt")
    task_stats[task_id] = {
        "anchor_grad": anchor_grad,
        "key_layers": key_layers,
        "fisher": fisher,
    }
    anchor_grads = {
        int(seen_task_id): stats["anchor_grad"]
        for seen_task_id, stats in task_stats.items()
    }

    rescore_buffer(
        model,
        tokenizer,
        config,
        replay_buffer,
        task_by_id,
        anchor_grads,
        device=device,
    )
    old_replay_examples = list(replay_buffer.examples)

    current_dataset = task_dataset_for_examples(config, task, tokenizer, current_examples)
    current_examples = score_example_dataset(
        model,
        tokenizer,
        current_examples,
        current_dataset,
        anchor_grads,
        share_ratio=replay_buffer.share_ratio,
        micro_batch_size=int(config.get("gradient_signature", {}).get("micro_batch_size", 1)),
        device=device,
    )
    write_scores_jsonl(stage_dir / "scores.jsonl", current_examples)
    shared_count = sum(1 for example in current_examples if example.split == "share")
    specific_count = len(current_examples) - shared_count
    print(
        f"[Task {task_id}] shared examples: {shared_count} / {len(current_examples)}",
        flush=True,
    )
    print(
        f"[Task {task_id}] specific examples: {specific_count} / {len(current_examples)}",
        flush=True,
    )

    train_metrics = train_one_task_adaptive(
        model,
        tokenizer,
        config,
        task_stats,
        current_examples,
        old_replay_examples,
        task_by_id,
        stage_dir,
        device=device,
        max_steps_override=max_steps_override,
    )
    save_json(train_metrics, stage_dir / "train_metrics.json")

    replay_buffer.update_with_task_examples(task_id, current_examples)
    buffer_path = run_dir(config) / "buffer.jsonl"
    replay_buffer.save(buffer_path)
    print(f"[Buffer] size: {len(replay_buffer)}", flush=True)
    print(f"[Buffer] task distribution: {replay_buffer.task_distribution()}", flush=True)

    final_adapter = stage_dir / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-adaptive TRACE LoRA training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", help="Override experiment/run directory name.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Optional per-task sample cap for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional optimizer-step cap for smoke tests.",
    )
    parser.add_argument(
        "--tasks",
        help="Optional comma-separated subset of task names, preserving config order.",
    )
    parser.add_argument("--device", help="Torch device override, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_name:
        config = dict(config)
        config["experiment_name"] = args.run_name
    set_seed(int(config.get("seed", 42)))
    output_dir = run_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "experiment_config.yaml")

    selected = task_specs(config)
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        selected = [task for task in selected if task["name"] in wanted]
        missing = wanted - {task["name"] for task in selected}
        if missing:
            raise ValueError(f"Unknown tasks requested: {sorted(missing)}")

    all_tasks = task_specs(config)
    task_by_id = {index + 1: task for index, task in enumerate(all_tasks)}
    device = resolve_device(config, args.device)
    tokenizer = load_tokenizer(config)

    buffer_config = config.get("buffer", {})
    replay_buffer = ReplayBuffer.load(
        output_dir / "buffer.jsonl",
        capacity=int(buffer_config.get("capacity", 512)),
        share_ratio=float(buffer_config.get("share_ratio", 0.3)),
        per_task_balance=bool(buffer_config.get("per_task_balance", True)),
    )

    previous_adapter: Path | None = None
    completed: list[dict[str, Any]] = []
    task_stats: dict[int, dict[str, Any]] = {}
    for task in selected:
        task_id = all_tasks.index(task) + 1
        stage_dir = output_dir / stage_name(task_id, task["name"])
        previous_adapter = train_one_stage(
            config,
            task,
            task_id,
            tokenizer,
            stage_dir,
            previous_adapter,
            task_stats,
            replay_buffer,
            task_by_id,
            device=device,
            max_train_samples=args.max_train_samples,
            max_steps_override=args.max_steps,
        )
        completed.append(
            {
                "index": task_id,
                "name": task["name"],
                "adapter": str(previous_adapter),
            }
        )
        torch.save(task_stats, output_dir / "task_stats.pt")
        save_json(
            {
                "completed_tasks": completed,
                "last_completed_adapter_path": str(previous_adapter),
            },
            output_dir / "run_state.json",
        )

    save_json(
        {
            "method": "lafcl",
            "num_tasks": len(selected),
            "buffer": buffer_config,
            "adaptive_training": config.get("adaptive_training", {}),
            "completed_tasks": completed,
            "last_completed_adapter_path": str(previous_adapter)
            if previous_adapter
            else None,
        },
        output_dir / "final_summary.json",
    )


if __name__ == "__main__":
    main()
