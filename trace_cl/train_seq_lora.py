from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .config import (
    chat_template_kwargs,
    load_config,
    model_path,
    run_dir,
    save_json,
    save_yaml,
    stage_name,
    task_specs,
    tokenizer_path,
)
from .data import CausalLMPaddedCollator, TraceSFTDataset, load_trace_split


def torch_dtype_from_config(dtype_name: str | None) -> torch.dtype | str:
    if dtype_name in (None, "auto"):
        return "auto"
    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_tokenizer(config: dict[str, Any]):
    model_config = config.get("model", {})
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path(config),
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_fast=model_config.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = model_config.get("padding_side", "right")
    return tokenizer


def load_base_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "dtype": torch_dtype_from_config(model_config.get("dtype", "bfloat16")),
    }
    attn_implementation = model_config.get("attn_implementation")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_path(config), **kwargs)


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    lora_config = dict(config.get("train", {}).get("lora", {}))
    return LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("lora_alpha", 32)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora_config.get("target_modules", [])),
    )


def prepare_model_for_stage(
    config: dict[str, Any],
    previous_adapter: str | Path | None,
):
    model = load_base_model(config)
    train_config = config.get("train", {})
    if train_config.get("gradient_checkpointing", True):
        model.config.use_cache = False

    if previous_adapter:
        model = PeftModel.from_pretrained(
            model, str(previous_adapter), is_trainable=True
        )
    else:
        model = get_peft_model(model, build_lora_config(config))

    if train_config.get("gradient_checkpointing", True) and hasattr(
        model, "enable_input_require_grads"
    ):
        model.enable_input_require_grads()
    return model


def trainer_init_kwargs(tokenizer: Any) -> dict[str, Any]:
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    *,
    max_steps_override: int | None = None,
) -> TrainingArguments:
    train_config = config.get("train", {})
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": bool(train_config.get("overwrite_output_dir", True)),
        "do_train": True,
        "do_eval": False,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "logging_steps": int(train_config.get("logging_steps", 10)),
        "save_steps": int(train_config.get("save_steps", 250)),
        "save_total_limit": int(train_config.get("save_total_limit", 2)),
        "num_train_epochs": float(train_config.get("num_train_epochs", 1)),
        "max_steps": int(
            max_steps_override
            if max_steps_override is not None
            else train_config.get("max_steps", -1)
        ),
        "learning_rate": float(train_config.get("learning_rate", 2e-4)),
        "lr_scheduler_type": train_config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(train_config.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "per_device_train_batch_size": int(
            train_config.get("per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            train_config.get("gradient_accumulation_steps", 8)
        ),
        "gradient_checkpointing": bool(
            train_config.get("gradient_checkpointing", True)
        ),
        "bf16": bool(train_config.get("bf16", True)),
        "fp16": bool(train_config.get("fp16", False)),
        "dataloader_num_workers": int(train_config.get("dataloader_num_workers", 0)),
        "report_to": list(train_config.get("report_to", [])),
        "optim": train_config.get("optim", "adamw_torch"),
        "remove_unused_columns": False,
        "seed": int(config.get("seed", 42)),
    }
    if "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**kwargs)


def build_stage_dataset(
    config: dict[str, Any],
    task: dict[str, Any],
    tokenizer: Any,
    *,
    max_train_samples: int | None = None,
) -> TraceSFTDataset:
    data_config = config.get("data", {})
    records = load_trace_split(
        data_config["root"],
        task["name"],
        data_config.get("train_split", "train"),
        prompt_field=data_config.get("prompt_field", "prompt"),
        response_field=data_config.get("response_field", "answer"),
        max_samples=max_train_samples,
    )
    return TraceSFTDataset(
        records,
        tokenizer,
        max_length=int(task.get("max_length", data_config.get("default_max_length", 512))),
        truncation=task.get("truncation", "tail"),
        chat_template_kwargs=chat_template_kwargs(config),
    )


def save_stage_config(
    config: dict[str, Any],
    stage_dir: Path,
    task: dict[str, Any],
    task_index: int,
    previous_adapter: str | Path | None,
) -> None:
    resolved = dict(config)
    resolved["current_task"] = {
        "index": task_index,
        "name": task["name"],
        "previous_adapter": str(previous_adapter) if previous_adapter else None,
    }
    save_yaml(resolved, stage_dir / "resolved_config.yaml")


def train_one_stage(
    config: dict[str, Any],
    task: dict[str, Any],
    task_index: int,
    tokenizer: Any,
    stage_dir: Path,
    previous_adapter: str | Path | None,
    *,
    max_train_samples: int | None = None,
    max_steps_override: int | None = None,
) -> Path:
    print(f"\n=== Training stage {task_index}: {task['name']} ===", flush=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_stage_config(config, stage_dir, task, task_index, previous_adapter)

    dataset = build_stage_dataset(
        config, task, tokenizer, max_train_samples=max_train_samples
    )
    token_stats = dataset.token_stats()
    token_stats.update(
        {
            "task": task["name"],
            "max_length_budget": int(
                task.get(
                    "max_length",
                    config.get("data", {}).get("default_max_length", 512),
                )
            ),
            "truncation": task.get("truncation", "tail"),
        }
    )
    save_json(token_stats, stage_dir / "token_stats.json")

    model = prepare_model_for_stage(config, previous_adapter)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_arguments(
            config, stage_dir, max_steps_override=max_steps_override
        ),
        train_dataset=dataset,
        data_collator=CausalLMPaddedCollator(tokenizer),
        **trainer_init_kwargs(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.state.save_to_json(str(stage_dir / "trainer_state.json"))
    save_json(train_result.metrics, stage_dir / "train_metrics.json")

    final_adapter = stage_dir / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential TRACE LoRA training.")
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
        help="Optional TrainingArguments max_steps override for smoke tests.",
    )
    parser.add_argument(
        "--tasks",
        help="Optional comma-separated subset of task names, preserving config order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    output_dir = run_dir(config, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "experiment_config.yaml")

    selected = task_specs(config)
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        selected = [task for task in selected if task["name"] in wanted]
        missing = wanted - {task["name"] for task in selected}
        if missing:
            raise ValueError(f"Unknown tasks requested: {sorted(missing)}")

    tokenizer = load_tokenizer(config)
    previous_adapter: Path | None = None
    completed: list[dict[str, Any]] = []
    for task in selected:
        task_index = task_specs(config).index(task) + 1
        stage_dir = output_dir / stage_name(task_index, task["name"])
        previous_adapter = train_one_stage(
            config,
            task,
            task_index,
            tokenizer,
            stage_dir,
            previous_adapter,
            max_train_samples=args.max_train_samples,
            max_steps_override=args.max_steps,
        )
        completed.append(
            {
                "index": task_index,
                "name": task["name"],
                "adapter": str(previous_adapter),
            }
        )
        save_json(
            {
                "completed_tasks": completed,
                "last_completed_adapter_path": str(previous_adapter),
            },
            output_dir / "run_state.json",
        )

    save_json(
        {
            "method": "sequential_lora",
            "num_tasks": len(selected),
            "completed_tasks": completed,
            "last_completed_adapter_path": str(previous_adapter)
            if previous_adapter
            else None,
        },
        output_dir / "final_summary.json",
    )


if __name__ == "__main__":
    main()
