from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .config import (
    chat_template_kwargs,
    load_config,
    model_path,
    run_dir,
    sanitize_name,
    save_json,
    task_specs,
    tokenizer_path,
)
from .data import load_trace_split, render_generation_prompt
from .metrics import clean_generation, compute_task_metrics
from .summarize import summarize_eval_dir


def load_eval_tokenizer(config: dict[str, Any]):
    model_config = config.get("model", {})
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path(config),
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_fast=model_config.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def infer_eval_output_dir(
    config: dict[str, Any],
    adapter_path: str | Path,
    run_name: str | None,
) -> Path:
    adapter = Path(adapter_path)
    stage_name = adapter.parent.name if adapter.name == "final_adapter" else adapter.name
    return run_dir(config, run_name) / config.get("eval", {}).get("output_subdir", "eval") / stage_name


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompts(
    tokenizer: Any,
    records: list[dict[str, str]],
    config: dict[str, Any],
) -> list[str]:
    kwargs = chat_template_kwargs(config)
    return [
        render_generation_prompt(tokenizer, record["prompt"], chat_template_kwargs=kwargs)
        for record in records
    ]


def write_predictions(
    path: str | Path,
    examples: list[dict[str, str]],
    predictions: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example, prediction in zip(examples, predictions):
            f.write(
                json.dumps(
                    {
                        "prompt": example["prompt"],
                        "gold": example["answer"],
                        "prediction": prediction,
                        "normalized_prediction": clean_generation(prediction),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def evaluate_task(
    llm: Any,
    tokenizer: Any,
    config: dict[str, Any],
    task: dict[str, Any],
    adapter_path: str | Path,
    output_dir: Path,
    *,
    split: str,
    max_samples: int | None,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    data_config = config.get("data", {})
    eval_config = config.get("eval", {})
    records = load_trace_split(
        data_config["root"],
        task["name"],
        split,
        prompt_field=data_config.get("prompt_field", "prompt"),
        response_field=data_config.get("response_field", "answer"),
        max_samples=max_samples,
    )
    prompts = build_prompts(tokenizer, records, config)
    sampling_params = SamplingParams(
        temperature=float(eval_config.get("temperature", 0.0)),
        top_p=float(eval_config.get("top_p", 1.0)),
        max_tokens=int(task.get("generate_max_tokens", 128)),
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    lora_request = LoRARequest("trace_lora", 1, str(adapter_path))
    predictions: list[str] = []
    for prompt_batch in batched(prompts, int(eval_config.get("batch_size", 128))):
        outputs = llm.generate(
            prompt_batch,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        predictions.extend(output.outputs[0].text for output in outputs)

    task_dir = output_dir / sanitize_name(task["name"])
    write_predictions(task_dir / "predictions.jsonl", records, predictions)
    metrics = compute_task_metrics(task, records, predictions)
    metrics.update(
        {
            "task": task["name"],
            "split": split,
            "adapter": str(adapter_path),
        }
    )
    save_json(metrics, task_dir / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual vLLM evaluation for TRACE LoRA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter.")
    parser.add_argument("--run-name", help="Override experiment/run directory name.")
    parser.add_argument("--output-dir", help="Where to write eval outputs.")
    parser.add_argument("--split", help="Override eval split; defaults to config eval.split.")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional per-task sample cap for smoke tests.",
    )
    parser.add_argument(
        "--tasks",
        help="Optional comma-separated subset of task names, preserving config order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_config = config.get("eval", {})
    split = args.split or eval_config.get("split", "test")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else infer_eval_output_dir(config, args.adapter, args.run_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = task_specs(config)
    if args.tasks:
        wanted = {name.strip() for name in args.tasks.split(",") if name.strip()}
        selected = [task for task in selected if task["name"] in wanted]
        missing = wanted - {task["name"] for task in selected}
        if missing:
            raise ValueError(f"Unknown tasks requested: {sorted(missing)}")

    tokenizer = load_eval_tokenizer(config)

    from vllm import LLM

    lora_rank = int(config.get("train", {}).get("lora", {}).get("r", 16))
    llm = LLM(
        model=model_path(config),
        tokenizer=tokenizer_path(config),
        trust_remote_code=config.get("model", {}).get("trust_remote_code", True),
        dtype=eval_config.get("dtype", "bfloat16"),
        max_model_len=int(eval_config.get("max_model_len", 32768)),
        enable_lora=True,
        max_lora_rank=lora_rank,
        tensor_parallel_size=int(eval_config.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(eval_config.get("gpu_memory_utilization", 0.9)),
    )

    all_metrics = []
    for task in selected:
        print(f"Evaluating {task['name']} on {split}", flush=True)
        all_metrics.append(
            evaluate_task(
                llm,
                tokenizer,
                config,
                task,
                args.adapter,
                output_dir,
                split=split,
                max_samples=args.max_samples,
            )
        )
    save_json({"tasks": all_metrics}, output_dir / "raw_metrics.json")
    summarize_eval_dir(output_dir)


if __name__ == "__main__":
    main()
