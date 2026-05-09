from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def task_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Config must define a non-empty `tasks` list.")
    return tasks


def task_spec_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    for task in task_specs(config):
        if task["name"] == name:
            return task
    raise KeyError(f"Unknown task: {name}")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def stage_name(index: int, task_name: str) -> str:
    return f"task_{index:02d}_{sanitize_name(task_name)}"


def run_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    train_config = config.get("train", {})
    output_root = Path(train_config.get("output_root", "runs"))
    return output_root / (run_name or config.get("experiment_name", "trace_cl_run"))


def model_path(config: dict[str, Any]) -> str:
    model_config = config.get("model", {})
    path = model_config.get("model_name_or_path")
    if not path:
        raise ValueError("Config model.model_name_or_path is required.")
    return str(path)


def tokenizer_path(config: dict[str, Any]) -> str:
    model_config = config.get("model", {})
    return str(model_config.get("tokenizer_name_or_path") or model_path(config))


def chat_template_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model", {})
    kwargs = dict(model_config.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", False)
    return kwargs
