from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_json, sanitize_name, save_json


def task_metric_dirs(eval_dir: str | Path) -> list[Path]:
    root = Path(eval_dir)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metrics.json").exists()
    )


def summarize_eval_dir(eval_dir: str | Path) -> dict[str, Any]:
    eval_dir = Path(eval_dir)
    rows: list[dict[str, Any]] = []
    for task_dir in task_metric_dirs(eval_dir):
        metrics = load_json(task_dir / "metrics.json")
        row = {
            "task": metrics.get("task", task_dir.name),
            "metric": metrics.get("metric"),
            "primary_value": metrics.get("primary_value"),
            "score": metrics.get("score"),
            "num_examples": metrics.get("num_examples"),
        }
        rows.append(row)

    table_path = eval_dir / "metrics_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "metric", "primary_value", "score", "num_examples"],
        )
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    summary = {
        "eval_dir": str(eval_dir),
        "num_tasks": len(rows),
        "macro_score": mean(scores) if scores else 0.0,
        "tasks": rows,
    }
    save_json(summary, eval_dir / "final_summary.json")
    return summary


def summarize_eval_root(eval_root: str | Path) -> dict[str, Any]:
    eval_root = Path(eval_root)
    stage_dirs = sorted(
        path
        for path in eval_root.iterdir()
        if path.is_dir() and any(child.is_dir() for child in path.iterdir())
    )
    stage_summaries = []
    all_tasks: list[str] = []
    matrix_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for stage_dir in stage_dirs:
        summary = summarize_eval_dir(stage_dir)
        stage_summaries.append(summary)
        row: dict[str, Any] = {"stage": stage_dir.name}
        score_row: dict[str, Any] = {"stage": stage_dir.name}
        for task in summary["tasks"]:
            task_name = task["task"]
            all_tasks.append(task_name)
            row[task_name] = task["primary_value"]
            score_row[task_name] = task["score"]
        matrix_rows.append(row)
        score_rows.append(score_row)

    all_tasks = sorted(set(all_tasks))
    matrix_path = eval_root / "cl_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage"] + all_tasks)
        writer.writeheader()
        writer.writerows(matrix_rows)

    score_matrix_path = eval_root / "cl_score_matrix.csv"
    with score_matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage"] + all_tasks)
        writer.writeheader()
        writer.writerows(score_rows)

    continual_metrics = compute_continual_metrics(stage_summaries)
    summary = {
        "eval_root": str(eval_root),
        "num_stages": len(stage_summaries),
        "continual_metrics": continual_metrics,
        "stages": stage_summaries,
    }
    save_json(summary, eval_root / "final_summary.json")
    return summary


def stage_task_name(stage: str) -> str | None:
    match = re.match(r"task_\d+_(.+)$", stage)
    if not match:
        return None
    return match.group(1)


def compute_continual_metrics(stage_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not stage_summaries:
        return {
            "final_average_score": 0.0,
            "average_forgetting": None,
            "bwt": None,
            "fwt": None,
        }

    score_by_stage: list[dict[str, float]] = []
    learned_tasks: list[str | None] = []
    for summary in stage_summaries:
        scores = {
            task["task"]: float(task["score"])
            for task in summary.get("tasks", [])
            if task.get("score") is not None
        }
        score_by_stage.append(scores)
        learned_tasks.append(stage_task_name(Path(summary["eval_dir"]).name))

    final_scores = score_by_stage[-1]
    final_average = mean(final_scores.values()) if final_scores else 0.0

    forgetting_values: list[float] = []
    bwt_values: list[float] = []
    for learned_index, learned_task in enumerate(learned_tasks[:-1]):
        if learned_task is None:
            continue
        task_name = next(
            (
                task
                for task in final_scores
                if sanitize_name(task) == learned_task
            ),
            None,
        )
        if task_name is None:
            continue

        final_score = final_scores.get(task_name)
        initial_score = score_by_stage[learned_index].get(task_name)
        if final_score is None or initial_score is None:
            continue

        history = [
            row[task_name]
            for row in score_by_stage[learned_index:-1]
            if task_name in row
        ]
        if history:
            forgetting_values.append(max(history) - final_score)
        bwt_values.append(final_score - initial_score)

    return {
        "final_average_score": final_average,
        "average_forgetting": mean(forgetting_values)
        if forgetting_values
        else None,
        "bwt": mean(bwt_values) if bwt_values else None,
        "fwt": None,
        "fwt_note": "Requires a pre-training zero-shot evaluation stage.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TRACE eval outputs.")
    parser.add_argument("--eval-dir", help="One eval directory containing task metrics.")
    parser.add_argument("--eval-root", help="Root containing multiple stage eval dirs.")
    args = parser.parse_args()

    if args.eval_root:
        summarize_eval_root(args.eval_root)
    elif args.eval_dir:
        summarize_eval_dir(args.eval_dir)
    else:
        parser.error("Provide --eval-dir or --eval-root")


if __name__ == "__main__":
    main()
