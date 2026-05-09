from __future__ import annotations

from pathlib import Path

import pytest

from trace_cl.data import IGNORE_INDEX, TraceSFTDataset, render_generation_prompt
from trace_cl.metrics import (
    compute_task_metrics,
    edit_distance,
    extract_choice,
    extract_number,
    rouge_l_f1,
    sari_sentence,
)
from trace_cl.summarize import compute_continual_metrics


MODEL_PATH = Path("/data/huggingface_models/Qwen3-1.7B")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="local Qwen3 tokenizer unavailable")
def test_qwen3_chat_template_disables_thinking_and_masks_prompt():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    prompt = "Choose A or B.\nAnswer:"
    rendered = render_generation_prompt(
        tokenizer, prompt, chat_template_kwargs={"enable_thinking": False}
    )
    assert "<think>\n\n</think>" in rendered

    dataset = TraceSFTDataset(
        [{"prompt": prompt, "answer": "A"}],
        tokenizer,
        max_length=128,
        truncation="tail",
        chat_template_kwargs={"enable_thinking": False},
    )
    item = dataset[0]
    first_label = next(i for i, label in enumerate(item["labels"]) if label != IGNORE_INDEX)
    assert all(label == IGNORE_INDEX for label in item["labels"][:first_label])
    unmasked = [
        token_id for token_id, label in zip(item["input_ids"], item["labels"]) if label != IGNORE_INDEX
    ]
    decoded = tokenizer.decode(unmasked, skip_special_tokens=False)
    assert decoded.startswith("A")
    assert "<|im_end|>" in decoded


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="local Qwen3 tokenizer unavailable")
def test_prompt_truncation_keeps_answer():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dataset = TraceSFTDataset(
        [{"prompt": "word " * 2000, "answer": "FINAL_ANSWER"}],
        tokenizer,
        max_length=96,
        truncation="tail",
        chat_template_kwargs={"enable_thinking": False},
    )
    item = dataset[0]
    unmasked = [
        token_id for token_id, label in zip(item["input_ids"], item["labels"]) if label != IGNORE_INDEX
    ]
    decoded = tokenizer.decode(unmasked, skip_special_tokens=False)
    assert "FINAL_ANSWER" in decoded
    assert dataset.token_stats()["truncated_count"] == 1


def test_choice_and_number_extraction():
    assert extract_choice("The answer is B.", ["A", "B", "C"]) == "B"
    assert extract_choice("答案：C\n理由如下", ["A", "B", "C"]) == "C"
    assert extract_number("There are 1,024 cars.") == extract_number("1024")


def test_core_metrics_are_finite_and_directional():
    assert rouge_l_f1("a b c", "a b c") == pytest.approx(1.0)
    assert rouge_l_f1("", "a b") == pytest.approx(0.0)
    assert edit_distance("kitten", "sitting") == 3
    perfect = sari_sentence("The quick brown fox.", "The quick fox.", "The quick fox.")
    worse = sari_sentence("The quick brown fox.", "completely different", "The quick fox.")
    assert 0 <= worse <= 100
    assert 0 <= perfect <= 100
    assert perfect > worse


def test_task_metric_dispatch():
    task = {"name": "FOMC", "metric": "accuracy", "answer_type": "choice", "choices": ["A", "B", "C"]}
    examples = [{"prompt": "p", "answer": "A"}, {"prompt": "p", "answer": "B"}]
    metrics = compute_task_metrics(task, examples, ["A", "C"])
    assert metrics["primary_value"] == pytest.approx(0.5)

    task = {"name": "Py150", "metric": "edit_distance", "answer_type": "code"}
    metrics = compute_task_metrics(task, [{"prompt": "p", "answer": "abc"}], ["axc"])
    assert metrics["primary_value"] == 1


def test_continual_metrics_from_stage_scores():
    stages = [
        {
            "eval_dir": "runs/x/eval/task_01_A",
            "tasks": [
                {"task": "A", "score": 0.8},
                {"task": "B", "score": 0.1},
            ],
        },
        {
            "eval_dir": "runs/x/eval/task_02_B",
            "tasks": [
                {"task": "A", "score": 0.6},
                {"task": "B", "score": 0.7},
            ],
        },
    ]
    metrics = compute_continual_metrics(stages)
    assert metrics["final_average_score"] == pytest.approx(0.65)
    assert metrics["average_forgetting"] == pytest.approx(0.2)
    assert metrics["bwt"] == pytest.approx(-0.2)
    assert metrics["fwt"] is None
