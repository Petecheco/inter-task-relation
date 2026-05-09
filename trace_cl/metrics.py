from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from statistics import mean
from typing import Any


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
EMPTY_THINK_RE = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)
CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def clean_generation(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = EMPTY_THINK_RE.sub("", text)
    if "</think>" in text and text.lstrip().startswith("<think>"):
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_choice(text: str, choices: list[str] | None = None) -> str | None:
    valid = {choice.upper() for choice in (choices or ["A", "B", "C", "D", "E"])}
    cleaned = clean_generation(text).upper()
    match = CHOICE_RE.search(cleaned)
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()
    for char in cleaned:
        if char in valid:
            return char
    return None


def extract_number(text: str) -> Decimal | None:
    cleaned = clean_generation(text).replace(",", "")
    match = NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def normalize_text(text: str) -> str:
    return " ".join(clean_generation(text).split())


def token_list(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", clean_generation(text).lower(), re.UNICODE)


def lcs_length(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    previous = [0] * (len(short) + 1)
    for token in long:
        current = [0]
        for idx, short_token in enumerate(short, start=1):
            if token == short_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = token_list(prediction)
    ref_tokens = token_list(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def safe_div(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def counter_intersection_size(a: Counter[Any], b: Counter[Any]) -> int:
    return sum((a & b).values())


def sari_sentence(source: str, prediction: str, reference: str) -> float:
    source_tokens = token_list(source)
    pred_tokens = token_list(prediction)
    ref_tokens = token_list(reference)
    scores: list[float] = []

    for n in range(1, 5):
        source_grams = ngrams(source_tokens, n)
        pred_grams = ngrams(pred_tokens, n)
        ref_grams = ngrams(ref_tokens, n)

        sys_add = pred_grams - source_grams
        ref_add = ref_grams - source_grams
        add_good = counter_intersection_size(sys_add, ref_add)
        add_p = safe_div(add_good, sum(sys_add.values()))
        add_r = safe_div(add_good, sum(ref_add.values()))
        add_score = f1(add_p, add_r)

        sys_keep = pred_grams & source_grams
        ref_keep = ref_grams & source_grams
        keep_good = counter_intersection_size(sys_keep, ref_keep)
        keep_p = safe_div(keep_good, sum(sys_keep.values()))
        keep_r = safe_div(keep_good, sum(ref_keep.values()))
        keep_score = f1(keep_p, keep_r)

        sys_delete = source_grams - pred_grams
        ref_delete = source_grams - ref_grams
        delete_good = counter_intersection_size(sys_delete, ref_delete)
        delete_score = safe_div(delete_good, sum(sys_delete.values()))

        scores.append((add_score + keep_score + delete_score) / 3)

    return 100 * mean(scores)


def source_from_20minuten_prompt(prompt: str) -> str:
    marker = "Paragraph:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def compute_task_metrics(
    task: dict[str, Any],
    examples: list[dict[str, str]],
    predictions: list[str],
) -> dict[str, Any]:
    metric = task["metric"]
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have the same length")

    cleaned_predictions = [clean_generation(prediction) for prediction in predictions]
    references = [example["answer"] for example in examples]

    if metric == "accuracy":
        answer_type = task.get("answer_type")
        correct = 0
        parsed_predictions: list[str | None] = []
        parsed_references: list[str | None] = []
        for prediction, reference in zip(cleaned_predictions, references):
            if answer_type == "number":
                pred_value = extract_number(prediction)
                ref_value = extract_number(reference)
                parsed_predictions.append(str(pred_value) if pred_value is not None else None)
                parsed_references.append(str(ref_value) if ref_value is not None else None)
                correct += int(pred_value is not None and ref_value is not None and pred_value == ref_value)
            else:
                choices = task.get("choices")
                pred_value = extract_choice(prediction, choices)
                ref_value = extract_choice(reference, choices)
                parsed_predictions.append(pred_value)
                parsed_references.append(ref_value)
                correct += int(pred_value is not None and pred_value == ref_value)
        accuracy = correct / len(examples) if examples else 0.0
        return {
            "metric": "accuracy",
            "primary_value": accuracy,
            "score": accuracy,
            "correct": correct,
            "num_examples": len(examples),
        }

    if metric == "rouge_l":
        values = [
            rouge_l_f1(prediction, reference)
            for prediction, reference in zip(cleaned_predictions, references)
        ]
        value = mean(values) if values else 0.0
        return {
            "metric": "rouge_l",
            "primary_value": value,
            "score": value,
            "num_examples": len(examples),
        }

    if metric == "edit_distance":
        distances = [
            edit_distance(normalize_text(prediction), normalize_text(reference))
            for prediction, reference in zip(cleaned_predictions, references)
        ]
        normalized = [
            distance / max(1, len(normalize_text(reference)))
            for distance, reference in zip(distances, references)
        ]
        mean_distance = mean(distances) if distances else 0.0
        mean_normalized = mean(normalized) if normalized else 0.0
        return {
            "metric": "edit_distance",
            "primary_value": mean_distance,
            "score": max(0.0, 1.0 - mean_normalized),
            "normalized_edit_distance": mean_normalized,
            "num_examples": len(examples),
        }

    if metric == "sari":
        values = [
            sari_sentence(
                source_from_20minuten_prompt(example["prompt"]),
                prediction,
                reference,
            )
            for example, prediction, reference in zip(
                examples, cleaned_predictions, references
            )
        ]
        value = mean(values) if values else 0.0
        return {
            "metric": "sari",
            "primary_value": value,
            "score": value / 100.0,
            "num_examples": len(examples),
        }

    raise ValueError(f"Unsupported metric: {metric}")
