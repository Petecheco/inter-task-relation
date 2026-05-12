from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from trace_cl.anchors import select_anchors
from trace_cl.features import pool_hidden_states
from trace_cl.fisher import compute_layerwise_fisher, select_top_layers
from trace_cl.lora_utils import (
    add_current_grads_to_accumulator,
    build_layer_to_lora_params,
    init_lora_grad_accumulator,
    parse_layer_id_from_param_name,
    zero_grad_except_layers,
)
from trace_cl.replay_buffer import ReplayBuffer, TrainExample
from trace_cl.scoring import (
    compute_raw_shared_score,
    cosine_similarity,
    rank_normalize_scores,
)


class LoraBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.lora_B = nn.Linear(2, 2, bias=False)


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = LoraBlock()


class FakeLoraModel(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Module()
        self.base_model.model.model = nn.Module()
        self.base_model.model.model.layers = nn.ModuleList(
            FakeLayer() for _ in range(num_layers)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        del attention_mask, labels, use_cache
        scale = input_ids.float().mean()
        total = sum(
            parameter.sum() * scale
            for name, parameter in self.named_parameters()
            if "lora_" in name
        )
        return SimpleNamespace(loss=total.pow(2))


def test_layer_mapping_and_gradient_mask():
    assert (
        parse_layer_id_from_param_name(
            "base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weight"
        )
        == 12
    )
    assert parse_layer_id_from_param_name("base_model.model.lm_head.weight") is None

    model = FakeLoraModel(num_layers=2)
    grouped = build_layer_to_lora_params(model)
    assert sorted(grouped) == [0, 1]
    assert all(len(params) == 2 for params in grouped.values())

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    zero_grad_except_layers(model, {1})
    for name, parameter in model.named_parameters():
        if ".layers.0." in name and "lora_" in name:
            assert torch.count_nonzero(parameter.grad) == 0
        if ".layers.1." in name and "lora_" in name:
            assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_gradient_accumulator_respects_allowed_layers():
    model = FakeLoraModel(num_layers=2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    accumulator = init_lora_grad_accumulator(model)
    add_current_grads_to_accumulator(model, accumulator, allowed_layers={1})

    for name, grad in accumulator.items():
        if ".layers.0." in name:
            assert torch.count_nonzero(grad) == 0
        if ".layers.1." in name:
            assert torch.count_nonzero(grad) == grad.numel()


def test_pool_hidden_states_mean_and_last_token():
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.allclose(
        pool_hidden_states(hidden, mask, "mean"),
        torch.tensor([[2.0, 2.0], [6.0, 8.0]]),
    )
    assert torch.allclose(
        pool_hidden_states(hidden, mask, "last_token"),
        torch.tensor([[3.0, 3.0], [10.0, 12.0]]),
    )


def test_select_anchors_nearest_points_per_cluster():
    examples = [{"uid": str(index)} for index in range(6)]
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.2, 10.0],
            [30.0, 30.0],
            [30.2, 30.0],
        ]
    )
    anchors = select_anchors(
        examples,
        features,
        num_clusters=3,
        anchors_per_cluster=1,
        seed=7,
    )
    assert len(anchors) == 3
    assert len({anchor["uid"] for anchor in anchors}) == 3


def test_fisher_returns_nonnegative_scores_and_top_layers():
    model = FakeLoraModel(num_layers=3)
    loader = DataLoader(
        [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([1, 2])}],
        batch_size=1,
    )
    fisher = compute_layerwise_fisher(
        model,
        loader,
        build_layer_to_lora_params(model),
        normalize_by_numel=True,
    )
    assert sorted(fisher) == [0, 1, 2]
    assert all(value >= 0 for value in fisher.values())
    assert len(select_top_layers(fisher, 2)) == 2


def test_score_math_and_rank_normalization():
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(torch.zeros(2), torch.ones(2)) == pytest.approx(0.0)

    raw_t1 = compute_raw_shared_score(
        torch.tensor([-1.0, 0.0]),
        1,
        {1: torch.tensor([1.0, 0.0])},
    )
    assert raw_t1 == pytest.approx(-1.0)

    raw = compute_raw_shared_score(
        torch.tensor([1.0, 0.0]),
        1,
        {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([-1.0, 0.0]),
        },
    )
    assert raw == pytest.approx(0.0)

    normalized = rank_normalize_scores([0.1, 0.9, 0.3])
    assert all(0.0 <= score <= 1.0 for score in normalized)
    assert normalized[1] == max(normalized)
    assert normalized[0] == min(normalized)


def make_example(task_id: int, index: int, score: float, split: str) -> TrainExample:
    return TrainExample(
        uid=f"{task_id}:train:{index}",
        prompt=f"p{index}",
        answer=f"a{index}",
        task_id=task_id,
        task_name=f"T{task_id}",
        sample_index=index,
        score=score,
        raw_score=score,
        split=split,
    )


def test_replay_buffer_capacity_balance_and_score_preferences():
    buffer = ReplayBuffer(capacity=4, share_ratio=0.5, per_task_balance=True)
    buffer.update_with_task_examples(
        1,
        [
            make_example(1, 0, 0.95, "share"),
            make_example(1, 1, 0.80, "share"),
            make_example(1, 2, 0.05, "specific"),
            make_example(1, 3, 0.20, "specific"),
        ],
    )
    buffer.update_with_task_examples(
        2,
        [
            make_example(2, 0, 0.99, "share"),
            make_example(2, 1, 0.70, "share"),
            make_example(2, 2, 0.01, "specific"),
            make_example(2, 3, 0.30, "specific"),
        ],
    )

    assert len(buffer) == 4
    assert buffer.task_distribution() == {1: 2, 2: 2}
    kept = {example.uid for example in buffer.examples}
    assert "1:train:0" in kept
    assert "1:train:2" in kept
    assert "2:train:0" in kept
    assert "2:train:2" in kept
