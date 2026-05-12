# inter-task-relation

TRACE continual LoRA framework for Qwen3.

## Environment

```bash
source ~/envs/research/bin/activate
```

The default config uses:

- base model: `/data/huggingface_models/Qwen3-1.7B`
- data root: `/data/shengke/TRACE-Benchmark/LLM-CL-Benchmark_5000`
- task order: `C-STANCE -> FOMC -> MeetingBank -> Py150 -> ScienceQA -> NumGLUE-cm -> NumGLUE-ds -> 20Minuten`
- Qwen3 native chat template with `enable_thinking=False`

## Train

Training is sequential LoRA only. It does not run eval, does not call vLLM, and does not compute task metrics during training.

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.train_seq_lora \
  --config configs/trace8_qwen3_1p7b_seq_lora.yaml
```

Outputs are written under `/data/shengke/ITR/trace8_qwen3_1p7b_seq_lora/`, with one stage directory per task. The final adapter is:

```text
/data/shengke/ITR/trace8_qwen3_1p7b_seq_lora/task_08_20Minuten/final_adapter
```

Smoke-test training:

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.train_seq_lora \
  --config configs/trace8_qwen3_1p7b_seq_lora.yaml \
  --tasks C-STANCE,FOMC \
  --max-train-samples 8 \
  --max-steps 1 \
  --run-name smoke_seq_lora
```

## LAF-CL Adaptive Replay

Layer-adaptive continual LoRA training uses anchors, LoRA Fisher key layers, exact gradient signatures, a scored replay buffer, and masked gradient accumulation:

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.train_lafcl \
  --config configs/trace8_qwen3_1p7b_lafcl.yaml
```

Smoke-test LAF-CL on two tasks:

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.train_lafcl \
  --config configs/trace8_qwen3_1p7b_lafcl.yaml \
  --tasks C-STANCE,FOMC \
  --max-train-samples 8 \
  --max-steps 1 \
  --run-name smoke_lafcl
```

Each stage writes `anchors.jsonl`, `fisher.json`, `key_layers.json`, `anchor_grad.pt`, `scores.jsonl`, `train_metrics.json`, and `final_adapter`; the run root writes `buffer.jsonl`, `task_stats.pt`, and `final_summary.json`.

## Manual vLLM Eval

Run eval after training has finished:

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.eval_vllm \
  --config configs/trace8_qwen3_1p7b_seq_lora.yaml \
  --adapter /data/shengke/ITR/trace8_qwen3_1p7b_seq_lora/task_08_20Minuten/final_adapter
```

This evaluates every task on `test.json` and writes `predictions.jsonl`, `metrics.json`, `metrics_table.csv`, and `final_summary.json` under:

```text
/data/shengke/ITR/trace8_qwen3_1p7b_seq_lora/eval/task_08_20Minuten/
```

For a quick eval smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 python -m trace_cl.eval_vllm \
  --config configs/trace8_qwen3_1p7b_seq_lora.yaml \
  --adapter /data/shengke/ITR/smoke_seq_lora/task_02_FOMC/final_adapter \
  --max-samples 4 \
  --run-name smoke_seq_lora
```

## Metrics

- `C-STANCE`, `FOMC`, `ScienceQA`: accuracy
- `NumGLUE-cm`, `NumGLUE-ds`: numeric accuracy
- `MeetingBank`: ROUGE-L F1
- `Py150`: mean edit distance and normalized edit distance
- `20Minuten`: SARI

## Tests

```bash
pytest -q
```
