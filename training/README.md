# Shared LoRA Training Baseline

This package defines the static shared-LoRA control for the math tutor research track.
It is intentionally outside `backend/` so Transformers, PEFT, Accelerate, and optional
QLoRA dependencies do not become FastAPI runtime dependencies.

The frozen backbone is:

```text
Qwen/Qwen3-4B-Instruct-2507
```

No dynamic adapter spawning, learned routing, or specialist assignment is implemented here.
This baseline trains one shared LoRA adapter across foundational, intermediate, and advanced
Manim-code tasks, then records metadata for later zero-shot/shared/dynamic comparison.

## Data Contract

Training data is JSONL. Each record must include:

- `id`: stable lowercase identifier matching `[a-z0-9][a-z0-9._-]*`
- `difficulty`: exactly `foundational`, `intermediate`, or `advanced`
- `topic`: non-empty string
- `prompt`: non-empty user task
- `manim_code`: non-empty target code
- `split`: exactly `train`
- `source`: object with non-empty `name` and `reference`

`../backend/data/evaluation/manim_eval_v1.jsonl` is the holdout. Training IDs must be disjoint
from that file. The tiny fixture in `fixtures/tiny_train.jsonl` is for dry-runs and tests only;
it is not a credible training corpus and must not be reported as model performance evidence.

## CPU Dry Run

From the repository root:

```bash
PYTHONPATH=training/src backend/.venv/bin/python -m shared_lora_baseline.cli dry-run \
  --config training/configs/shared_lora_qwen3_4b.json
```

The dry run validates schema and holdout disjointness, prints a deterministic redacted plan,
and writes machine-readable metadata to `training/artifacts/shared_lora_qwen3_4b/run_metadata.json`.
It performs no heavyweight imports and downloads no weights.

## GPU Training Command

The checked-in `shared_lora_qwen3_4b.json` is a one-step smoke configuration backed by three
fixture records. Do not spend GPU credits using it as a real experiment. First create a separate
config that points at the curated training JSONL and records the chosen step budget. Then run from
the repository root:

```bash
uv sync --project training --extra train --python 3.12
test -n "${TRAIN_CONFIG:-}" || { echo "Set TRAIN_CONFIG to the curated run config"; exit 2; }
PYTHONPATH=training/src uv run --project training python -m shared_lora_baseline.cli train \
  --config "$TRAIN_CONFIG"
```

Expected artifacts:

- `training/artifacts/shared_lora_qwen3_4b/adapter/adapter_config.json`
- `training/artifacts/shared_lora_qwen3_4b/adapter/adapter_model.safetensors`
- `training/artifacts/shared_lora_qwen3_4b/run_metadata.json`
- `training/artifacts/shared_lora_qwen3_4b/adapter/checkpoint-*/trainer_state.json`

## Experiment Tracking

Training runs use W&B when `WANDB_API_KEY` is present and `tracking.mode` is `auto` or `online`.
The default project is `math-tutor-dynamic-lora`. The run config records the frozen model, dataset
hashes, LoRA settings, git revision, output directory, and Modal artifact path. Hugging Face trainer
metrics are reported to the same W&B run with grouped names such as `train/loss` and
`train/runtime_seconds`.

Set `tracking.mode` to `disabled` for local runs that should never contact W&B. Set it to `online`
when missing W&B setup should fail fast instead of silently falling back to local metadata only.

## Assumptions

- The target GPU can load Qwen/Qwen3-4B-Instruct-2507 with 4-bit quantization.
- The model revision is frozen to Hugging Face commit
  `1b4199c4f36b0cef378bfb12390c18780c18af4c`.
- The real training JSONL will replace the tiny fixture and remain holdout-disjoint.
- Evaluation is run separately against the preserved backend holdout.
- Render success, parse rate, and Manim API validity are evaluation metrics, not training metrics.

## Current Blocker

Actual launch is blocked on a GPU machine with access to the frozen Qwen checkpoint and the real
curated training JSONL. This worktree intentionally does not provision cloud resources, call paid
APIs, download model weights, or run training.

## Inference Integration Contract

The inference owner should treat the saved adapter directory as a static PEFT adapter for the frozen
base model. The adapter metadata identifies:

- `condition`: `shared_lora_static_control`
- `model_id`: `Qwen/Qwen3-4B-Instruct-2507`
- `model_revision`: `1b4199c4f36b0cef378bfb12390c18780c18af4c`
- `adapter_id`: `shared-lora-qwen3-4b-manim-v1`
- `adapter_kind`: `static_shared_lora`
- `dynamic_adapter_spawning`: `false`
- `learned_router`: `false`

Serving should load this adapter as one named option and report that adapter ID in response metadata
when selected. No router confidence or specialist ID is produced by this baseline.
