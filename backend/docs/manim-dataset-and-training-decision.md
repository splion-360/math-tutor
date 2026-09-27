# Manim dataset and training-input decision

This Runme notebook records the data used by the custom Qwen3-4B LoRA experiments and why the training input changed. Run cells from this repository; no cell starts a training job or reads credentials. The token-length cell needs the frozen Qwen tokenizer cached locally and the training virtual environment with `transformers` and `jinja2` installed.

## Repository identity

The commands below run from the repository root and report the checkout being inspected. They do not load `.env` files.

```sh {"name":"dataset_repo_identity","interpreter":"bash","skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

## Dataset inventory

**Observed:** [`bespoke_manim_train.jsonl`](../../training/data/bespoke_manim_train.jsonl) has 995 records, 23 subjects, and 426 distinct topic names. Each record has an ID, topic, question (`prompt`), reference Manim Python (`manim_code`), source, and metadata. The separate project-authored [evaluation slice](../data/evaluation/manim_eval_v1.jsonl) has 15 prompt-only examples, five in each difficulty band.

The prepared corpus lives in ignored `training/data/`, so a fresh checkout needs a local copy before these cells can run. The preparation script records source revision `4542ab8b32483c30d1772946dacae2ad1ae9274c` of `bespokelabs/bespoke-manim`.

```sh {"name":"dataset_inventory","interpreter":"bash","skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
jq -s '{rows:length, subjects:([.[].subject]|unique|length), topics:([.[].topic]|unique|length), difficulty:(group_by(.difficulty)|map({label:.[0].difficulty,rows:length})), source_names:([.[].source.name]|unique)}' training/data/bespoke_manim_train.jsonl
jq -s '{rows:length, difficulty:(group_by(.difficulty)|map({label:.[0].difficulty,rows:length}))}' backend/data/evaluation/manim_eval_v1.jsonl
```

The [preparation script](../../training/scripts/prepare_bespoke_manim.py) reads the source Parquet file, keeps rows with non-empty `python_code` and no recorded `error`, and strips surrounding code whitespace. It does not independently render, syntax-check, or deduplicate the scripts. Its difficulty label is a **code-length proxy**, not a mathematical-difficulty judgment: at the current tertile boundaries, code of at most 93 lines is `foundational`, 94–119 lines is `intermediate`, and longer code is `advanced`.

## What Qwen sees

**Decision:** The model input is a system instruction followed by `Topic: ...` and `Task: ...`. The assistant target is the reference Manim Python. The `difficulty` field remains in the record for stratified splits and analysis, but is not in the model prompt. A label derived from the target code's length must not be supplied as though it were known from the user's question.

During training, the whole chat-formatted example is fed to the model. In the placement experiment, loss is calculated only on assistant/code tokens; prompt labels are masked. The earlier three fixed specialists and previous gradient/placement runs used a different training contract, so their results cannot be treated as measurements of this label-free setup. See [data formatting](../../training/src/dynamic_lora/data.py) and the [placement trainer](../../training/src/dynamic_lora/placement_run.py).

## Complete-example token lengths

**Measured on 2026-09-26** with `Qwen/Qwen3-4B-Instruct-2507` tokenizer revision `1b4199c4f36b0cef378bfb12390c18780c18af4c`, all 995 current records, and the label-free chat template: median 1,049 tokens; p90 1,489; p95 1,647; p99 2,028; maximum 2,963. Exactly 533 records exceed 1,024 tokens, six exceed 2,048, and none exceed 3,072. These are complete prompt-plus-target lengths, not output-token limits or GPU-memory measurements.

This cell is read-only and uses a locally cached tokenizer; it does not download model weights. It is excluded from Run All because it requires the optional training dependencies and tokenizer cache.

```sh {"name":"measure_complete_qwen_tokens","interpreter":"bash","excludeFromRunAll":true,"skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
PYTHONPATH=training/src HF_HUB_OFFLINE=1 training/.venv/bin/python - <<'PY'
import json
from pathlib import Path

from transformers import AutoTokenizer

from dynamic_lora.constants import FROZEN_MODEL_ID, FROZEN_MODEL_REVISION
from dynamic_lora.data import tokenize_completion_record

records = [
    json.loads(line)
    for line in Path("training/data/bespoke_manim_train.jsonl").read_text(encoding="utf-8").splitlines()
    if line
]
tokenizer = AutoTokenizer.from_pretrained(
    FROZEN_MODEL_ID, revision=FROZEN_MODEL_REVISION, local_files_only=True
)
lengths = sorted(
    len(tokenize_completion_record(record, tokenizer, max_seq_length=3072)["input_ids"])
    for record in records
)
print(json.dumps({
    "rows": len(lengths),
    "p50": lengths[len(lengths) // 2],
    "p90": lengths[int(len(lengths) * 0.90)],
    "p95": lengths[int(len(lengths) * 0.95)],
    "p99": lengths[int(len(lengths) * 0.99)],
    "max": lengths[-1],
    "above_1024": sum(length > 1024 for length in lengths),
    "above_2048": sum(length > 2048 for length in lengths),
    "above_3072": sum(length > 3072 for length in lengths),
}, sort_keys=True))
PY
```

**Decision:** Full-dataset configs use a 3,072-token safety ceiling. The tokenizer never silently cuts a target; an overlong example fails before model weights load. The ceiling leaves 109 tokens above the current maximum. It is not a claim that 3,072 is Qwen's context limit, nor proof that the longest example fits GPU memory during backward propagation. A changed dataset requires measuring lengths again before training.

## Experiment boundary

The 995 rows are the source pool. The placement comparison reserves 100 stratified rows for validation and excludes the 16 gradient-probe examples from that validation set; the remaining 895 are eligible for training. The prior 64-step, batch-one pilot did not complete an epoch. The experiment sequence is a label-free one-step shared-LoRA probe, a no-update base-weight gradient probe, then matched placement arms. Each new run writes to a distinct `label_free` artifact path so historical metrics remain identifiable. Render success is not established by training loss alone.

## One-step Modal probe observation

**Measured on 2026-09-26:** The first attempt ran out of memory during the first 1,480-token gradient backward pass on an L4 (22.03 GiB total); it produced no ranking or adapter. The retry at source revision `2cceb7d71af95b3a9b3698ea756ca88258a4b3c1` completed on an A100 80 GB. It measured 16 example gradients, performed one optimizer step, and saved `adapter_model.safetensors` and `run_metadata.json` to `modal-volume://dream-ai-training-artifacts/shared_lora_probe_16_label_free`. The persisted files were verified with a read-only Modal volume listing.

The raw-energy top four were `layer_6.down_proj`, `layer_1.up_proj`, `layer_7.up_proj`, and `layer_3.k_proj`. The single training step reported loss 1.0048. These are probe observations, **not** evidence that those placements improve held-out Manim code or that dynamic splitting is warranted. The base-weight and placement comparisons are separate experiments.

## No-update base-weight probe

**Measured on 2026-09-26:** A BF16 Qwen3-4B probe on an A100 40 GB computed base-weight gradients for the same 16 deterministic example indices and made zero optimizer updates. Its top four eligible modules were `layer_6.down_proj`, `layer_34.q_proj`, `layer_32.q_proj`, and `layer_22.v_proj`. All four appeared in the top four of each of four disjoint, four-example stability subsets. A fresh LoRA attached to this BF16 model and scored by the same squared-mean-gradient method selected the same four modules, in a different order. The saved result is `modal-volume://dream-ai-training-artifacts/base_weight_probe_16_label_free/run_metadata.json` at source revision `f1f4598b480b12895bb6b8663aa764b0e6ac6dcf`.

Only `layer_6.down_proj` overlaps the earlier LoRA probe's top four. That earlier probe averaged **per-example squared gradient norms** on a four-bit model; this base probe scores the **squared norm of the mean gradient** on a BF16 model. The 1-of-4 overlap cannot be attributed solely to attaching LoRA. The matched BF16 comparison isolates the gradient-source change more cleanly, but neither probe establishes downstream lesson quality.

**Decision:** Use the no-update base probe's four modules as the discovered arm in the matched placement experiment. The low-energy and two random controls keep the same one-`down_proj`, two-`q_proj`, one-`v_proj` module mix, LoRA rank, training steps, and 895/100 train/validation split. Compare held-out code loss before claiming the placement helped; render success still requires a separate check. Probe measurements are preserved in Modal metadata; old W&B diagnostic runs are removed so the W&B project can focus on the four placement arms.

## Matched placement result

**Measured on 2026-09-26:** Four label-free placement arms each trained for 64 steps on the same 895 training records and were evaluated on the same 100 held-out records. Each adapted four modules with 466,944 trainable parameters. Saved metadata confirmed the same data hashes, split IDs, seed, and source revision `5797dabb6eb25199f852a9cf5080ee9d83422183`; all four adapter weight files persisted in `modal-volume://dream-ai-training-artifacts/placement_ablation_64_label_free`.

| Arm | Held-out code loss ↓ | W&B |
| --- | ---: | --- |
| Discovered | 0.6199 | [run](https://wandb.ai/splion/math-tutor-dynamic-lora/runs/bnf5fndx) |
| Low energy | 0.8106 | [run](https://wandb.ai/splion/math-tutor-dynamic-lora/runs/jtzhjknb) |
| Random 1 | 0.6400 | [run](https://wandb.ai/splion/math-tutor-dynamic-lora/runs/etyzuiiu) |
| Random 2 | 0.6806 | [run](https://wandb.ai/splion/math-tutor-dynamic-lora/runs/rktuk6k8) |

The discovered arm had the lowest loss in this single-seed comparison, but the margin over Random 1 is only 0.0201. Each arm saw 64 of 895 eligible training examples (about 7% of an epoch). This result supports further testing of the placement; it does not establish statistical reliability, runnable Manim output, lesson quality, or a need for dynamic adapter splitting. The W&B project was verified to contain exactly these four finished placement runs after removal of 21 older diagnostic runs; that W&B deletion is not recoverable there, while their saved Modal artifacts were retained.
