# Gradient signatures for issue #21

This Runme notebook records the first label-free signature capture on Qwen3-4B-Instruct-2507. The [Modal run](https://modal.com/apps/splion-360/main/ap-EIpm2UMqwCpMyD2Fe1x61s) completed on 2026-09-27 from source revision `495a8ccd1177a59cfd2b48c68f6aac74ff9f4104`. It created no W&B run; the four placement-arm runs remain the only intended W&B records.

## Setup and capture

The run trained one shared LoRA for 64 optimizer steps on the 995-row label-free Manim corpus, with batch size one, seed 42, a 3,072-token complete-example ceiling, and a four-bit base model. LoRA was attached to all seven projection types in every transformer layer. Signatures were observed only at the four module keys selected by the earlier **no-update base-weight probe**: `layer_6.down_proj`, `layer_34.q_proj`, `layer_32.q_proj`, and `layer_22.v_proj`. The saved probe's model revision, dataset hashes, seed, and token ceiling were checked before loading model weights.

For each selected module, the pre-optimizer LoRA gradients were projected into 256 dimensions with a deterministic signed-bucket mapping. Capture began at step 17 and continued every step through 64: 48 vectors per module, 192 total. The [run metadata](modal-volume://dream-ai-training-artifacts/shared_lora_signatures_64_label_free/run_metadata.json), [manifest](modal-volume://dream-ai-training-artifacts/shared_lora_signatures_64_label_free/adapter/gradient_signatures/gradient_signatures_manifest.json), and [raw JSONL snapshots](modal-volume://dream-ai-training-artifacts/shared_lora_signatures_64_label_free/adapter/gradient_signatures/gradient_signatures.jsonl) are saved on the Modal volume. The raw file was downloaded and its 192 rows and 256 values per signature were checked independently.

## Observed directions

Cosine compares each signature with the preceding optimizer step for the same module; a negative cosine is counted as a conflict. Each module has 47 such comparisons. The all-pairs column compares every distinct pair of its 48 captured vectors, not just adjacent steps.

| Module | Adjacent negative | Adjacent mean cosine | All-pairs negative |
| --- | ---: | ---: | ---: |
| `layer_6.down_proj` | 16/47 (34.0%) | +0.0705 | 46.5% |
| `layer_34.q_proj` | 24/47 (51.1%) | +0.0103 | 48.5% |
| `layer_32.q_proj` | 22/47 (46.8%) | +0.0388 | 51.3% |
| `layer_22.v_proj` | 18/47 (38.3%) | +0.0198 | 47.7% |

All 192 vectors have nonzero norm. The 64-step run reported training loss 0.5717. That is **not** comparable to the four-arm held-out code losses: this is training loss from the shared-LoRA trainer, while the placement arms used a separate validation split and completion-only loss.

**Interpretation:** Reversals occur, but mean cosines are near zero and roughly half of all vector pairs have negative cosine. This does not yet show two persistent gradient groups or establish that any module should split. The snapshots do not contain a reliable training-example ID, so they cannot be assigned a mathematical topic or difficulty after the fact. Issue #22 can use these artifacts to design and test a split gate, but a gate must distinguish stable separation from ordinary minibatch variation; a negative adjacent cosine alone is not enough.

## Fixed-prompt extension

**Measured on 2026-09-27:** The [extension run](https://modal.com/apps/splion-360/main/ap-FphfHYqalOIy8adhCH7Yah) at source revision `14a62c000f593c2f18cbabbd672a73603f8df085` reserved 64 examples by a seed-and-record-ID hash and excluded them from optimizer training. The training pool contained the remaining 931 records, and the run made 64 optimizer steps; it did not complete an epoch. The same 64 reserved examples were probed at steps 0, 32, and 64 in evaluation mode without optimizer updates from those probe passes. Prompt IDs were saved with projected gradients but were not included in model inputs. W&B tracking remained disabled.

The [raw identified signatures](modal-volume://dream-ai-training-artifacts/shared_lora_fixed_prompt_signatures_64_label_free/adapter/fixed_prompt_signatures/fixed_prompt_signatures.jsonl) and [manifest](modal-volume://dream-ai-training-artifacts/shared_lora_fixed_prompt_signatures_64_label_free/adapter/fixed_prompt_signatures/fixed_prompt_signatures_manifest.json) are in a new Modal artifact directory; the earlier run was not overwritten. The downloaded JSONL was checked: all 64 IDs × 3 checkpoints × 4 modules are present exactly once (768 vectors), each with 256 values and nonzero norm. Saved metadata confirms the 931/64 train/probe split and W&B-disabled state.

For steps 32 and 64, the table compares each prompt with itself across checkpoints and with the other 63 prompts. “Top-1 ID” asks whether the step-64 vector from the same prompt is the nearest of 64 candidates to its step-32 vector. This is an analysis of gradient fingerprints, **not** a model-inference accuracy metric.

| Module | Same-prompt mean cosine | Other-prompt mean cosine | Top-1 ID |
| --- | ---: | ---: | ---: |
| `layer_6.down_proj` | 0.8411 | 0.0005 | 63/64 |
| `layer_34.q_proj` | 0.9152 | 0.0148 | 64/64 |
| `layer_32.q_proj` | 0.9134 | −0.0066 | 64/64 |
| `layer_22.v_proj` | 0.9168 | 0.0082 | 64/64 |

**Interpretation:** Once the shared adapter has had 32 updates, a held-out prompt's gradient direction is highly repeatable 32 steps later and distinguishable from gradients of other prompts. This is useful evidence that the vectors retain prompt-specific information. It does **not** show two stable groups or justify a split. Step-0-to-32 similarity is much weaker, especially for the query projections; LoRA parameter initialization and early training change the gradient representation, so that transition should not be treated as a failed persistent cluster. The next decision belongs to #22, using held-out clustering and a one-centroid comparison rather than identity retrieval alone.

The following cell repeats the coverage and step-32-to-64 comparison. It only reads a Modal artifact and creates a temporary local download; it is excluded from Run All because authenticated Modal access is needed.

```sh {"name":"inspect_fixed_prompt_signatures","interpreter":"bash","excludeFromRunAll":true,"skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
fixed_signature_dir=$(mktemp -d /tmp/math-fixed-signatures.XXXXXX)
modal_cli=$(command -v modal || true)
if [ -z "$modal_cli" ]; then modal_cli="$HOME/.local/bin/modal"; fi
"$modal_cli" volume get --profile splion-360 dream-ai-training-artifacts shared_lora_fixed_prompt_signatures_64_label_free/adapter/fixed_prompt_signatures/fixed_prompt_signatures.jsonl "$fixed_signature_dir/signatures.jsonl"
training/.venv/bin/python - "$fixed_signature_dir/signatures.jsonl" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
ids = sorted({row["record_id"] for row in rows})
modules = sorted({row["layer"] for row in rows})
steps = sorted({row["step"] for row in rows})
vectors = {
    (row["layer"], row["record_id"], row["step"]): np.asarray(row["signature"], dtype=float)
    for row in rows
}
print(json.dumps({
    "rows": len(rows), "unique_keys": len(vectors), "ids": len(ids),
    "modules": len(modules), "steps": steps,
    "complete": len(vectors) == len(ids) * len(modules) * len(steps),
}, sort_keys=True))
for module in modules:
    def normalized(step):
        matrix = np.stack([vectors[(module, record_id, step)] for record_id in ids])
        return matrix / np.linalg.norm(matrix, axis=1)[:, None]
    similarity = normalized(32) @ normalized(64).T
    same = np.diag(similarity)
    other = similarity[~np.eye(len(ids), dtype=bool)]
    print(json.dumps({
        "module": module,
        "same_mean": round(float(np.mean(same)), 4),
        "other_mean": round(float(np.mean(other)), 4),
        "top1_id": int(np.count_nonzero(np.argmax(similarity, axis=1) == np.arange(len(ids)))),
    }, sort_keys=True))
PY
```

## Verify the saved artifact

This cell reads the Modal volume and creates only a temporary local download. It is excluded from Run All because it requires an authenticated Modal CLI and network access. It does not start training or create W&B runs.

```sh {"name":"inspect_gradient_signatures","interpreter":"bash","excludeFromRunAll":true,"skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
signature_analysis_dir=$(mktemp -d /tmp/math-signatures.XXXXXX)
modal_cli=$(command -v modal || true)
if [ -z "$modal_cli" ]; then modal_cli="$HOME/.local/bin/modal"; fi
"$modal_cli" volume get --profile splion-360 dream-ai-training-artifacts shared_lora_signatures_64_label_free/adapter/gradient_signatures/gradient_signatures.jsonl "$signature_analysis_dir/gradient_signatures.jsonl"
wc -l "$signature_analysis_dir/gradient_signatures.jsonl"
PYTHONPATH=training/src training/.venv/bin/python - "$signature_analysis_dir/gradient_signatures.jsonl" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

by_module = defaultdict(list)
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    by_module[row["layer"]].append(row)
for module, rows in sorted(by_module.items()):
    vectors = np.asarray([row["signature"] for row in rows], dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / norms[:, None]
    all_pairs = (unit @ unit.T)[np.triu_indices(len(rows), k=1)]
    adjacent = np.asarray([row["cosine_to_previous"] for row in rows[1:]], dtype=float)
    print(json.dumps({
        "module": module,
        "observations": len(rows),
        "dimension": vectors.shape[1],
        "zero_norms": int(np.count_nonzero(norms == 0)),
        "adjacent_conflicts": int(np.count_nonzero(adjacent < 0)),
        "adjacent_mean_cosine": round(float(np.mean(adjacent)), 4),
        "all_pairs_negative_fraction": round(float(np.mean(all_pairs < 0)), 4),
    }, sort_keys=True))
PY
```
