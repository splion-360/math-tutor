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

## Verify the saved artifact

This cell reads the Modal volume and creates only a temporary local download. It is excluded from Run All because it requires an authenticated Modal CLI and network access. It does not start training or create W&B runs.

```sh {"name":"inspect_gradient_signatures","interpreter":"bash","excludeFromRunAll":true,"skipPrompts":true}
cd "$(git rev-parse --show-toplevel)"
signature_analysis_dir=$(mktemp -d /tmp/math-signatures.XXXXXX)
modal volume get --profile splion-360 dream-ai-training-artifacts shared_lora_signatures_64_label_free/adapter/gradient_signatures/gradient_signatures.jsonl "$signature_analysis_dir/gradient_signatures.jsonl"
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
