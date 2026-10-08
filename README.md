# Math Tutor

Math Tutor is a simple agentic application that turns a math prompt into an animated lesson with optional narration synchronized to the video.


## Demo

[![Watch the Math Tutor demo](https://cdn.loom.com/sessions/thumbnails/9a158cbb6ad14d0da5550907970b0aab-1ed971aff2086393.gif)](https://www.loom.com/share/9a158cbb6ad14d0da5550907970b0aab)

## How it works

Math Tutor takes a question and turns it into an animated explanation. The validator
records a 30–45 second target as an advisory measurement, so otherwise valid lessons
are not rejected solely because their rendered duration falls outside that range.

The adapter-based system uses a **Qwen3-4B** backbone fine-tuned with LoRA on
[Bespoke-Manim](https://huggingface.co/datasets/bespokelabs/bespoke-manim). This
synthetic dataset contains 1,000 educational animation examples, pairing questions
with narration, visual descriptions, and Python code for [Manim](https://www.manim.community/).

The initial setup trained three stand-alone adapters: **foundational**, **intermediate**, and
**advanced** based on the question difficulty which was inferred from the dataset. The API endpoint accepts an explicit difficulty label for _oracle routing_.

The flow looks like this, __selected adapter__ produces a draft &rarr; __cleanup__ step turns that draft
into the scene format expected by the renderer &rarr; generated code is validated for compilation correctness &rarr; renders it the app with optional synchronized narration and captions.

![Math Tutor routes each request to a foundational, intermediate, or advanced LoRA adapter sharing one backbone, then cleans up, validates, and renders the generated scene.](assets/math-tutor-system.drawio.svg)


## Setup

### Prerequisites

- An existing Modal inference endpoint and its authentication token
- Optional: an ElevenLabs API key for narration and captions

### Configuration

```bash
git clone https://github.com/splion-360/math-tutor.git
cd math-tutor
make setup
```

Edit `backend/.env` to configure lesson generation:

| Variable | Used for |
| --- | --- |
| `MODAL_VLLM_BASE_URL` | An existing Modal inference endpoint, including `/v1`; selects the three-adapter serving path |
| `MODAL_VLLM_API_KEY` | Authentication with that endpoint |
| `ELEVENLABS_API_KEY` | Optional narration and captions |

### Run

```bash
docker compose up --build
```

Compose prepares the rendering images and starts the backend and frontend.
Python, Node.js, and FFmpeg are included in the containers.

Stop the app with `docker compose down`. Generated files are saved in
`backend/artifacts/` and excluded from Git.

## Reproduce the experiment figures

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run:

```bash
make evidence-data
make evidence-figures
```

These CPU commands download the pinned dataset and public evidence, check their
hashes, reconstruct the split IDs, and generate the plots in
`training/artifacts/evidence/figures/`. No Modal credentials are required.
The [evidence release](https://github.com/splion-360/math-tutor/releases/tag/evidence-v1)
includes the numeric summaries and shared adapter checkpoint. The
[manifest](training/evidence/manifest.json) records their provenance and missing
evidence: original split metadata and completion of the third training epoch
could not be verified.

The [paired pilot release](https://github.com/splion-360/math-tutor/releases/tag/paired-pilot-v2)
contains the saved responses, render results, videos, and review worksheet.
Human review of mathematical correctness and prompt adherence is pending.

To run another paired lesson pilot, first run `make evidence-data`, then:

```bash
export EVALUATION_RUN=paired-pilot-$(date -u +%Y%m%dT%H%M%SZ)
make evaluation-freeze
make evaluation-generate  # Requires Modal authentication and uses GPU credits
make evaluation-download
make evaluation-render   # Requires Docker
```

Freeze the plan once and reuse it for additional run IDs. It records prompts,
overlap checks, token measurements, and renderer versions. Outputs include raw
responses, render diagnostics, and a human-review
CSV in `training/artifacts/$EVALUATION_RUN/evaluation/`.
Watch the rendered videos and fill in the reviewer, UTC timestamp, duration,
judgments, and evidence notes in `human_review_bound.csv`. Run
`make evaluation-review` to create a separate reviewed report; the automatic
results are preserved.

## Project structure

```text
math-tutor/
├── frontend/           # Lesson interface and video player
├── backend/            # Lesson generation, rendering, and evaluation
├── training/           # LoRA training, gradient probes, and configurations
├── deployments/modal/  # Remote training and serving
├── compose.yaml        # Local application services
└── Makefile            # Setup and development commands
```
