# Math Tutor

Math Tutor is a simple agentic application that turns a math prompt into an animated lesson with optional narration synchronized to the video.


## Demo

[![Watch the Math Tutor demo](https://cdn.loom.com/sessions/thumbnails/9a158cbb6ad14d0da5550907970b0aab-1ed971aff2086393.gif)](https://www.loom.com/share/9a158cbb6ad14d0da5550907970b0aab)

## How it works

Math Tutor takes a question and turns it into an animated explanation. The intended
lesson length is __30–45__ seconds.

The adapter-based system uses a **Qwen3-4B** backbone fine-tuned with LoRA on
[Bespoke-Manim](https://huggingface.co/datasets/bespokelabs/bespoke-manim). This
synthetic dataset contains 1,000 educational animation examples, pairing questions
with narration, visual descriptions, and Python code for [Manim](https://www.manim.community/).

The initial setup trained three stand-alone adapters: **foundational**, **intermediate**, and
**advanced** based on the question difficulty which was inferred from the dataset. The API endpoint accepts an explicit difficulty label for _oracle routing_.

The flow looks like this, __selected adapter__ produces a draft $\rarr$ __cleanup__ step turns that draft
into the scene format expected by the renderer $\rarr$ generated code is validated for compilation correctness $\rarr$ renders it the app with optional synchronized narration and captions.

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

## Reproduce the dataset

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run:

```bash
make evidence-data
```

This downloads the pinned dataset, checks the original corpus hash, and writes
the reconstructed split IDs to `training/artifacts/evidence/split.json`.
The [evidence inventory](training/evidence/manifest.json) identifies the numeric
summaries and checkpoint metadata that still need to be recovered before the
experiment figures can be regenerated publicly.

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
