# Math Tutor

**Ask a math question. Get a visual lesson made for that question.**

Math is often easier to understand when you can watch an idea unfold. Math Tutor turns a typed question into a short animation with an explanation, narration, and captions.

It is designed for more than elementary math. The same product can explain geometry, calculus, differential equations, Fourier transforms, and other advanced topics.

## Quick start

### Requirements

- Git
- Docker Desktop or another Docker installation with Compose
- At least 1 GB of memory available for each Manim render

The full local app runs through Docker Compose. Python, Node, and the application dependencies are installed inside the development containers.

### 1. Get the repository

```bash
git clone https://github.com/splion-360/math-tutor.git
cd math-tutor
```

### 2. Configure local credentials

Create the ignored backend environment file:

```bash
cp backend/.env.example backend/.env
```

Add only the integrations you want to use:

| Setting | Purpose |
| --- | --- |
| `MODAL_VLLM_BASE_URL` | URL of the hosted Qwen and LoRA endpoint |
| `MODAL_VLLM_API_KEY` | Credentials for the protected Modal endpoint |
| `NEBIUS_API_KEY` | Optional alternative model provider |
| `ELEVENLABS_API_KEY` | Optional synchronized voice and captions |

The website and API can start without these values, but generated lessons need a configured model provider. Without ElevenLabs, lessons use the silent-video path.

Training credentials and experiment tracking are covered separately in the [training guide](training/README.md).

### 3. Build the narration renderer

This step is only needed when `ELEVENLABS_API_KEY` is configured:

```bash
docker compose --profile renderer-build build manim-voiceover
```

### 4. Start the app

Run this from the repository root:

```bash
docker compose up --build
```

Open:

- Math Tutor: [http://localhost:5173](http://localhost:5173)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

The frontend waits for the backend health check before starting. Generated lesson files are written beneath `backend/artifacts/` and are ignored by Git.

### 5. Stop the app

```bash
docker compose down
```

## From question to lesson

1. The learner asks a question in plain English or with a typed equation.
2. The app estimates the level and chooses a model path.
3. The model writes a Manim animation for that question.
4. The generated code is checked before it is allowed to run.
5. Manim renders the lesson in an isolated container.
6. ElevenLabs provides synchronized narration and captions when available.

![A question moves through routing, generation, validation, rendering, and narration to become a visual lesson.](assets/product-flow.svg)

If a specialist, narration, or media step fails, the app can fall back to a simpler lesson path instead of hiding what happened.

## A tutor that can adapt

A lesson about fractions should not sound like a lecture on partial differential equations. Different questions may need different teaching styles, notation, and visual detail.

The current product uses small LoRA specialists for foundational, intermediate, and advanced lessons. They share one Qwen model, so each specialist only needs to learn a small set of changes.

Our research goes one step further: can the model discover useful specializations during training instead of having us decide them in advance?

That is the idea behind our Dynamic LoRA work. We observe how different training examples try to change the model and look for persistent disagreements that may justify a new specialist.

[Read the plain-language Dynamic LoRA explanation, including the mathematics.](training/src/dynamic_lora/README.md)

## Repository guide

```text
backend/       lesson API, validation, rendering, narration, and media
frontend/      learner-facing React application
training/      LoRA experiments, data checks, and experiment tracking
deployments/   Modal training and inference entrypoints
assets/        product and training diagrams
```

More detailed instructions live with each part of the repository:

- [Backend and rendering guide](backend/README.md)
- [Training guide](training/README.md)
- [Dynamic LoRA explanation](training/src/dynamic_lora/README.md)

## Local checks

Backend:

```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

Frontend:

```bash
cd frontend
npm ci
npm test
npm run typecheck
```

Dynamic LoRA training modules:

```bash
cd training
uv sync
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

## The goal

Math Tutor is not a library of prerecorded clips. Each lesson is generated for the learner's question.

The product goal is simple: make difficult mathematics easier to see, hear, and explore. The research goal is to help the model develop the right kinds of expertise without assuming those categories beforehand.
