// Render the lesson request, progress, and generated artifact experience.
// Progress labels mirror the backend lesson-stage contract.
import { type FormEvent, useRef, useState } from "react";

import type {
  LessonJob,
  LessonStage,
  LessonStatus,
  LessonTransport,
  NarrationStatus,
} from "./contracts";
import { isTerminal } from "./contracts";
import { HttpLessonTransport } from "./transport";
import "./styles.css";

const defaultTransport = new HttpLessonTransport();
const DEFAULT_PROMPT = "Explain why a² + b² = c² using a visual proof.";

interface AppProps {
  transport?: LessonTransport;
  pollIntervalMs?: number;
}

interface ValidationAxisSummary {
  validator: string;
  status: "pass" | "fail" | "uncertain" | "validator_error";
}

const examples = [
  "Explain why √2 is irrational",
  "Visualize Euler’s identity",
  "Gradient descent intuition",
  "Fourier transform from heat diffusion",
];

const progressSteps = [
  "Route prompt",
  "Generate code",
  "Validate code",
  "Render + narrate",
  "Validate output",
];

const stageOrder: LessonStage[] = [
  "routing",
  "generating_code",
  "validating_code",
  "rendering",
  "validating_output",
];

const stageLabels: Record<LessonStage, string> = {
  routing: "Routing prompt",
  generating_code: "Generating code",
  validating_code: "Validating code",
  rendering: "Rendering + narrating",
  validating_output: "Validating output",
  repairing: "Repairing lesson",
  ready: "Ready",
  failed: "Failed",
};

export function App({ transport = defaultTransport, pollIntervalMs = 700 }: AppProps) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [lesson, setLesson] = useState<LessonJob | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [polling, setPolling] = useState(false);
  const [artifactMode, setArtifactMode] = useState<"view" | "code">("view");
  const [captionsEnabled, setCaptionsEnabled] = useState(true);
  const mounted = useRef(true);
  const busy = polling;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim() || busy) return;
    mounted.current = true;
    setPolling(true);
    setRequestError(null);

    try {
      let next = await transport.submitLesson({ prompt: prompt.trim() });
      if (mounted.current) setLesson(next);
      while (!isTerminal(next.status)) {
        await delay(pollIntervalMs);
        next = await transport.getLesson(next.id);
        if (mounted.current) setLesson(next);
      }
    } catch (error) {
      if (mounted.current) {
        setRequestError(error instanceof Error ? error.message : "The lesson request failed");
      }
    } finally {
      if (mounted.current) setPolling(false);
    }
  }

  return (
    <main className="studio-shell">
      <header className="studio-header">
        <a className="brand" href="/" aria-label="Math Tutor home">
          <span className="brand-mark">∑</span>
          <span>
            <strong>Math Tutor</strong>
            <small>Visual math studio</small>
          </span>
        </a>
        <span className="engine-pill">Modal Qwen · LoRA specialists · Manim</span>
      </header>

      <section className="workspace-grid">
        <form className="composer-panel" onSubmit={submit}>
          <div className="panel-copy">
            <p className="section-kicker">Create a visual lesson</p>
            <h1>What should we make visible?</h1>
            <p>Describe a concept or ask a question. The studio turns it into a visual lesson.</p>
          </div>

          <label htmlFor="lesson-prompt">Prompt or formula</label>
          <textarea
            id="lesson-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={5}
            spellCheck={false}
          />

          <div className="sample-row" aria-label="Example prompts">
            {examples.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setPrompt(example)}
              >
                {example}
              </button>
            ))}
          </div>

          <button className="primary-action" type="submit" disabled={busy || !prompt.trim()}>
            {busy ? "Generating…" : "Generate lesson"}
            <span aria-hidden="true">→</span>
          </button>

          <div className="composer-meta">
            <span>1080p · 60 fps</span>
            <span>Typical render 18–24 sec</span>
          </div>
        </form>

        <section className="video-panel" aria-live="polite">
          <div className="video-panel-head">
            <div>
              <p className="section-kicker">Generated visual lesson</p>
              <h2>{lesson ? lessonTitle(lesson) : "Video output"}</h2>
            </div>
            <StatusBadge lesson={lesson} busy={busy} />
          </div>

          <div className="artifact-toolbar" aria-label="Lesson artifact controls">
            <div className="segmented-control" role="group" aria-label="Choose lesson artifact">
              <button
                type="button"
                className={artifactMode === "view" ? "active" : ""}
                onClick={() => setArtifactMode("view")}
              >
                View
              </button>
              <button
                type="button"
                className={artifactMode === "code" ? "active" : ""}
                onClick={() => setArtifactMode("code")}
              >
                {"</>"}
              </button>
            </div>
            <label className="caption-toggle">
              <input
                type="checkbox"
                checked={captionsEnabled}
                disabled={!lesson?.captions_url || artifactMode !== "view"}
                onChange={(event) => setCaptionsEnabled(event.target.checked)}
              />
              Captions
            </label>
          </div>

          <VideoStage
            lesson={lesson}
            busy={busy}
            mode={artifactMode}
            captionsEnabled={captionsEnabled}
          />

          <div className="video-controls-strip">
            <span>{artifactMode === "code" ? "Generated Manim source" : artifactStatus(lesson, captionsEnabled)}</span>
            <span>{lesson ? narrationLabel(lesson.narration_status) : "Narration pending"}</span>
          </div>

          {requestError && <p className="request-error" role="alert">{requestError}</p>}
        </section>
      </section>

      <SupportTabs lesson={lesson} />
    </main>
  );
}

function LessonResult({ lesson }: { lesson: LessonJob }) {
  const [artifactMode, setArtifactMode] = useState<"view" | "code">("view");
  const [captionsEnabled, setCaptionsEnabled] = useState(true);

  return (
    <section className="standalone-result">
      <div className="video-panel-head">
        <div>
          <p className="section-kicker">Generated visual lesson</p>
          <h2>{lessonTitle(lesson)}</h2>
        </div>
        <StatusBadge lesson={lesson} busy={false} />
      </div>
      <div className="artifact-toolbar" aria-label="Lesson artifact controls">
        <div className="segmented-control" role="group" aria-label="Choose lesson artifact">
          <button
            type="button"
            className={artifactMode === "view" ? "active" : ""}
            onClick={() => setArtifactMode("view")}
          >
            View
          </button>
          <button
            type="button"
            className={artifactMode === "code" ? "active" : ""}
            onClick={() => setArtifactMode("code")}
          >
            {"</>"}
          </button>
        </div>
        <label className="caption-toggle">
          <input
            type="checkbox"
            checked={captionsEnabled}
            disabled={!lesson.captions_url || artifactMode !== "view"}
            onChange={(event) => setCaptionsEnabled(event.target.checked)}
          />
          Captions
        </label>
      </div>
      <VideoStage
        lesson={lesson}
        busy={lesson.status === "queued" || lesson.status === "running"}
        mode={artifactMode}
        captionsEnabled={captionsEnabled}
      />
      {lesson.status !== "queued" && lesson.status !== "running" && <SupportTabs lesson={lesson} />}
    </section>
  );
}

function VideoStage({
  lesson,
  busy,
  mode,
  captionsEnabled,
}: {
  lesson: LessonJob | null;
  busy: boolean;
  mode: "view" | "code";
  captionsEnabled: boolean;
}) {
  if (!lesson) return <EmptyVideo />;
  if (lesson.status === "queued" || lesson.status === "running") {
    return <GeneratingVideo lesson={lesson} busy={busy} />;
  }
  if (mode === "code") return <CodeStage lesson={lesson} />;
  if (lesson.status === "failed") return <FailedVideo lesson={lesson} />;
  if (lesson.status === "partial") return <PartialVideo lesson={lesson} />;
  return <ReadyVideo lesson={lesson} captionsEnabled={captionsEnabled} />;
}

function EmptyVideo() {
  return (
    <div className="video-stage video-empty">
      <div className="chalk-orbit" />
      <div className="empty-video-copy">
        <span className="play-glyph">▶</span>
        <h3>Your generated visual lesson will appear here.</h3>
        <p>Enter a concept or equation, then generate a Manim-powered explanation.</p>
      </div>
    </div>
  );
}

function GeneratingVideo({ lesson }: { lesson: LessonJob; busy: boolean }) {
  const progressState = useRef({ jobId: lesson.id, furthestStage: -1 });
  if (progressState.current.jobId !== lesson.id) {
    progressState.current = { jobId: lesson.id, furthestStage: -1 };
  }
  const reportedStage = lesson.stage === "repairing"
    ? stageOrder.length - 1
    : stageOrder.indexOf(lesson.stage);
  progressState.current.furthestStage = Math.max(
    progressState.current.furthestStage,
    reportedStage,
  );
  const activeStage = progressState.current.furthestStage;
  const progress = activeStage < 0 ? 100 : ((activeStage + 0.5) / stageOrder.length) * 100;
  const heading = lesson.status === "queued"
    ? "Queued for a render worker"
    : stageLabels[lesson.stage];
  const phase = lesson.status === "queued" ? "Queued" : stageLabels[lesson.stage];

  return (
    <div className="video-stage video-generating">
      <div className="generation-topline">
        <span>{lesson.lesson}</span>
        <span>{phase}</span>
      </div>
      <div className="generation-center">
        <span className="spinner-mark">∑</span>
        <h3>{heading}</h3>
        <div className="progress-track">
          <span style={{ width: `${progress}%` }} />
        </div>
      </div>
      <div className="inline-steps">
        {progressSteps.map((step, index) => {
          const className = index < activeStage
            ? "is-complete"
            : index === activeStage
              ? "is-active"
              : "";
          return (
            <div
              key={step}
              className={className}
            >
              <span />
              <p>{index + 1}. {step}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ReadyVideo({
  lesson,
  captionsEnabled,
}: {
  lesson: LessonJob;
  captionsEnabled: boolean;
}) {
  const playableVideo = lesson.video_url ?? lesson.silent_video_url;

  return (
    <div className="video-stage video-ready">
      {playableVideo ? (
        <video data-testid="lesson-video" src={playableVideo} controls preload="metadata">
          {captionsEnabled && lesson.captions_url && (
            <track
              title="English captions"
              kind="captions"
              src={lesson.captions_url}
              srcLang="en"
              label="English"
              default
            />
          )}
        </video>
      ) : (
        <div className="video-placeholder">Video asset unavailable</div>
      )}
    </div>
  );
}

function CodeStage({ lesson }: { lesson: LessonJob }) {
  return (
    <div className="video-stage code-stage">
      <div className="code-stage-head">
        <span>Generated Manim · Python</span>
        <span>{lesson.generated_code ? "Ready" : "Waiting for source"}</span>
      </div>
      <pre><code>{lesson.generated_code ?? "# Manim source will appear here"}</code></pre>
    </div>
  );
}

function PartialVideo({ lesson }: { lesson: LessonJob }) {
  return (
    <div className="video-stage video-partial">
      <h3>Video render timed out</h3>
      <p>Some lesson assets could not be generated, but the available output is preserved.</p>
    </div>
  );
}

function FailedVideo({ lesson: _lesson }: { lesson: LessonJob }) {
  return (
    <div className="video-stage video-failed">
      <h3>Generation stopped</h3>
      <p>We couldn't generate this lesson. Please try again.</p>
    </div>
  );
}

function SupportTabs({ lesson }: { lesson: LessonJob | null }) {
  const axes = lesson ? readValidationAxes(lesson.diagnostics) : [];

  return (
    <section className="support-panel">
      <div className="support-content">
        {lesson ? (
          <article className="lesson-details-grid">
            <section className="detail-card">
              <p className="section-kicker">Mathematical intuition</p>
              <h2>Explanation</h2>
              <p>
                {lesson.explanation ?? `Generated visual lesson for: ${lesson.lesson}`}
              </p>
            </section>
            <section className="detail-card adapter-card">
              <p className="section-kicker">Model routing</p>
              <h2>Inference trace</h2>
              <InferenceRouting lesson={lesson} />
            </section>
            {axes.length > 0 && (
              <section className="detail-card validation-card">
                <p className="section-kicker">Quality checks</p>
                <h2>Output validation</h2>
                <div className="validation-results">
                  {axes.map((axis) => (
                    <div key={axis.validator}>
                      <span>{validatorLabel(axis.validator)}</span>
                      <strong className={`validation-${axis.status}`}>
                        {validationStatusLabel(axis.status)}
                      </strong>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </article>
        ) : (
          <div className="awaiting-content">
            Generate a visual lesson to unlock its explanation and adapter diagnostics.
          </div>
        )}
      </div>
    </section>
  );
}

function InferenceRouting({ lesson }: { lesson: LessonJob }) {
  const reportedPath = readText(lesson.diagnostics, ["routing_path", "inference_path"]);
  const usesBaseNormalizer = reportedPath === "lora_adapter_with_base_normalizer";
  const usesAdapter = reportedPath === "lora_adapter" || usesBaseNormalizer || (
    reportedPath === null && lesson.difficulty != null
  );
  const inferenceModel = readText(lesson.diagnostics, ["inference_model"]);
  const specialistModel = readText(lesson.diagnostics, ["specialist_model"])
    ?? lesson.difficulty;
  const normalizationModel = readText(lesson.diagnostics, ["normalization_model"])
    ?? inferenceModel;
  const model = usesBaseNormalizer
    ? `${specialistModel ?? "LoRA specialist"} → ${normalizationModel ?? "base model"}`
    : inferenceModel ?? (usesAdapter ? lesson.difficulty : "Qwen/Qwen3-4B");
  const policy = readText(lesson.diagnostics, ["routing_policy"]);
  const fallback = readText(lesson.diagnostics, ["routing_fallback"]);
  const route = policy === "automatic_heuristic"
    ? fallback === "base_model"
      ? "Automatic heuristic · base fallback"
      : "Automatic heuristic"
    : policy === "explicit_difficulty" || usesAdapter
      ? fallback === "base_model"
        ? "Explicit specialist route · base fallback"
        : "Explicit specialist route"
      : "Default synchronized path";

  return (
    <div className="adapter-routing">
      <div>
        <span>Inference</span>
        <strong>
          {usesBaseNormalizer
            ? "LoRA specialist + base normalizer"
            : usesAdapter ? "LoRA specialist" : "Base model"}
        </strong>
      </div>
      <div>
        <span>Model or adapter</span>
        <strong>{model}</strong>
      </div>
      <div>
        <span>Routing</span>
        <strong>{route}</strong>
      </div>
    </div>
  );
}

function StatusBadge({ lesson, busy }: { lesson: LessonJob | null; busy: boolean }) {
  const status = lesson?.status ?? (busy ? "running" : "idle");
  const text = !lesson
    ? busy ? "Rendering" : "Awaiting prompt"
    : status === "ready" ? "Ready"
      : status === "partial" ? "Partial output"
        : status === "failed" ? "Failed"
          : status === "queued" ? "Queued"
            : "Rendering";

  return <span className={`status-badge status-${status}`}>{text}</span>;
}

function lessonTitle(lesson: LessonJob) {
  if (lesson.status === "failed") return "Lesson failed";
  if (lesson.status === "partial") return "Partial lesson available";
  if (lesson.status === "queued") return "Queued";
  if (lesson.status === "running") return "Rendering";
  if (lesson.narration_status === "ready") return "Narrated lesson ready";
  if (lesson.narration_status === "unavailable") return "Video ready · narration unavailable";
  return "Video ready";
}

function narrationLabel(status: NarrationStatus) {
  if (status === "ready") return "Voice + captions";
  if (status === "pending") return "Narration pending";
  if (status === "unavailable") return "Silent fallback active";
  return "Narration not requested";
}

function artifactStatus(lesson: LessonJob | null, captionsEnabled: boolean) {
  if (!lesson) return "Video appears here";
  if (!lesson.captions_url) return "Captions unavailable";
  return captionsEnabled ? "Captions on" : "Captions off";
}

function readText(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return null;
}

function readValidationAxes(
  diagnostics: Record<string, unknown>,
): ValidationAxisSummary[] {
  const axes = diagnostics.validation_axes;
  if (!Array.isArray(axes)) return [];
  const supportedStatuses = new Set([
    "pass",
    "fail",
    "uncertain",
    "validator_error",
  ]);
  return axes.flatMap((axis) => {
    if (typeof axis !== "object" || axis === null) return [];
    const validator = "validator" in axis ? axis.validator : null;
    const status = "status" in axis ? axis.status : null;
    if (
      typeof validator !== "string"
      || typeof status !== "string"
      || !supportedStatuses.has(status)
    ) return [];
    return [{
      validator,
      status: status as ValidationAxisSummary["status"],
    }];
  });
}

function validatorLabel(validator: string) {
  if (validator === "media") return "Media";
  if (validator === "spatial") return "Layout";
  if (validator === "visual_evidence") return "Visual evidence";
  return validator.replaceAll("_", " ");
}

function validationStatusLabel(status: ValidationAxisSummary["status"]) {
  if (status === "pass") return "Passed";
  if (status === "fail") return "Failed";
  if (status === "validator_error") return "Check unavailable";
  return "Uncertain";
}

const delay = (milliseconds: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

export { LessonResult };
