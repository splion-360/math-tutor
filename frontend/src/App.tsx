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
  findingCount?: number;
  advisoryCount?: number;
}

const examples = [
  "Explain why √2 is irrational",
  "Visualize Euler’s identity",
  "Gradient descent intuition",
  "Fourier transform from heat diffusion",
];

const stageLabels: Record<LessonStage, string> = {
  accepted: "Prompt accepted",
  generating_code: "Generating with the shared LoRA",
  validating_code: "Checking generated source",
  rendering: "Rendering the Manim scene",
  validating_output: "Running validators in parallel",
  repairing: "Sending validator feedback to the shared LoRA",
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
      </header>

      <section className={`workspace-grid${lesson ? " workspace-with-details" : ""}`}>
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
            rows={4}
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
        </form>

        <section className="video-panel" aria-live="polite">
          <div className="video-panel-head">
            <div>
              <p className="section-kicker">Generated visual lesson</p>
              <h2>{lesson ? lessonTitle(lesson) : "Video output"}</h2>
            </div>
            <StatusBadge lesson={lesson} busy={busy} />
          </div>

          <WorkflowGraph lesson={lesson} />

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

        {lesson && <SupportTabs lesson={lesson} />}
      </section>
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
      <WorkflowGraph lesson={lesson} />
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

type WorkflowNodeState = "pending" | "running" | "passed" | "failed" | "uncertain";

interface WorkflowNode {
  id: string;
  label: string;
  state: WorkflowNodeState;
}

interface WorkflowFinding {
  code: string;
  message: string;
  repairInstruction: string | null;
}

interface WorkflowStageDetails {
  description: string;
  facts: { label: string; value: string }[];
  findings: WorkflowFinding[];
}

function WorkflowGraph({ lesson }: { lesson: LessonJob | null }) {
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const nodes = workflowNodes(lesson);
  const node = (id: string) => nodes.find((item) => item.id === id)!;
  const selectedNode = selectedStage ? node(selectedStage) : null;
  const repairLabel = lesson
    && lesson.attempt > 0
    && (lesson.status === "queued" || lesson.status === "running")
    ? `Repair attempt ${lesson.attempt}`
    : null;

  return (
    <section className="workflow" aria-label="Lesson generation workflow">
      <div className="workflow-heading">
        <span>Workflow</span>
        <strong>{repairLabel ?? (lesson ? stageLabels[lesson.stage] : "Waiting for a prompt")}</strong>
      </div>
      <div className="workflow-graph">
        <WorkflowNodeView
          node={node("prompt")}
          selected={selectedStage === "prompt"}
          onSelect={setSelectedStage}
        />
        <WorkflowArrow />
        <WorkflowNodeView
          node={node("adapter")}
          selected={selectedStage === "adapter"}
          onSelect={setSelectedStage}
        />
        <WorkflowArrow />
        <WorkflowNodeView
          node={node("source")}
          selected={selectedStage === "source"}
          onSelect={setSelectedStage}
        />
        <WorkflowArrow />
        <WorkflowNodeView
          node={node("render")}
          selected={selectedStage === "render"}
          onSelect={setSelectedStage}
        />
        <WorkflowArrow />
        <div className="workflow-validation" aria-label="Parallel validators">
          <span className="workflow-branch-label">Parallel validation</span>
          <WorkflowNodeView
            node={node("media")}
            selected={selectedStage === "media"}
            onSelect={setSelectedStage}
          />
          <WorkflowNodeView
            node={node("spatial")}
            selected={selectedStage === "spatial"}
            onSelect={setSelectedStage}
          />
          <WorkflowNodeView
            node={node("visual")}
            selected={selectedStage === "visual"}
            onSelect={setSelectedStage}
          />
        </div>
        <WorkflowArrow />
        <WorkflowNodeView
          node={node("publish")}
          selected={selectedStage === "publish"}
          onSelect={setSelectedStage}
        />
      </div>
      {selectedNode && <WorkflowStageInspector node={selectedNode} lesson={lesson} />}
    </section>
  );
}

function WorkflowNodeView({
  node,
  selected,
  onSelect,
}: {
  node: WorkflowNode;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const symbol = node.state === "passed"
    ? "✓"
    : node.state === "failed"
      ? "×"
      : node.state === "uncertain"
        ? "?"
        : node.state === "running"
          ? "●"
          : "○";
  return (
    <button
      type="button"
      className={`workflow-node workflow-${node.state}${selected ? " workflow-selected" : ""}`}
      data-node={node.id}
      aria-label={`Inspect ${node.label} stage`}
      aria-pressed={selected}
      onClick={() => onSelect(node.id)}
    >
      <span aria-hidden="true">{symbol}</span>
      <strong>{node.label}</strong>
      <small>{workflowStateLabel(node.state)}</small>
    </button>
  );
}

function WorkflowArrow() {
  return <span className="workflow-arrow" aria-hidden="true">→</span>;
}

function workflowNodes(lesson: LessonJob | null): WorkflowNode[] {
  const labels: Record<string, string> = {
    prompt: "Prompt",
    adapter: "Shared LoRA",
    source: "Source check",
    render: "Manim render",
    media: "Media",
    spatial: "Layout",
    visual: "Visual",
    publish: "Video ready",
  };
  const states = Object.fromEntries(
    Object.keys(labels).map((id) => [id, "pending" as WorkflowNodeState]),
  );
  if (!lesson) return Object.entries(labels).map(([id, label]) => ({ id, label, state: states[id] }));

  states.prompt = "passed";
  const stage = lesson.stage;
  const stageIndex = ["accepted", "generating_code", "validating_code", "rendering", "validating_output", "ready"].indexOf(stage);
  if (stageIndex >= 1 || stage === "repairing") states.adapter = "passed";
  if (stageIndex >= 2) states.source = "passed";
  if (stageIndex >= 3) states.render = "passed";
  if (stageIndex >= 4) {
    states.media = "running";
    states.spatial = "running";
    states.visual = "running";
  }
  if (stage === "accepted") states.prompt = "running";
  if (stage === "generating_code" || stage === "repairing") states.adapter = "running";
  if (stage === "validating_code") states.source = "running";
  if (stage === "rendering") states.render = "running";

  const axes = readValidationAxes(lesson.diagnostics);
  for (const axis of axes) {
    const id = axis.validator === "visual_evidence" ? "visual" : axis.validator;
    if (!(id in states)) continue;
    states[id] = axis.status === "pass"
      ? "passed"
      : axis.status === "uncertain"
        ? "uncertain"
        : "failed";
  }

  if (lesson.status === "ready") {
    for (const id of Object.keys(states)) states[id] = "passed";
  } else if (lesson.status === "failed" || lesson.status === "partial") {
    const failedNode = failedWorkflowNode(lesson);
    if (failedNode) states[failedNode] = "failed";
  }

  return Object.entries(labels).map(([id, label]) => ({ id, label, state: states[id] }));
}

function failedWorkflowNode(lesson: LessonJob): string | null {
  const failure = readText(lesson.diagnostics, ["failure_stage", "failed_stage"]);
  if (failure === "provider" || failure === "generating_code") return "adapter";
  if (["extraction", "parse", "validation", "validating_code"].includes(failure ?? "")) {
    return "source";
  }
  if (failure === "render" || failure === "rendering") return "render";
  if (failure === "output_validation" || failure === "validating_output") return null;
  return "publish";
}

function workflowStateLabel(state: WorkflowNodeState) {
  if (state === "running") return "Running";
  if (state === "passed") return "Passed";
  if (state === "failed") return "Failed";
  if (state === "uncertain") return "Uncertain";
  return "Waiting";
}

function WorkflowStageInspector({
  node,
  lesson,
}: {
  node: WorkflowNode;
  lesson: LessonJob | null;
}) {
  const details = workflowStageDetails(node, lesson);
  return (
    <div className={`workflow-inspector workflow-inspector-${node.state}`}>
      <div className="workflow-inspector-heading">
        <div>
          <span>Stage details</span>
          <strong>{node.label}</strong>
        </div>
        <span>{workflowStateLabel(node.state)}</span>
      </div>
      <p>{details.description}</p>
      {details.facts.length > 0 && (
        <dl>
          {details.facts.map((fact) => (
            <div key={fact.label}>
              <dt>{fact.label}</dt>
              <dd>{fact.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {details.findings.length > 0 && (
        <div className="workflow-findings">
          {details.findings.map((finding) => (
            <article key={`${finding.code}-${finding.message}`}>
              <code>{finding.code}</code>
              <p>{finding.message}</p>
              {finding.repairInstruction && <small>{finding.repairInstruction}</small>}
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function workflowStageDetails(
  node: WorkflowNode,
  lesson: LessonJob | null,
): WorkflowStageDetails {
  const descriptions: Record<string, string> = {
    prompt: "The submitted lesson request enters the generation queue.",
    adapter: "The shared LoRA generates a complete Manim scene from the prompt.",
    source: (
      "Deterministic checks parse the Python and reject unsafe or invalid Manim source."
    ),
    render: "The admitted scene runs inside the isolated Manim renderer.",
    media: "Media checks inspect the rendered file, streams, duration, and captions contract.",
    spatial: "Layout checks inspect frame boundaries, margins, object size, and intersections.",
    visual: "The visual model checks sampled frames for bounded visible defects.",
    publish: "A lesson is published only after every required validation axis passes.",
  };
  const facts: { label: string; value: string }[] = [];
  if (!lesson) return { description: descriptions[node.id], facts, findings: [] };

  const diagnostics = lesson.diagnostics;
  const addFact = (label: string, value: unknown) => {
    if (typeof value === "string" && value.trim()) facts.push({ label, value });
    if (typeof value === "number") facts.push({ label, value: String(value) });
  };
  if (node.id === "prompt") addFact("Prompt", lesson.lesson);
  if (node.id === "adapter") {
    const provenance = readRecord(diagnostics.generation_provenance);
    addFact(
      "Model",
      readText(diagnostics, ["inference_model"])
        ?? (provenance ? readText(provenance, ["model"]) : null),
    );
    addFact("Repair attempt", lesson.attempt);
  }
  if (node.id === "source") {
    addFact("Failure phase", readText(diagnostics, ["failure_stage"]));
    addFact("Line", diagnostics.line);
  }
  if (node.id === "render") {
    addFact("Renderer", diagnostics.renderer);
    addFact("Elapsed seconds", diagnostics.elapsed_seconds);
    addFact("Infrastructure retries", diagnostics.infrastructure_retry_count);
  }
  if (["media", "spatial", "visual"].includes(node.id)) {
    const validator = node.id === "visual" ? "visual_evidence" : node.id;
    const axis = readValidationAxes(diagnostics).find(
      (item) => item.validator === validator,
    );
    addFact("Findings", axis?.findingCount);
    addFact("Advisories", axis?.advisoryCount);
  }
  if (node.id === "publish") {
    addFact("Validation", diagnostics.validation_status);
    addFact("Attempts", diagnostics.attempt_count);
  }

  const findings = readStageFindings(diagnostics, node.id);
  if (node.state === "failed" && findings.length === 0) {
    findings.push(fallbackStageFinding(diagnostics, node.id));
  }
  return { description: descriptions[node.id], facts, findings };
}

function fallbackStageFinding(
  diagnostics: Record<string, unknown>,
  stageId: string,
): WorkflowFinding {
  const failure = readText(diagnostics, ["failure_stage", "failed_stage"]);
  if (stageId === "source" && failure === "parse") {
    const line = typeof diagnostics.line === "number" ? ` at line ${diagnostics.line}` : "";
    return {
      code: "source_parse_failed",
      message: `The generated source could not be parsed as Python${line}.`,
      repairInstruction: null,
    };
  }
  if (stageId === "source" && failure === "extraction") {
    return {
      code: "source_extraction_failed",
      message: "The model response did not contain an extractable Manim scene.",
      repairInstruction: null,
    };
  }
  if (stageId === "source") {
    return {
      code: "source_validation_failed",
      message: "The generated scene violated a deterministic source-admission rule.",
      repairInstruction: null,
    };
  }
  return {
    code: `${stageId}_failed`,
    message: failure
      ? `The workflow stopped during ${failure.replaceAll("_", " ")}.`
      : "This stage did not complete.",
    repairInstruction: null,
  };
}

function GeneratingVideo({ lesson }: { lesson: LessonJob; busy: boolean }) {
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
        <p>Follow the workflow above for live progress.</p>
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
              <p className="section-kicker">Generation</p>
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
            Generate a visual lesson to see its explanation and validation results.
          </div>
        )}
      </div>
    </section>
  );
}

function InferenceRouting({ lesson }: { lesson: LessonJob }) {
  const model = readText(lesson.diagnostics, ["inference_model"])
    ?? "shared-lora-qwen3-4b-manim-v1";
  const repairCount = lesson.attempt || Number(lesson.diagnostics.repair_count ?? 0);

  return (
    <div className="adapter-routing">
      <div>
        <span>Inference</span>
        <strong>Shared LoRA</strong>
      </div>
      <div>
        <span>Adapter</span>
        <strong>{model}</strong>
      </div>
      <div>
        <span>Generation path</span>
        <strong>{repairCount > 0 ? `Direct · ${repairCount} repair` : "Direct"}</strong>
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
  return "Video ready";
}

function narrationLabel(status: NarrationStatus) {
  if (status === "ready") return "Voice + captions";
  if (status === "pending") return "Narration pending";
  if (status === "unavailable") return "Narration unavailable";
  return "Silent lesson";
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

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? value as Record<string, unknown>
    : null;
}

function readStageFindings(
  diagnostics: Record<string, unknown>,
  stageId: string,
): WorkflowFinding[] {
  const validator = stageId === "visual" ? "visual_evidence" : stageId;
  const reports = Array.isArray(diagnostics.validation_reports)
    ? diagnostics.validation_reports
    : [];
  const report = reports
    .map(readRecord)
    .find((item) => item && item.validator === validator);
  const values = report && Array.isArray(report.findings)
    ? report.findings
    : stageId === "source" && Array.isArray(diagnostics.findings)
      ? diagnostics.findings
      : [];
  return values.flatMap((value) => {
    const finding = readRecord(value);
    if (!finding) return [];
    const code = typeof finding.code === "string" ? finding.code : null;
    const message = typeof finding.message === "string" ? finding.message : null;
    if (!code || !message) return [];
    return [{
      code,
      message,
      repairInstruction: typeof finding.repair_instruction === "string"
        ? finding.repair_instruction
        : null,
    }];
  });
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
    const findingCount = "finding_count" in axis ? axis.finding_count : null;
    const advisoryCount = "advisory_count" in axis ? axis.advisory_count : null;
    if (
      typeof validator !== "string"
      || typeof status !== "string"
      || !supportedStatuses.has(status)
    ) return [];
    return [{
      validator,
      status: status as ValidationAxisSummary["status"],
      findingCount: typeof findingCount === "number" ? findingCount : undefined,
      advisoryCount: typeof advisoryCount === "number" ? advisoryCount : undefined,
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
