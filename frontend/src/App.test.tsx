// Verify the prompt flow, workflow graph, and generated lesson artifacts.
// Tests use the public transport contract and backend-reported job stages.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App, LessonResult } from "./App";
import type { CreateLessonInput, LessonJob, LessonTransport } from "./contracts";
import {
  failedLesson,
  narratedLesson,
  partialLesson,
  queuedLesson,
  runningLesson,
} from "./fixtures";
import { MockLessonTransport } from "./transport";

function node(id: string) {
  return document.querySelector(`[data-node="${id}"]`);
}

describe("LessonResult", () => {
  it("shows a queued prompt at the workflow entry", () => {
    render(<LessonResult lesson={queuedLesson} />);

    expect(screen.getByText("Queued for a render worker")).toBeInTheDocument();
    expect(node("prompt")).toHaveClass("workflow-running");
    expect(node("adapter")).toHaveClass("workflow-pending");
  });

  it("marks the shared adapter as the active generation component", () => {
    render(<LessonResult lesson={runningLesson} />);

    expect(screen.getAllByText("Generating with the shared LoRA").length).toBeGreaterThan(1);
    expect(node("prompt")).toHaveClass("workflow-passed");
    expect(node("adapter")).toHaveClass("workflow-running");
    expect(node("source")).toHaveClass("workflow-pending");
  });

  it("shows all three validators running in parallel", () => {
    render(
      <LessonResult lesson={{ ...runningLesson, stage: "validating_output" }} />,
    );

    expect(node("media")).toHaveClass("workflow-running");
    expect(node("spatial")).toHaveClass("workflow-running");
    expect(node("visual")).toHaveClass("workflow-running");
    expect(node("render")).toHaveClass("workflow-passed");
  });

  it("updates completed validators while the remaining checks keep running", () => {
    render(
      <LessonResult
        lesson={{
          ...runningLesson,
          stage: "validating_output",
          initial_video_url: "/lessons/demo/video/initial",
          diagnostics: {
            validation_axes: [{ validator: "media", status: "pass" }],
          },
        }}
      />,
    );

    expect(node("media")).toHaveClass("workflow-passed");
    expect(node("spatial")).toHaveClass("workflow-running");
    expect(node("visual")).toHaveClass("workflow-running");
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video/initial",
    );
  });

  it("shows a repair through the same shared adapter", () => {
    render(
      <LessonResult
        lesson={{ ...runningLesson, stage: "repairing", attempt: 1 }}
      />,
    );

    expect(screen.getByText("Repair attempt 1")).toBeInTheDocument();
    expect(node("repair")).toHaveClass("workflow-running");
    expect(node("adapter")).not.toBeInTheDocument();
    expect(node("source")).not.toBeInTheDocument();
    expect(node("render")).not.toBeInTheDocument();
  });

  it("shows terminal validator results in the graph", () => {
    const failedValidation: LessonJob = {
      ...failedLesson,
      diagnostics: {
        failure_stage: "output_validation",
        validation_axes: [
          { validator: "media", status: "pass" },
          { validator: "spatial", status: "fail" },
          { validator: "visual_evidence", status: "uncertain" },
        ],
      },
    };
    render(<LessonResult lesson={failedValidation} />);

    expect(node("media")).toHaveClass("workflow-passed");
    expect(node("spatial")).toHaveClass("workflow-failed");
    expect(node("visual")).toHaveClass("workflow-uncertain");
    expect(node("publish")).toHaveClass("workflow-pending");
  });

  it("identifies the failed component", () => {
    render(
      <LessonResult
        lesson={{
          ...failedLesson,
          diagnostics: { failure_stage: "provider" },
        }}
      />,
    );

    expect(node("adapter")).toHaveClass("workflow-failed");
    expect(screen.queryByTestId("lesson-video")).not.toBeInTheDocument();
  });

  it("explains a failed source stage when selected", () => {
    render(
      <LessonResult
        lesson={{
          ...failedLesson,
          diagnostics: {
            failure_stage: "parse",
            line: 14,
            attempt_count: 2,
            repair_count: 1,
            validation_reports: [
              {
                validator: "source",
                findings: [
                  {
                    code: "source_admission_failed",
                    message: "Generated source did not pass deterministic admission checks.",
                    repair_instruction: "Return valid Python without an unterminated string.",
                  },
                ],
              },
            ],
          },
        }}
      />,
    );

    expect(node("adapter")).toHaveClass("workflow-passed");
    expect(node("source")).toHaveClass("workflow-failed");
    fireEvent.click(
      screen.getByRole("button", { name: "Inspect Source check stage" }),
    );

    const stageLog = screen.getByText("Stage log").closest(".workflow-inspector");
    expect(stageLog).toHaveClass("detail-card");
    expect(stageLog?.closest(".support-panel")).toBeInTheDocument();
    expect(document.querySelector(".workflow .workflow-inspector")).not.toBeInTheDocument();
    expect(screen.getByText("Generation attempts")).toBeInTheDocument();
    expect(screen.getByText("Repair attempts")).toBeInTheDocument();
    expect(screen.getByText("source_admission_failed")).toBeInTheDocument();
    expect(
      screen.getByText("Return valid Python without an unterminated string."),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Back to lesson" }));
    expect(screen.getByText("Generation stopped")).toBeInTheDocument();
  });

  it("shows available render logs in the details panel", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          diagnostics: { ...narratedLesson.diagnostics, logs: "Rendered scene.mp4" },
        }}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Inspect Manim render stage" }),
    );

    expect(screen.getByText("Execution output")).toBeInTheDocument();
    expect(screen.getByText("Rendered scene.mp4")).toBeInTheDocument();
    expect(screen.getByTestId("lesson-video")).toBeInTheDocument();
    expect(screen.getByText("Stage log").closest(".support-panel")).toBeInTheDocument();
  });

  it("shows narration failures on the media stage", () => {
    render(
      <LessonResult
        lesson={{
          ...failedLesson,
          initial_video_url: "/lessons/demo/video/initial",
          narration_status: "unavailable",
          diagnostics: {
            failure_stage: "narration",
            failure_kind: "operational",
            narration_error_code: "narration_media_assembly_failed",
            narration_error: "video muxing failed",
            narration_plan_attempt_count: 1,
          },
        }}
      />,
    );

    expect(node("adapter")).toHaveClass("workflow-passed");
    expect(node("source")).toHaveClass("workflow-passed");
    expect(node("render")).toHaveClass("workflow-passed");
    expect(node("media")).toHaveClass("workflow-failed");

    fireEvent.click(screen.getByRole("button", { name: "Inspect Media stage" }));

    expect(screen.getByText("Narration attempts")).toBeInTheDocument();
    expect(screen.getByText("narration_media_assembly_failed")).toBeInTheDocument();
    expect(screen.getByText("video muxing failed")).toBeInTheDocument();
  });

  it("switches between the initial and validated videos", () => {
    render(<LessonResult lesson={narratedLesson} />);

    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );
    expect(node("publish")).toHaveClass("workflow-passed");

    fireEvent.click(screen.getByRole("button", { name: "Initial video" }));
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video/initial",
    );

    fireEvent.click(screen.getByRole("button", { name: "Validated video" }));
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );
    expect(screen.queryByRole("button", { name: "</>" })).not.toBeInTheDocument();
  });

  it("shows only the shared-adapter inference path", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          diagnostics: {
            inference_path: "lora_adapter",
            inference_model: "shared-lora-qwen3-4b-manim-v1",
            repair_count: 0,
          },
        }}
      />,
    );

    expect(screen.getAllByText("Shared LoRA")).toHaveLength(2);
    expect(screen.getByText("shared-lora-qwen3-4b-manim-v1")).toBeInTheDocument();
    expect(screen.getByText("Direct")).toBeInTheDocument();
    expect(screen.queryByText(/specialist/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/fallback/i)).not.toBeInTheDocument();
  });

  it("keeps useful partial output visible", () => {
    render(<LessonResult lesson={partialLesson} />);
    expect(screen.getByText("Partial lesson available")).toBeInTheDocument();
    expect(screen.getByText("The explanation is ready, but video rendering failed.")).toBeInTheDocument();
  });
});

describe("App lesson flow", () => {
  it("submits, polls, and renders a terminal lesson", async () => {
    const transport = new MockLessonTransport([
      queuedLesson,
      runningLesson,
      narratedLesson,
    ]);
    render(<App transport={transport} pollIntervalMs={0} />);

    expect(screen.getByText("Create a visual lesson")).toBeInTheDocument();
    expect(screen.queryByText("Shared LoRA · Manim · Parallel validation")).not.toBeInTheDocument();
    expect(screen.queryByText("One repair attempt")).not.toBeInTheDocument();
    expect(screen.getByText("Your generated visual lesson will appear here.")).toBeInTheDocument();
    expect(document.querySelector(".workspace-grid > .support-panel")).toBeInTheDocument();
    expect(document.querySelector(".workspace-grid")).toHaveClass("workspace-with-details");
    expect(
      screen.getByText(
        "Generate a visual lesson to see its explanation and validation results.",
      ),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));

    await waitFor(() => expect(node("publish")).toHaveClass("workflow-passed"));
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );
    expect(screen.getByRole("button", { name: /generate lesson/i })).toBeEnabled();
  });

  it("returns to the initial video when a new lesson is submitted", async () => {
    let submission = 0;
    const validatingSecondLesson: LessonJob = {
      ...runningLesson,
      id: "second",
      stage: "validating_output",
      initial_video_url: "/lessons/second/video/initial",
    };
    const transport: LessonTransport = {
      async submitLesson(_input: CreateLessonInput) {
        submission += 1;
        return submission === 1 ? narratedLesson : validatingSecondLesson;
      },
      async getLesson(_id: string) {
        return new Promise<LessonJob>(() => undefined);
      },
    };
    render(<App transport={transport} pollIntervalMs={0} />);

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));
    await waitFor(() => expect(node("publish")).toHaveClass("workflow-passed"));
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));

    await waitFor(() =>
      expect(screen.getByTestId("lesson-video")).toHaveAttribute(
        "src",
        "/lessons/second/video/initial",
      ),
    );
  });

  it("re-enables submission when polling fails", async () => {
    const transport: LessonTransport = {
      async submitLesson(_input: CreateLessonInput) {
        return runningLesson;
      },
      async getLesson(_id: string) {
        throw new Error("polling unavailable");
      },
    };
    render(<App transport={transport} pollIntervalMs={0} />);

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("polling unavailable");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /generate lesson/i })).toBeEnabled(),
    );
  });
});
