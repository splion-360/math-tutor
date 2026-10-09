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

    expect(screen.getByText("Stage details")).toBeInTheDocument();
    expect(screen.getByText("Generation attempts")).toBeInTheDocument();
    expect(screen.getByText("Repair attempts")).toBeInTheDocument();
    expect(screen.getByText("source_admission_failed")).toBeInTheDocument();
    expect(
      screen.getByText("Return valid Python without an unterminated string."),
    ).toBeInTheDocument();
  });

  it("plays the accepted video and exposes generated code", () => {
    render(<LessonResult lesson={narratedLesson} />);

    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );
    expect(node("publish")).toHaveClass("workflow-passed");

    fireEvent.click(screen.getByRole("button", { name: "</>" }));
    expect(screen.getByText("Generated Manim · Python")).toBeInTheDocument();
    expect(screen.getByText(/MathTex/)).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));

    await waitFor(() => expect(node("publish")).toHaveClass("workflow-passed"));
    expect(document.querySelector(".workspace-grid > .support-panel")).toBeInTheDocument();
    expect(document.querySelector(".workspace-grid")).toHaveClass("workspace-with-details");
    expect(screen.getByRole("button", { name: /generate lesson/i })).toBeEnabled();
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
