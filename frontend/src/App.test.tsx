import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App, LessonResult } from "./App";
import type { CreateLessonInput, LessonTransport } from "./contracts";
import {
  failedLesson,
  narratedLesson,
  partialLesson,
  queuedLesson,
  runningLesson,
  silentFallbackLesson,
} from "./fixtures";
import { MockLessonTransport } from "./transport";

describe("LessonResult", () => {
  it.each([
    [queuedLesson, "Queued for a render worker"],
    [runningLesson, "Generating code"],
  ])("renders progress state", (lesson, text) => {
    render(<LessonResult lesson={lesson} />);
    expect(screen.getAllByText(text).length).toBeGreaterThan(0);
  });

  it("shows the backend-reported generation stage", () => {
    render(
      <LessonResult
        lesson={{ ...runningLesson, stage: "validating_code" }}
      />,
    );

    expect(screen.getAllByText("Validating code")).toHaveLength(2);
    expect(screen.getByText("1. Route prompt").parentElement).toHaveClass("is-complete");
    expect(screen.getByText("2. Generate code").parentElement).toHaveClass("is-complete");
    expect(screen.getByText("3. Validate code").parentElement).toHaveClass("is-active");
    expect(screen.getByText("4. Render + narrate").parentElement).not.toHaveClass("is-complete");
  });

  it("plays the narrated result and exposes captions", () => {
    render(<LessonResult lesson={narratedLesson} />);
    expect(screen.getByText("Narrated lesson ready")).toBeInTheDocument();
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video",
    );
    expect(screen.getByTitle("English captions")).toHaveAttribute(
      "src",
      "/lessons/demo/captions",
    );
    expect(screen.queryByText("Scene 03 · visual proof")).not.toBeInTheDocument();
    expect(screen.queryByText("Synchronized captions")).not.toBeInTheDocument();
  });

  it("lets the user toggle captions off", () => {
    render(<LessonResult lesson={narratedLesson} />);

    fireEvent.click(screen.getByLabelText("Captions"));

    expect(screen.queryByTitle("English captions")).not.toBeInTheDocument();
  });

  it("shows generated Manim code inside the main artifact panel", () => {
    render(<LessonResult lesson={narratedLesson} />);

    fireEvent.click(screen.getByRole("button", { name: "</>" }));

    expect(screen.getByText("Generated Manim · Python")).toBeInTheDocument();
    expect(screen.getByText(/MathTex/)).toBeInTheDocument();
  });

  it("presents silent fallback as a successful lesson", () => {
    render(<LessonResult lesson={silentFallbackLesson} />);
    expect(screen.getByText("Video ready · narration unavailable")).toBeInTheDocument();
    expect(screen.getByTestId("lesson-video")).toHaveAttribute(
      "src",
      "/lessons/demo/video/silent",
    );
    expect(screen.queryByText(/lesson failed/i)).not.toBeInTheDocument();
  });

  it("keeps useful partial output visible", () => {
    render(<LessonResult lesson={partialLesson} />);
    expect(screen.getByText("Partial lesson available")).toBeInTheDocument();
    expect(screen.getByText("The explanation is ready, but video rendering failed.")).toBeInTheDocument();
  });

  it("shows terminal failure without a video player", () => {
    render(
      <LessonResult
        lesson={{
          ...failedLesson,
          error: "Manim exited with code 1: Traceback (most recent call last)",
        }}
      />,
    );
    expect(screen.getByText("Lesson failed")).toBeInTheDocument();
    expect(screen.getByText("We couldn't generate this lesson. Please try again.")).toBeInTheDocument();
    expect(screen.queryByText(/Traceback/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("lesson-video")).not.toBeInTheDocument();
  });

  it("shows the submitted prompt instead of a hard-coded theorem", () => {
    render(
      <LessonResult
        lesson={{
          ...silentFallbackLesson,
          lesson: "Explain why √2 is irrational",
          explanation: null,
        }}
      />,
    );

    expect(screen.getByText("Generated visual lesson for: Explain why √2 is irrational")).toBeInTheDocument();
    expect(screen.queryByText("a² + b² = c²")).not.toBeInTheDocument();
  });

  it("shows the actual base-model inference route", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          diagnostics: {
            inference_path: "base_model",
            inference_model: "Qwen/Qwen3-4B",
            routing_policy: "default",
          },
        }}
      />,
    );

    expect(screen.getByText("Inference trace")).toBeInTheDocument();
    expect(screen.getByText("Base model")).toBeInTheDocument();
    expect(screen.getByText("Qwen/Qwen3-4B")).toBeInTheDocument();
    expect(screen.getByText("Default synchronized path")).toBeInTheDocument();
    expect(screen.queryByText("Dynamic LoRA trace")).not.toBeInTheDocument();
  });

  it("shows the independent output-validation results", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          diagnostics: {
            validation_axes: [
              { validator: "media", status: "pass", finding_count: 0 },
              { validator: "spatial", status: "pass", finding_count: 0 },
              {
                validator: "visual_evidence",
                status: "uncertain",
                finding_count: 1,
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText("Output validation")).toBeInTheDocument();
    expect(screen.getByText("Media")).toBeInTheDocument();
    expect(screen.getByText("Layout")).toBeInTheDocument();
    expect(screen.getByText("Visual evidence")).toBeInTheDocument();
    expect(screen.getAllByText("Passed")).toHaveLength(2);
    expect(screen.getByText("Uncertain")).toBeInTheDocument();
  });

  it("shows the selected LoRA specialist when one was explicitly routed", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          difficulty: "advanced",
          diagnostics: {
            inference_path: "lora_adapter",
            inference_model: "advanced",
            routing_policy: "explicit_difficulty",
          },
        }}
      />,
    );

    expect(screen.getByText("LoRA specialist")).toBeInTheDocument();
    expect(screen.getByText("advanced")).toBeInTheDocument();
    expect(screen.getByText("Explicit specialist route")).toBeInTheDocument();
  });

  it("labels automatically inferred LoRA routing honestly", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          difficulty: "advanced",
          diagnostics: {
            inference_path: "lora_adapter",
            inference_model: "advanced",
            routing_policy: "automatic_heuristic",
          },
        }}
      />,
    );

    expect(screen.getByText("LoRA specialist")).toBeInTheDocument();
    expect(screen.getByText("Automatic heuristic")).toBeInTheDocument();
  });

  it("shows when a LoRA draft is normalized by the base model", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          difficulty: "intermediate",
          diagnostics: {
            inference_path: "base_model",
            routing_path: "lora_adapter_with_base_normalizer",
            inference_model: "Qwen/Qwen3-4B",
            specialist_model: "intermediate",
            normalization_model: "Qwen/Qwen3-4B",
            routing_policy: "automatic_heuristic",
          },
        }}
      />,
    );

    expect(screen.getByText("LoRA specialist + base normalizer")).toBeInTheDocument();
    expect(screen.getByText("intermediate → Qwen/Qwen3-4B")).toBeInTheDocument();
    expect(screen.getByText("Automatic heuristic")).toBeInTheDocument();
  });

  it("shows when an inferred specialist fell back to the base model", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          difficulty: "intermediate",
          diagnostics: {
            inference_path: "base_model",
            inference_model: "Qwen/Qwen3-4B",
            routing_policy: "automatic_heuristic",
            routing_fallback: "base_model",
          },
        }}
      />,
    );

    expect(screen.getByText("Base model")).toBeInTheDocument();
    expect(screen.getByText("Automatic heuristic · base fallback")).toBeInTheDocument();
  });

  it("shows when an explicitly selected specialist fell back to the base model", () => {
    render(
      <LessonResult
        lesson={{
          ...narratedLesson,
          difficulty: "advanced",
          diagnostics: {
            inference_path: "base_model",
            inference_model: "Qwen/Qwen3-4B",
            routing_policy: "explicit_difficulty",
            routing_fallback: "base_model",
          },
        }}
      />,
    );

    expect(screen.getByText("Base model")).toBeInTheDocument();
    expect(screen.getByText("Explicit specialist route · base fallback")).toBeInTheDocument();
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
    expect(screen.getByText("Your generated visual lesson will appear here.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /generate lesson/i }));

    expect(await screen.findByText("Narrated lesson ready")).toBeInTheDocument();
    expect(screen.getByText("Generated visual lesson")).toBeInTheDocument();
    expect(screen.getByText("Captions on")).toBeInTheDocument();
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
