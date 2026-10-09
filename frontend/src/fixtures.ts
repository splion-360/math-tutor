// Provide deterministic lesson states for frontend workflow and artifact tests.
// Fixtures mirror the public lesson-job contract exposed by the backend.

import type { LessonJob } from "./contracts";

const base: LessonJob = {
  id: "demo",
  lesson: "pythagorean-theorem",
  status: "queued",
  stage: "accepted",
  created_at: "2026-09-19T18:00:00Z",
  attempt: 0,
  started_at: null,
  completed_at: null,
  explanation: null,
  generated_code: null,
  initial_video_url: null,
  video_url: null,
  silent_video_url: null,
  captions_url: null,
  narration_status: "pending",
  diagnostics: {},
  error: null,
};

export const queuedLesson: LessonJob = { ...base };

export const runningLesson: LessonJob = {
  ...base,
  status: "running",
  stage: "generating_code",
  started_at: "2026-09-19T18:00:01Z",
};

export const narratedLesson: LessonJob = {
  ...base,
  status: "ready",
  stage: "ready",
  started_at: "2026-09-19T18:00:01Z",
  completed_at: "2026-09-19T18:00:09Z",
  explanation: "A right triangle relates its two legs to its hypotenuse.",
  generated_code: "equation = MathTex(r\"a^2 + b^2 = c^2\")",
  initial_video_url: "/lessons/demo/video/initial",
  video_url: "/lessons/demo/video",
  silent_video_url: "/lessons/demo/video/silent",
  captions_url: "/lessons/demo/captions",
  narration_status: "ready",
};

export const partialLesson: LessonJob = {
  ...base,
  status: "partial",
  stage: "failed",
  completed_at: "2026-09-19T18:00:09Z",
  explanation: "The explanation is ready, but video rendering failed.",
  narration_status: "unavailable",
  error: "Manim render unavailable",
};

export const failedLesson: LessonJob = {
  ...base,
  status: "failed",
  stage: "failed",
  completed_at: "2026-09-19T18:00:04Z",
  narration_status: "unavailable",
  error: "The lesson could not be generated.",
};
