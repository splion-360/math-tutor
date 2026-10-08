// Define the HTTP contracts shared by the Math Tutor UI and transport.
// Stage values stay aligned with the backend lesson domain.
export type LessonStatus = "queued" | "running" | "ready" | "partial" | "failed";

export type LessonStage =
  | "routing"
  | "generating_code"
  | "validating_code"
  | "rendering"
  | "validating_output"
  | "repairing"
  | "ready"
  | "failed";

export type Difficulty = "foundational" | "intermediate" | "advanced";

export type NarrationStatus =
  | "not_requested"
  | "pending"
  | "ready"
  | "unavailable";

export interface LessonJob {
  id: string;
  lesson: string;
  status: LessonStatus;
  stage: LessonStage;
  created_at: string;
  difficulty?: Difficulty | null;
  started_at: string | null;
  completed_at: string | null;
  explanation: string | null;
  generated_code: string | null;
  video_url: string | null;
  silent_video_url: string | null;
  captions_url: string | null;
  narration_status: NarrationStatus;
  diagnostics: Record<string, unknown>;
  error: string | null;
}

export interface CreateLessonInput {
  prompt: string;
  difficulty?: Difficulty;
}

export interface LessonTransport {
  submitLesson(input: CreateLessonInput): Promise<LessonJob>;
  getLesson(id: string): Promise<LessonJob>;
}

export const isTerminal = (status: LessonStatus): boolean =>
  status === "ready" || status === "partial" || status === "failed";
