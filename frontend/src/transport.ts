// Implement browser and in-memory transports for the lesson-job API.
// Both transports preserve the same prompt submission and polling contract.

import type { CreateLessonInput, LessonJob, LessonTransport } from "./contracts";

export class HttpLessonTransport implements LessonTransport {
  constructor(private readonly baseUrl = "") {}

  async submitLesson(input: CreateLessonInput): Promise<LessonJob> {
    return this.request("/lessons", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: input.prompt }),
    });
  }

  async getLesson(id: string): Promise<LessonJob> {
    return this.request(`/lessons/${encodeURIComponent(id)}`);
  }

  private async request(path: string, init?: RequestInit): Promise<LessonJob> {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    if (!response.ok) {
      throw new Error(`Lesson API returned ${response.status}`);
    }
    return (await response.json()) as LessonJob;
  }
}

export class MockLessonTransport implements LessonTransport {
  private pollIndex = 0;

  constructor(private readonly states: readonly LessonJob[]) {
    if (states.length === 0) {
      throw new Error("MockLessonTransport needs at least one state");
    }
  }

  async submitLesson(_input: CreateLessonInput): Promise<LessonJob> {
    this.pollIndex = 0;
    return this.states[0];
  }

  async getLesson(_id: string): Promise<LessonJob> {
    this.pollIndex = Math.min(this.pollIndex + 1, this.states.length - 1);
    return this.states[this.pollIndex];
  }
}
