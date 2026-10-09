// Verify that the HTTP transport sends the public prompt-only request shape.
// These tests guard the boundary between the frontend and lesson API.

import { afterEach, describe, expect, it, vi } from "vitest";

import { HttpLessonTransport } from "./transport";

describe("HttpLessonTransport", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends the typed prompt to the lesson API", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: "job-1", status: "queued" }),
    });
    vi.stubGlobal("fetch", fetch);

    await new HttpLessonTransport().submitLesson({
      prompt: "Explain Euler's identity",
    });

    expect(fetch).toHaveBeenCalledWith("/lessons", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        prompt: "Explain Euler's identity",
      }),
    });
  });
});
