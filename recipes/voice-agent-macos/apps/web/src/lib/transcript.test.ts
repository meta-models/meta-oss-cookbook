import { describe, expect, it } from "vitest";

import { mergeTranscriptUpdates } from "./transcript";

describe("transcript reducer", () => {
  it("replaces interim text and preserves its order when final", () => {
    const interim = mergeTranscriptUpdates(
      [],
      [
        {
          segmentId: "one",
          participantIdentity: "web-1",
          speaker: "user",
          text: "what is the",
          final: false,
        },
      ],
    );
    const final = mergeTranscriptUpdates(interim, [
      {
        segmentId: "one",
        participantIdentity: "web-1",
        speaker: "user",
        text: "What is the weather like?",
        final: true,
      },
    ]);

    expect(final).toHaveLength(1);
    expect(final[0]).toMatchObject({
      text: "What is the weather like?",
      final: true,
      order: 0,
    });
  });

  it("keeps reused segment ids distinct by participant", () => {
    const transcript = mergeTranscriptUpdates(
      [],
      [
        {
          segmentId: "one",
          participantIdentity: "web-1",
          speaker: "user",
          text: "Hello",
          final: true,
        },
        {
          segmentId: "one",
          participantIdentity: "agent-1",
          speaker: "agent",
          text: "Hi",
          final: true,
        },
      ],
    );

    expect(transcript.map((entry) => entry.key)).toEqual([
      "web-1:one",
      "agent-1:one",
    ]);
  });

  it("bounds history to the newest entries", () => {
    const transcript = mergeTranscriptUpdates(
      [],
      Array.from({ length: 4 }, (_, index) => ({
        segmentId: String(index),
        participantIdentity: "web-1",
        speaker: "user" as const,
        text: String(index),
        final: true,
      })),
      2,
    );

    expect(transcript.map((entry) => entry.text)).toEqual(["2", "3"]);
  });
});
