import { describe, expect, it } from "vitest";

import type { TranscriptEntry } from "../lib/transcript";
import { selectMuseFace } from "./avatarExpression";

function entry(
  speaker: "user" | "agent",
  text: string,
  order = 0,
): TranscriptEntry {
  return {
    key: `${speaker}-${order}`,
    segmentId: String(order),
    participantIdentity: speaker,
    speaker,
    text,
    final: true,
    order,
  };
}

describe("avatar expression selection", () => {
  it("uses lifecycle faces while listening and connecting", () => {
    const transcript = [entry("user", "Why is that scary?")];

    expect(selectMuseFace("listening", transcript)).toBe("attentive");
    expect(selectMuseFace("working", transcript)).toBe("attentive");
  });

  it.each([
    ["Why does this work?", "curious"],
    ["I am confused by that", "confused"],
    ["Are you sure this is really true?", "suspicious"],
    ["Wow, no way!", "surprised"],
    ["I feel scared and worried", "scared"],
    ["This makes me angry", "angry"],
    ["That is a private and embarrassing question", "shy"],
    ["I am tired and sleepy", "sleepy"],
    ["Whatever, this is boring", "unimpressed"],
  ] as const)("maps a user question to %s-style expression", (text, face) => {
    expect(selectMuseFace("thinking", [entry("user", text)])).toBe(face);
  });

  it.each([
    ["That is wonderful!", "excited"],
    ["Haha, that was funny", "laughing"],
    ["Congratulations, you did it", "proud"],
    ["I am sorry that happened", "sad"],
    ["Here is the answer.", "happy"],
  ] as const)("maps an agent response to %s-style expression", (text, face) => {
    expect(selectMuseFace("happy", [entry("agent", text)])).toBe(face);
  });

  it("uses neutral when there is no conversation yet", () => {
    expect(selectMuseFace("idle", [])).toBe("neutral");
  });

  it("uses the newest matching speaker entry", () => {
    const transcript = [
      entry("user", "Why?", 0),
      entry("agent", "I am not sure.", 1),
      entry("user", "Are you sure?", 2),
    ];

    expect(selectMuseFace("thinking", transcript)).toBe("suspicious");
    expect(selectMuseFace("happy", transcript)).toBe("confused");
  });
});
