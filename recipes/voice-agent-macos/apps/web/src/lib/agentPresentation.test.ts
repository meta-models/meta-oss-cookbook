import { describe, expect, it } from "vitest";

import { getAgentPresentation, normalizeAgentState } from "./agentPresentation";

describe("agent presentation", () => {
  it.each([
    ["idle", "idle", "Ready when you are", "ready"],
    ["listening", "listening", "Listening", "active"],
    ["thinking", "thinking", "Thinking", "active"],
    ["speaking", "happy", "Speaking", "speaking"],
  ] as const)(
    "maps %s to its visible voice state",
    (state, animation, status, tone) => {
      expect(getAgentPresentation("active", true, true, state)).toEqual({
        animation,
        status,
        tone,
      });
    },
  );

  it("prioritizes connection and named agent discovery", () => {
    expect(getAgentPresentation("active", false, false)).toMatchObject({
      status: "Joining",
    });
    expect(getAgentPresentation("active", true, false)).toMatchObject({
      status: "Waking up Muse",
    });
  });

  it("represents requesting and error phases independently of the room", () => {
    expect(getAgentPresentation("requesting", false, false)).toMatchObject({
      animation: "working",
      status: "Preparing your conversation",
    });
    expect(getAgentPresentation("error", false, false)).toMatchObject({
      status: "Could not connect",
      tone: "error",
    });
  });

  it("normalizes only known states", () => {
    expect(normalizeAgentState("thinking")).toBe("thinking");
    expect(normalizeAgentState("disconnected")).toBeUndefined();
    expect(normalizeAgentState({ state: "idle" })).toBeUndefined();
  });
});
