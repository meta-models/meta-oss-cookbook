export type AgentState =
  | "initializing"
  | "idle"
  | "listening"
  | "thinking"
  | "speaking";
export type SessionPhase =
  | "idle"
  | "requesting"
  | "active"
  | "ending"
  | "error";
export type MuseAnimation =
  | "idle"
  | "listening"
  | "thinking"
  | "working"
  | "happy";

export interface AgentPresentation {
  animation: MuseAnimation;
  status: string;
  tone: "ready" | "active" | "speaking" | "error";
}

export function normalizeAgentState(value: unknown): AgentState | undefined {
  switch (value) {
    case "initializing":
    case "idle":
    case "listening":
    case "thinking":
    case "speaking":
      return value;
    default:
      return undefined;
  }
}

export function getAgentPresentation(
  phase: SessionPhase,
  isConnected: boolean,
  hasNamedAgent: boolean,
  agentState?: AgentState,
): AgentPresentation {
  if (phase === "error") {
    return { animation: "idle", status: "Could not connect", tone: "error" };
  }
  if (phase === "requesting" || phase === "ending") {
    return {
      animation: "working",
      status:
        phase === "requesting"
          ? "Preparing your conversation"
          : "Ending conversation",
      tone: "active",
    };
  }
  if (phase === "idle") {
    return { animation: "idle", status: "Ready to talk", tone: "ready" };
  }
  if (!isConnected) {
    return { animation: "working", status: "Joining", tone: "active" };
  }
  if (!hasNamedAgent || agentState === "initializing") {
    return { animation: "working", status: "Waking up Muse", tone: "active" };
  }

  switch (agentState) {
    case "idle":
      return { animation: "idle", status: "Ready when you are", tone: "ready" };
    case "listening":
      return { animation: "listening", status: "Listening", tone: "active" };
    case "thinking":
      return { animation: "thinking", status: "Thinking", tone: "active" };
    case "speaking":
      return { animation: "happy", status: "Speaking", tone: "speaking" };
    default:
      return { animation: "working", status: "Working", tone: "active" };
  }
}
