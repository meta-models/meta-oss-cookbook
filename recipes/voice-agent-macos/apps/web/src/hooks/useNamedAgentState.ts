import { useVoiceAssistant } from "@livekit/components-react";

import { AGENT_NAME } from "../config";
import { normalizeAgentState, type AgentState } from "../lib/agentPresentation";

export interface NamedAgentState {
  hasNamedAgent: boolean;
  agentState?: AgentState;
  agentIdentity?: string;
}

export function useNamedAgentState(): NamedAgentState {
  const assistant = useVoiceAssistant();
  const agentName = assistant.agent?.attributes["lk.agent.name"];
  const hasNamedAgent = agentName === AGENT_NAME;

  return {
    hasNamedAgent,
    agentState: hasNamedAgent
      ? normalizeAgentState(assistant.state)
      : undefined,
    agentIdentity: hasNamedAgent ? assistant.agent?.identity : undefined,
  };
}
