import {
  RoomAudioRenderer,
  StartAudio,
  useConnectionState,
  useLocalParticipant,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";

import { selectMuseFace } from "../avatar/avatarExpression";
import { MuseAvatar } from "../avatar/MuseAvatar";
import { useNamedAgentState } from "../hooks/useNamedAgentState";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion";
import { useTranscriptionSegments } from "../hooks/useTranscriptionSegments";
import { getAgentPresentation } from "../lib/agentPresentation";
import { CompactTranscript } from "./CompactTranscript";
import { RuntimeBadge } from "./RuntimeBadge";
import { SessionControls } from "./SessionControls";

interface VoiceSessionProps {
  participantIdentity: string;
  onEnding: () => void;
  onEnded: () => void;
}

export function VoiceSession({
  participantIdentity,
  onEnding,
  onEnded,
}: VoiceSessionProps) {
  const connectionState = useConnectionState();
  const { isMicrophoneEnabled } = useLocalParticipant();
  const namedAgent = useNamedAgentState();
  const transcript = useTranscriptionSegments(participantIdentity);
  const reducedMotion = usePrefersReducedMotion();
  const presentation = getAgentPresentation(
    "active",
    connectionState === ConnectionState.Connected,
    namedAgent.hasNamedAgent,
    namedAgent.agentState,
  );
  const face = selectMuseFace(presentation.animation, transcript);

  return (
    <main
      className={`voice-shell conversation-shell tone-${presentation.tone}`}
    >
      <Header />
      <section className="presence" aria-label="Muse voice assistant">
        <div className="presence-avatar">
          <MuseAvatar
            animation={presentation.animation}
            face={face}
            reducedMotion={reducedMotion}
          />
        </div>
        <div className="state-caption" aria-live="polite" aria-atomic="true">
          <span className="state-mark" aria-hidden="true" />
          {presentation.status}
          {!isMicrophoneEnabled ? (
            <span className="muted-note"> | Microphone muted</span>
          ) : null}
        </div>
        <div className="transcript-row">
          <CompactTranscript entries={transcript} />
        </div>
      </section>
      <div className="conversation-dock">
        <StartAudio className="audio-unlock" label="Enable sound" />
        <SessionControls onEnding={onEnding} onEnded={onEnded} />
      </div>
      <RoomAudioRenderer />
    </main>
  );
}

function Header() {
  return (
    <header className="app-header">
      <div className="header-title">
        <RuntimeBadge />
        <p className="eyebrow">Local Voice Agent</p>
        <h1>Talk with Muse Glimmer</h1>
      </div>
    </header>
  );
}
