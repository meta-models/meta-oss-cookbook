import { LiveKitRoom } from "@livekit/components-react";
import { useRef, useState } from "react";

import { MuseAvatar } from "./avatar/MuseAvatar";
import { RuntimeBadge } from "./components/RuntimeBadge";
import { VoiceSession } from "./components/VoiceSession";
import { usePrefersReducedMotion } from "./hooks/usePrefersReducedMotion";
import {
  getAgentPresentation,
  type SessionPhase,
} from "./lib/agentPresentation";
import { requestConnection, type ConnectionDetails } from "./lib/tokenClient";

export default function App() {
  const [phase, setPhase] = useState<SessionPhase>("idle");
  const [connection, setConnection] = useState<ConnectionDetails>();
  const [error, setError] = useState<string>();
  const intentionalDisconnect = useRef(false);
  const reducedMotion = usePrefersReducedMotion();

  const startConversation = async () => {
    intentionalDisconnect.current = false;
    setPhase("requesting");
    setError(undefined);
    try {
      const details = await requestConnection();
      setConnection(details);
      setPhase("active");
    } catch (caught) {
      setConnection(undefined);
      setPhase("error");
      setError(
        caught instanceof Error
          ? caught.message
          : "The conversation could not start.",
      );
    }
  };

  const finishConversation = () => {
    setConnection(undefined);
    setError(undefined);
    setPhase("idle");
  };

  const failConversation = () => {
    setConnection(undefined);
    setError("The local voice connection failed.");
    setPhase("error");
  };

  const handleDisconnected = () => {
    if (intentionalDisconnect.current) {
      finishConversation();
    } else {
      failConversation();
    }
  };

  if (connection && phase === "active") {
    return (
      <LiveKitRoom
        key={connection.roomName}
        serverUrl={connection.serverUrl}
        token={connection.participantToken}
        connect
        audio
        video={false}
        onDisconnected={handleDisconnected}
        onError={failConversation}
        onMediaDeviceFailure={() => {
          setConnection(undefined);
          setError(
            "Chrome could not use the microphone. Check its site permission and try again.",
          );
          setPhase("error");
        }}
        data-lk-theme="default"
      >
        <VoiceSession
          participantIdentity={connection.participantIdentity}
          onEnding={() => {
            intentionalDisconnect.current = true;
          }}
          onEnded={finishConversation}
        />
      </LiveKitRoom>
    );
  }

  const presentation = getAgentPresentation(phase, false, false);
  const isRequesting = phase === "requesting";

  return (
    <main className={`voice-shell welcome-shell tone-${presentation.tone}`}>
      <header className="app-header">
        <div className="header-title">
          <RuntimeBadge />
          <p className="eyebrow">Local Voice Agent</p>
          <h1>Talk with Muse Glimmer</h1>
        </div>
      </header>
      <section
        className="presence welcome-presence"
        aria-labelledby="welcome-title"
      >
        <MuseAvatar
          animation={presentation.animation}
          reducedMotion={reducedMotion}
        />
        <div className="welcome-copy">
          <h2 id="welcome-title">
            <span>One voice agent.</span>
            <span>Entirely on-device.</span>
          </h2>
          <p>
            Local ASR hears you, an on-device LLM thinks, and local TTS speaks
            back. No cloud required.
          </p>
        </div>
      </section>
      <div className="welcome-actions">
        {error ? (
          <p className="welcome-error" role="alert">
            {error}
          </p>
        ) : (
          <p className="privacy-note">
            Your microphone starts only after you choose to begin.
          </p>
        )}
        <button
          className="start-button"
          type="button"
          onClick={() => void startConversation()}
          disabled={isRequesting}
        >
          <span>
            {isRequesting ? "Preparing conversation" : "Start conversation"}
          </span>
          <ArrowIcon />
        </button>
      </div>
    </main>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 12h13M14 7l5 5-5 5" />
    </svg>
  );
}
