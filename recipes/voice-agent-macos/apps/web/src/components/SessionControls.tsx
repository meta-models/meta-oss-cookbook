import { useLocalParticipant, useRoomContext } from "@livekit/components-react";
import { useState } from "react";

interface SessionControlsProps {
  onEnding: () => void;
  onEnded: () => void;
}

export function SessionControls({ onEnding, onEnded }: SessionControlsProps) {
  const room = useRoomContext();
  const { isMicrophoneEnabled, localParticipant } = useLocalParticipant();
  const [pendingAction, setPendingAction] = useState<"microphone" | "end">();
  const [controlError, setControlError] = useState<string>();

  const toggleMicrophone = async () => {
    setPendingAction("microphone");
    setControlError(undefined);
    try {
      await localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled);
    } catch {
      setControlError("Microphone access was not available.");
    } finally {
      setPendingAction(undefined);
    }
  };

  const endConversation = async () => {
    setPendingAction("end");
    onEnding();
    try {
      await localParticipant.setMicrophoneEnabled(false);
    } catch {
      // Disconnect even when the browser has already removed the track.
    }
    try {
      await room.disconnect();
    } catch {
      // Local teardown still completes when the transport has already failed.
    } finally {
      onEnded();
    }
  };

  return (
    <div className="controls-wrap">
      {controlError ? (
        <p className="control-error" role="alert">
          {controlError}
        </p>
      ) : null}
      <div className="controls" aria-label="Conversation controls">
        <button
          className={`control-button control-button--mic ${isMicrophoneEnabled ? "" : "is-muted"}`}
          type="button"
          onClick={() => void toggleMicrophone()}
          disabled={pendingAction !== undefined}
          aria-pressed={!isMicrophoneEnabled}
        >
          <MicrophoneIcon muted={!isMicrophoneEnabled} />
          <span>{isMicrophoneEnabled ? "Mute" : "Unmute"}</span>
        </button>
        <button
          className="control-button control-button--end"
          type="button"
          onClick={() => void endConversation()}
          disabled={pendingAction !== undefined}
        >
          <EndIcon />
          <span>End</span>
        </button>
      </div>
    </div>
  );
}

function MicrophoneIcon({ muted }: { muted: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      {muted ? (
        <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 5.2 2.05M15 11.2V6a3 3 0 0 0-5.72-1.25M5 11a7 7 0 0 0 11.86 5.04M19 11a7 7 0 0 1-.38 2.28M12 18v3M8 21h8M3 3l18 18" />
      ) : (
        <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3ZM5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" />
      )}
    </svg>
  );
}

function EndIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5.4 15.4a10 10 0 0 1 13.2 0M7.8 18.5l-3-4.1M16.2 18.5l3-4.1" />
    </svg>
  );
}
