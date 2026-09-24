import { useRoomContext } from "@livekit/components-react";
import {
  RoomEvent,
  type Participant,
  type TranscriptionSegment,
} from "livekit-client";
import { useEffect, useState } from "react";

import { AGENT_NAME } from "../config";
import {
  mergeTranscriptUpdates,
  type TranscriptEntry,
  type TranscriptUpdate,
} from "../lib/transcript";

export function useTranscriptionSegments(
  localParticipantIdentity: string,
): TranscriptEntry[] {
  const room = useRoomContext();
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);

  useEffect(() => {
    const handleTranscription = (
      segments: TranscriptionSegment[],
      participant?: Participant,
    ) => {
      if (!participant) return;

      const speaker =
        participant.identity === localParticipantIdentity
          ? "user"
          : participant.attributes["lk.agent.name"] === AGENT_NAME
            ? "agent"
            : undefined;
      if (!speaker) return;

      const updates: TranscriptUpdate[] = segments
        .filter((segment) => segment.text.trim().length > 0)
        .map((segment) => ({
          segmentId: segment.id,
          participantIdentity: participant.identity,
          speaker,
          text: segment.text,
          final: segment.final,
        }));

      if (updates.length > 0) {
        setEntries((current) => mergeTranscriptUpdates(current, updates));
      }
    };

    room.on(RoomEvent.TranscriptionReceived, handleTranscription);
    return () => {
      room.off(RoomEvent.TranscriptionReceived, handleTranscription);
    };
  }, [localParticipantIdentity, room]);

  return entries;
}
