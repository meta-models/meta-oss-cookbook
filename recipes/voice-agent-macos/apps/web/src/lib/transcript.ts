export type TranscriptSpeaker = "user" | "agent";

export interface TranscriptEntry {
  key: string;
  segmentId: string;
  participantIdentity: string;
  speaker: TranscriptSpeaker;
  text: string;
  final: boolean;
  order: number;
}

export interface TranscriptUpdate {
  segmentId: string;
  participantIdentity: string;
  speaker: TranscriptSpeaker;
  text: string;
  final: boolean;
}

export function mergeTranscriptUpdates(
  current: TranscriptEntry[],
  updates: TranscriptUpdate[],
  maxEntries = 50,
): TranscriptEntry[] {
  const byKey = new Map(current.map((entry) => [entry.key, entry]));
  let nextOrder =
    current.reduce((maximum, entry) => Math.max(maximum, entry.order), -1) + 1;

  for (const update of updates) {
    const key = `${update.participantIdentity}:${update.segmentId}`;
    const existing = byKey.get(key);
    byKey.set(key, {
      key,
      segmentId: update.segmentId,
      participantIdentity: update.participantIdentity,
      speaker: update.speaker,
      text: update.text,
      final: update.final,
      order: existing?.order ?? nextOrder++,
    });
  }

  return Array.from(byKey.values())
    .sort((left, right) => left.order - right.order)
    .slice(-maxEntries);
}
