import { useEffect, useRef } from "react";

import type { TranscriptEntry } from "../lib/transcript";

interface CompactTranscriptProps {
  entries: TranscriptEntry[];
}

export function CompactTranscript({ entries }: CompactTranscriptProps) {
  const visibleEntries = entries.slice(-6);
  const transcriptRef = useRef<HTMLElement>(null);
  const latestEntry = visibleEntries.at(-1);

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (!transcript) return;

    if (typeof transcript.scrollTo === "function") {
      transcript.scrollTo({
        top: transcript.scrollHeight,
        behavior: "smooth",
      });
    } else {
      transcript.scrollTop = transcript.scrollHeight;
    }
  }, [latestEntry?.key, latestEntry?.text, visibleEntries.length]);

  if (visibleEntries.length === 0) return null;

  return (
    <section
      className="transcript"
      aria-label="Conversation transcript"
      ref={transcriptRef}
    >
      <ol
        className="transcript-list"
        aria-live="polite"
        aria-relevant="additions text"
      >
        {visibleEntries.map((entry) => (
          <li
            className={`transcript-line transcript-line--${entry.speaker}`}
            key={entry.key}
          >
            <span className="transcript-speaker">
              {entry.speaker === "user" ? "You" : "Muse"}
            </span>
            <span className={entry.final ? undefined : "transcript-interim"}>
              {entry.text}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
