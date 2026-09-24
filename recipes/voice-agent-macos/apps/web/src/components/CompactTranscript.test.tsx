import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { TranscriptEntry } from "../lib/transcript";
import { CompactTranscript } from "./CompactTranscript";

function entry(
  index: number,
  speaker: "user" | "agent" = "user",
): TranscriptEntry {
  return {
    key: `${speaker}-${index}`,
    segmentId: String(index),
    participantIdentity: speaker,
    speaker,
    text: `Line ${index}`,
    final: index % 2 === 0,
    order: index,
  };
}

describe("compact transcript", () => {
  it("stays hidden until the first transcript arrives", () => {
    const { container } = render(<CompactTranscript entries={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("scrolls smoothly when the newest transcript text changes", () => {
    const scrollTo = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      value: scrollTo,
    });
    const initialEntry = entry(1, "agent");
    const { rerender } = render(<CompactTranscript entries={[initialEntry]} />);
    scrollTo.mockClear();

    rerender(
      <CompactTranscript
        entries={[{ ...initialEntry, text: "A longer interim response" }]}
      />,
    );

    expect(scrollTo).toHaveBeenCalledWith({
      top: 0,
      behavior: "smooth",
    });
  });

  it("renders only the six newest entries with speaker labels and interim styling", () => {
    render(
      <CompactTranscript
        entries={Array.from({ length: 8 }, (_, index) =>
          entry(index, index % 2 ? "agent" : "user"),
        )}
      />,
    );

    expect(screen.queryByText("Line 1")).not.toBeInTheDocument();
    expect(screen.getByText("Line 2")).toBeVisible();
    expect(screen.getByText("Line 7")).toHaveClass("transcript-interim");
    expect(screen.getAllByText("Muse")).toHaveLength(3);
    expect(screen.getAllByText("You")).toHaveLength(3);
  });
});
