import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const disconnect = vi.fn();
const setMicrophoneEnabled = vi.fn();
let isMicrophoneEnabled = true;

vi.mock("@livekit/components-react", () => ({
  useRoomContext: () => ({ disconnect }),
  useLocalParticipant: () => ({
    isMicrophoneEnabled,
    localParticipant: { setMicrophoneEnabled },
  }),
}));

import { SessionControls } from "./SessionControls";

afterEach(() => {
  disconnect.mockReset();
  setMicrophoneEnabled.mockReset();
  isMicrophoneEnabled = true;
});

describe("session controls", () => {
  it("mutes an active microphone", async () => {
    setMicrophoneEnabled.mockResolvedValue(undefined);
    render(<SessionControls onEnding={vi.fn()} onEnded={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Mute" }));

    await waitFor(() =>
      expect(setMicrophoneEnabled).toHaveBeenCalledWith(false),
    );
    expect(disconnect).not.toHaveBeenCalled();
  });

  it("shows a safe error when microphone control fails", async () => {
    setMicrophoneEnabled.mockRejectedValue(new Error("device details"));
    render(<SessionControls onEnding={vi.fn()} onEnded={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Mute" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Microphone access was not available.",
    );
  });

  it("completes local teardown when disconnect rejects", async () => {
    const onEnding = vi.fn();
    const onEnded = vi.fn();
    setMicrophoneEnabled.mockResolvedValue(undefined);
    disconnect.mockRejectedValue(new Error("transport closed"));
    render(<SessionControls onEnding={onEnding} onEnded={onEnded} />);

    fireEvent.click(screen.getByRole("button", { name: "End" }));

    await waitFor(() => expect(onEnded).toHaveBeenCalledOnce());
    expect(onEnding).toHaveBeenCalledOnce();
    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("disables the microphone, disconnects, and reports a completed end action", async () => {
    const onEnding = vi.fn();
    const onEnded = vi.fn();
    setMicrophoneEnabled.mockResolvedValue(undefined);
    disconnect.mockResolvedValue(undefined);
    render(<SessionControls onEnding={onEnding} onEnded={onEnded} />);

    fireEvent.click(screen.getByRole("button", { name: "End" }));

    await waitFor(() => expect(onEnded).toHaveBeenCalledOnce());
    expect(onEnding).toHaveBeenCalledOnce();
    expect(setMicrophoneEnabled).toHaveBeenCalledWith(false);
    expect(disconnect).toHaveBeenCalledOnce();
  });
});
