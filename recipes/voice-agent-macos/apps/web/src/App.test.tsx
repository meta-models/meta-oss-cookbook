import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  requestConnection: vi.fn(),
  disconnectRoom: undefined as (() => void) | undefined,
}));

vi.mock("./lib/tokenClient", () => ({
  requestConnection: mocks.requestConnection,
}));
vi.mock("./avatar/MuseAvatar", () => ({ MuseAvatar: () => <div>Muse</div> }));
vi.mock("./components/RuntimeBadge", () => ({
  RuntimeBadge: () => <div>ExecuTorch</div>,
}));
vi.mock("@livekit/components-react", () => ({
  LiveKitRoom: ({
    children,
    onDisconnected,
  }: {
    children: ReactNode;
    onDisconnected: () => void;
  }) => {
    mocks.disconnectRoom = onDisconnected;
    return <div>{children}</div>;
  },
}));
vi.mock("./components/VoiceSession", () => ({
  VoiceSession: ({
    onEnding,
    onEnded,
  }: {
    onEnding: () => void;
    onEnded: () => void;
  }) => (
    <div>
      <button type="button" onClick={onEnding}>
        Mark ending
      </button>
      <button type="button" onClick={onEnded}>
        Finish
      </button>
    </div>
  ),
}));

import App from "./App";

beforeEach(() => {
  mocks.disconnectRoom = undefined;
  mocks.requestConnection.mockReset();
  mocks.requestConnection.mockResolvedValue({
    serverUrl: "ws://127.0.0.1:7880",
    participantToken: "token",
    roomName: "room",
    participantIdentity: "participant",
  });
});

async function start() {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: /Start conversation/ }));
  await screen.findByRole("button", { name: "Mark ending" });
}

describe("conversation disconnect handling", () => {
  it("shows an error after an unexpected disconnect", async () => {
    await start();

    mocks.disconnectRoom?.();

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The local voice connection failed.",
    );
  });

  it("returns to idle after an intentional disconnect", async () => {
    await start();
    fireEvent.click(screen.getByRole("button", { name: "Mark ending" }));

    mocks.disconnectRoom?.();

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Start conversation/ }),
      ).toBeEnabled(),
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
