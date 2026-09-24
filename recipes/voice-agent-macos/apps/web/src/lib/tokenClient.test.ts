import { afterEach, describe, expect, it, vi } from "vitest";

import { LIVEKIT_SERVER_URL, TOKEN_ENDPOINT } from "../config";
import { requestConnection } from "./tokenClient";

const validConnection = {
  serverUrl: LIVEKIT_SERVER_URL,
  participantToken: "local-token",
  roomName: "glimmer-one",
  participantIdentity: "web-one",
};

function mockResponse(body: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    ),
  );
}

afterEach(() => vi.restoreAllMocks());

describe("token client", () => {
  it("posts to the fixed local endpoint without credentials or referrer data", async () => {
    mockResponse(validConnection);

    await expect(requestConnection()).resolves.toEqual(validConnection);
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledWith(
      TOKEN_ENDPOINT,
      expect.objectContaining({
        method: "POST",
        cache: "no-store",
        credentials: "omit",
        referrerPolicy: "no-referrer",
        headers: { Accept: "application/json" },
      }),
    );
  });

  it.each([
    "wss://127.0.0.1:7880",
    "ws://localhost:7880",
    "ws://127.0.0.1:7881",
    "ws://192.168.1.5:7880",
  ])("rejects the unapproved LiveKit URL %s", async (serverUrl) => {
    mockResponse({ ...validConnection, serverUrl });

    await expect(requestConnection()).rejects.toThrow("unapproved media URL");
  });

  it("rejects unknown response fields", async () => {
    mockResponse({
      ...validConnection,
      debug: "should-not-cross-the-browser-boundary",
    });

    await expect(requestConnection()).rejects.toThrow("unexpected response");
  });

  it("rejects missing or empty approved fields without exposing response values", async () => {
    mockResponse({ ...validConnection, participantToken: " " });

    await expect(requestConnection()).rejects.toThrow("incomplete response");
  });

  it("uses a generic error when the service returns malformed JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("{", { status: 200 })),
    );

    await expect(requestConnection()).rejects.toThrow("invalid response");
  });

  it("uses a generic error when the local service is unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new TypeError("connection refused")),
    );

    await expect(requestConnection()).rejects.toThrow(
      "local connection service is not available",
    );
  });

  it("preserves abort errors for cancellation handling", async () => {
    const abortError = new DOMException("aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abortError));

    await expect(requestConnection()).rejects.toBe(abortError);
  });
});
