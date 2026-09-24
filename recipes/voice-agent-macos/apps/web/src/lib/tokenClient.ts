import { LIVEKIT_SERVER_URL, TOKEN_ENDPOINT } from "../config";

export interface ConnectionDetails {
  serverUrl: string;
  participantToken: string;
  roomName: string;
  participantIdentity: string;
}

const RESPONSE_FIELDS = [
  "participantIdentity",
  "participantToken",
  "roomName",
  "serverUrl",
] as const;

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0;

export async function requestConnection(
  signal?: AbortSignal,
): Promise<ConnectionDetails> {
  let response: Response;
  try {
    response = await fetch(TOKEN_ENDPOINT, {
      method: "POST",
      cache: "no-store",
      credentials: "omit",
      headers: { Accept: "application/json" },
      referrerPolicy: "no-referrer",
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new Error("The local connection service is not available.");
  }

  if (!response.ok) {
    throw new Error(
      "The local connection service could not start a conversation.",
    );
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      "The local connection service returned an invalid response.",
    );
  }

  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw new Error(
      "The local connection service returned an invalid response.",
    );
  }

  const candidate = body as Record<string, unknown>;
  const responseFields = Object.keys(candidate).sort();
  if (
    responseFields.length !== RESPONSE_FIELDS.length ||
    responseFields.some((field, index) => field !== RESPONSE_FIELDS[index])
  ) {
    throw new Error(
      "The local connection service returned an unexpected response.",
    );
  }

  if (
    !isNonEmptyString(candidate.serverUrl) ||
    !isNonEmptyString(candidate.participantToken) ||
    !isNonEmptyString(candidate.roomName) ||
    !isNonEmptyString(candidate.participantIdentity)
  ) {
    throw new Error(
      "The local connection service returned an incomplete response.",
    );
  }

  if (candidate.serverUrl !== LIVEKIT_SERVER_URL) {
    throw new Error(
      "The local connection service returned an unapproved media URL.",
    );
  }

  return {
    serverUrl: candidate.serverUrl,
    participantToken: candidate.participantToken,
    roomName: candidate.roomName,
    participantIdentity: candidate.participantIdentity,
  };
}
