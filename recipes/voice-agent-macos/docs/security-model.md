# Security Model

## Guarantees

The supported profile binds every network service to IPv4 loopback. LiveKit
advertises only loopback candidates and disables external-IP discovery. The
browser receives a short-lived room token restricted to microphone publication
and subscription; it cannot publish camera, screen, or data tracks.

Runtime credentials are generated locally for each `up`, written mode 0600,
injected only into the LiveKit, token, and worker processes, and deleted by
normal shutdown. They are protocol credentials required by a local LiveKit
server, not cloud credentials.

The browser may know only the public product name, `assistant` agent identity,
room/participant identities, the participant JWT, the fixed token endpoint,
the fixed LiveKit URL, state enums, and conversation transcripts. Model IDs,
quantization, artifact paths, LLM port 8000, native executable details, private
endpoints, and credentials stay server-side.

## Trust boundary

Loopback is a network boundary, not a same-user authentication boundary. A
process running as the same operating-system user can call the token endpoint
and may read files that user can access. Requests without an Origin are
accepted for local native clients. Browser origins and Host headers are still
restricted exactly to approved loopback values.

## Enforcement

- Configuration rejects non-loopback LiveKit and MuseGlimmer endpoints.
- The web client rejects token responses containing unknown fields or any
  LiveKit URL other than `ws://127.0.0.1:7880`.
- A production Content Security Policy excludes the LLM endpoint.
- The post-start privacy audit checks listeners, managed connections, token
  responses, runtime credential permissions, and browser bundle strings.
- The publication check prevents private artifacts and workstation paths from
  entering source control.

This design does not defend against malware executing as the same user, a
compromised browser, or an intentionally modified local build.
