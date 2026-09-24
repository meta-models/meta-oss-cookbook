# Security Policy

## Supported configuration

The initial supported configuration is macOS on Apple silicon with every
service bound to loopback. Do not expose the web, token, LiveKit, LLM, or worker
ports to a LAN or the internet.

## Reporting

Do not include credentials, transcripts, audio, model paths, or runtime logs in
a public issue. Use the repository host's private security advisory mechanism.

## Local trust boundary

A process running as the same operating-system user can reach loopback services
and read files that user can access. The token service is not an authentication
boundary against local malware. Runtime credentials are ephemeral, mode 0600,
and removed by normal shutdown.

The browser receives only a short-lived participant token and the fixed local
LiveKit URL. It must never receive model identifiers, artifact variants, local
paths, the LLM endpoint, native worker details, or cloud metadata.
