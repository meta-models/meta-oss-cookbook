# Local Observability

Logs remain under ignored `.local/logs` and are not uploaded. The worker emits
structured records for:

- User speech state transitions.
- Final ASR transcript availability and character count.
- VAD-detected speech that produced no final transcript.
- Agent state transitions.
- Pipeline/provider errors without browser-visible native detail.
- Session closure and reason.
- LLM prompt/completion token counts, time to first token, generation duration,
  finish reason, and cancellation outcome.

Transcript text is debug-only and must not be included in public issue reports
by default. Shareable diagnostic ZIPs contain the redacted report and structured
events only; raw `runtime.log` remains local and is never bundled automatically.
Native stderr is bounded and local. `make logs` follows all five managed service
logs.
