# H20: local bilingual routing transport diagnosis

Status: Diagnostic source and measured records independently reviewed; encoding config repaired; no learned routing adoption.

Acceptance criteria:

- Reproduce the existing Arabic MCP HTTP 400 without modifying the old evidence.
- Compare ASCII-escaped JSON and native UTF-8 on the installed loopback runtime,
  preserving request bytes/hash, status/error body, outputs and elapsed time.
- Isolate transport, schema and model explanations using matched controls; do not
  infer Arabic incapability from malformed requests.
- Compare d1 to a standard-library learned classifier and deterministic routing on
  authored bilingual data with disjoint application-family train/test splits.
- Preserve every test prediction and report errors, abstentions, CPU latency and
  serialized classifier size. Tiny synthetic results cannot qualify adoption.
- Tests cover valid/invalid inputs, split leakage, conservative authority boundaries
  and malformed/nonfinite runtime responses. No new dependencies, weights or jobs.
- Preserve installed MCP/runtime/model source. Apply the authorized minimal
  d1_local-only config environment repair with a full local backup and verify a
  fresh unchanged MCP child using that configured environment. Do not restart
  the desktop or live inference server. Disclose that the existing desktop MCP
  process may retain its old environment until reconnect.

All outputs are advisory route suggestions. They cannot grant authority, execute
actions, approve releases or replace independent review. Original H14/H17 frozen
fixtures and the old d1 diagnostic are excluded from training and tuning.
