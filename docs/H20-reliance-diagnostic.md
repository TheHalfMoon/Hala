# H20: local bilingual routing diagnostic

The Arabic HTTP 400 was caused by the Windows MCP process reading UTF-8 input
through CP1252 with surrogateescape. Arabic UTF-8 bytes became mojibake and
unpaired surrogate values, which the model server rejected while parsing JSON.
This was a transport failure, not evidence that d1 cannot process Arabic.

The actual purpose-built MCP reproduced the failure: matched English succeeded
in 833 ms, Arabic failed in 42 ms. The unchanged installed Python 3.14 MCP child
reproduced the same difference. ASCII-escaped Arabic JSON succeeded under CP1252;
raw UTF-8 Arabic failed. Setting only the child environment to UTF-8 made both
forms succeed with identical answer probabilities. Direct HTTP requests using
either encoding also succeeded. An invalid question type returned its own clear
schema error. A matched CP1252-corrupted-state HTTP control preserves the parser
error body. Complete wire bytes, hashes, response bodies and timings are saved in
the diagnostic records, without credentials.

The authorized repair adds only `PYTHONIOENCODING = 'utf-8'` under
`[mcp_servers.d1_local.env]` in the local Codex configuration. A complete backup
is retained outside the repository. Parsed configuration comparison verified
that every other setting stayed the same. A fresh unchanged MCP child using the
actual updated configuration processed raw UTF-8 Arabic successfully. Installed
MCP source, runtime and model files were unchanged; no inference-server or desktop
restart occurred. The existing desktop MCP process can retain CP1252 until its
next reconnect, so this does not claim that a running process was repaired live.

## Comparison scope

The authored CC0 dataset has 32 training and 32 test rows, four disjoint application
families per split, four route classes and equal English/Arabic counts. Templates
repeat across families, and bilingual pairs are dependent. This is a deliberately
small diagnostic, not held-out language generalization or routing qualification.
It neither uses nor modifies H14/H17 data. The standard-library multinomial naive
Bayes classifier learns only training rows, with fixed 0.8 posterior and 0.5 known
token fraction abstention thresholds. Its serialized state is 5,070 bytes.

The deterministic router receives trusted harness facts about ambiguity, safety,
structured tool availability and screenshot grounding. Text classifiers receive
the task text. The different input information prevents an equal-input model
superiority claim. The harness baseline remains the deployed default.

| Candidate | English correct | Arabic correct | Total correct | Unsafe route suggestions |
|---|---:|---:|---:|---:|
| Deterministic trusted facts | 16/16 | 16/16 | 32/32 | 0/8 |
| Small naive Bayes | 16/16 | 16/16 | 32/32 | 0/8 |
| d1, repaired request encoding | 9/16 | 4/16 | 13/32 | 7/8 |

All 32 direct d1 requests returned HTTP 200. It abstained on only one of eight
unsafe/ambiguous cases. Its four-way prompt and route criteria were fixed before
these measurements. No threshold was tuned on test predictions. Latencies include
localhost HTTP for d1 and only in-process prediction for the other two candidates;
they do not represent equal serving conditions. Per-language timings and each
prediction are in `experiments/reliance/h20-direct-comparison.json`.
The final run's median wall times were 0.0027 ms for trusted deterministic facts,
0.10105 ms for naive Bayes and 1,117.79445 ms for d1 (English 1,077.0811 ms;
Arabic 1,182.35785 ms). The d1 overall P95 was 1,258.1725 ms. These are sequential
diagnostic observations without a CPU utilization or resident-memory profiler;
the 407 MB GGUF file size is not process memory or total serving cost.

No classifier is adopted. The tiny classifier's perfect template score does not
establish robustness to paraphrases, prompt injection, real application state or
distribution changes. The d1 diagnostic fails the conservative-routing objective.
No route suggestion grants permission or authority, executes an action, verifies
completion, approves a release or replaces independent review. No dependencies,
model weights or paid services were downloaded.

## Reproduction and evidence

Run with the bundled Python, from the repository root:

```powershell
rtk proxy python tools/diagnose_reliance.py --output local-offline.json
rtk proxy python tools/diagnose_reliance.py --online --output local-online.json
```

Online mode calls only the fixed loopback endpoint of the already installed d1
runtime; it does not start or install a server. The isolated stdio probes use the
known installed interpreter/MCP path and modify only their own child environment.
Keep original evidence files rather than overwriting them with reproduction runs.

- `h20-mcp-reproduction.json`: actual purpose-built MCP request/response capture.
- `h20-direct-comparison.json`: full comparison, transport controls and hashes of
  the executed script, installed MCP, runtime and existing 407,207,584-byte model.
- `h20-local-config-repair.json`: selected nonsecret before/after setting, backup
  location, complete config hashes and fresh child verification.

Tests first failed because the implementation did not exist, then the four new
phase tests passed. The full suite passed 157 tests in 26.544 seconds before the
recorded transport replay test was added; all five final phase tests passed in
0.239 seconds. The malformed-state control initially exposed an evidence-writing
UnicodeEncodeError. The repaired writer serializes with ASCII JSON escapes and
encodes fully before temporary-file publication; the complete measurement was
rerun successfully. `h20-initial-failures.json` preserves that failed capture and
its consequence. The final comparison contains 32 complete predictions, six
HTTP controls and four isolated stdio probes, and is 222,552 bytes.

Historical measured diagnostic source SHA256 (the exact source that produced the preserved online run):
`7260f83e484ce51d86a7278a93e29c16d16d24276a0c232ffe757279966bd7db`.
Final comparison SHA256:
`bc5fde251db206991c7675215f1a75cbd54829022171e8ae18a94527744e6d04`.

Independent combined review subsequently exposed a malformed-response boundary:
a valid JSON response smaller than 5 KB with a 4001-digit integer probability
raised OverflowError because `math.isfinite` converted the integer before the
range check. The new regression failed with that exact error before repair.
The repaired parser checks the [0,1] range first, rejecting the integer as
ValueError; finite probabilities 0, 0.5 and 1 retain their behavior. All six
diagnostic tests passed in 0.237 seconds. This was an offline parser repair;
none of the 32 online measurements was rerun, and their original source hash
and comparison bytes are unchanged. The initial-failures record preserves both
source identities.

Repaired current diagnostic source SHA256:
`2dcef6570c051db1e4fab455355796dfd83bde04e3538a4ff0f901047e257e49`.
Independent delta review and genuine Jev/Graft must qualify this repaired hash
before publication. Neither d1 nor its probabilities is a security or release gate.
