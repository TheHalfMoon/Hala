# H387 — exact-byte Qwen3-VL processor handoff to a temporary directory

**Public Qwen3VLProcessor assets only. Not model weights, LoRA training,
patient data, private source screenshots, legal TRAIN or blind OOD.**

## Source-backed verification-to-load weakness

H216 / H361 previously called `verify_cached_processor_root`, which
recomputed the SHA-256 of the eight exact H218 public pinned files, then
passed the **same owner-supplied mutable directory** to Hugging Face
`AutoProcessor.from_pretrained`. The latter independently reopens
`tokenizer.json`, `config.json`, the chat template and related assets.
A cache file changed *between the preflight verification and HF loading*
could be different bytes from those H218 actually qualified. An
incorrectly loaded tokenizer or multimodal configuration can silently
change model-visible native-action inputs, despite a successful upstream
byte receipt. This is an inspectable TOCTOU in the source code; we do
**not** claim an actual compromise of the owner's local public cache.

## Narrow fail-closed correction

New `h387.verified_processor_handoff.verified_processor_handoff`
retains the existing exact eight-file H218 source verification, then
creates one short-lived staging directory **on the same drive as the
owner's original public processor snapshot** (avoids C: pressure).
It rereads each of the eight files into a bounded per-file byte
snapshot, checks exact H218 length and SHA-256 against the hardcoded
non-overridable public asset registry, writes only those verified
bytes into a newly created private temporary revision directory,
and calls H218's original verifier AGAIN on the staged directory.
Only the staged path may be passed to `AutoProcessor.from_pretrained`.
The context remains open throughout the entire H216/H361 source-only
processor audit and is removed both after normal completion and after
any exception. Any altered, extra or unknown file fails closed;
model weights are not copied and no network retrieval is permitted.

The original H218 pinned SHA byte registry is **unchanged**, along
with tokenizer configurations, image processing, native Qwen actions,
assistant completion masks, source families and official legal
TRAIN admission guards.

The isolation closes substitution of the **original owner cache** after
preflight. It is *not* a cryptographic defense against another process
running under the same identity that can write to the temporary staging
directory itself. It does not add independent signatures or legal
third-party approvals. The public processor files temporarily require
roughly 16 MB of disk and are cleaned by context management.

## Exact-byte adversarial tests

Six invented-only tests fabricate eight tiny public-named files with
test-scoped SHA values. They prove the staged files match exact
expected bytes; later mutation of original cache does not affect what
HF receives; a deterministic source mutation **immediately after the
first successful H218 verification** is refused before handoff;
unexpected extra files reject; exceptions remove stage contents; both
production H216 and H361 real loader *interfaces* use the staged
rather than owner source path, with H216's tokenization faked only in
these tests. No test modifies the production pinned SHA registry.

Owner Windows after implementing the fix at exact H387 source logic
commit `cde7655`: Python3.11 **6/6** invented tests PASS and
Python3.12 **6/6** PASS. Combined source-only H387/H216/H361/H386
with available Pillow CPU runtime: **26 total tests, 25 PASS and
one original-browser opt-in SKIP**, zero failures. The inherited
H216 artificial fake-processor test is updated to mock only the
temporary staging context, preserving its checks for the exact
Qwen3VLProcessor type and all real public-source guards. A further
real owner offline pinned Qwen3VLProcessor original H216 and H361
browser/source run is required for substantive current-head evidence.

The established H361 four-way GitHub Actions matrix (Ubuntu/Windows
× Python3.11/3.12) now explicitly runs H387 tests with existing
H386 finite tensor and H385 source-rows SHA binding tests, H216
fake-source guards, H359 192 original public browser PNG+native
action contract, and H199. All hosted CI must complete at
the **exact latest PR head** before qualification is claimed.

## Unchanged NO-GO model-release gates

H199 officially legally admitted TRAIN manifest SHA allowlist EMPTY.
H355 independently reviewed rendered-source screenshots, title and
derivative-weights rights trusted key registry EMPTY, Issue #384 OPEN.
H356 truly new never-exposed source-family-disjoint, independently
custodied, sealed bilingual EN/AR blind OOD trusted registry EMPTY,
Issue #385 OPEN. Public original 192 source images and Qwen processor
assets are EXPOSED; they are not legal TRAIN or genuinely blind OOD.
No lawful LoRA training, learned base-vs-adapted gain, model weights
or release. Keep PR DRAFT, do not merge protected main.
