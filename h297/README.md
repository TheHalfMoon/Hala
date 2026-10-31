# H297 — prospective strict two-stage FieldPort model-time grammar

**Code-only model-facing action protocol; no real Qwen inference, GPU,
fine-tuning, adapted GUI score, official TRAIN/DEV or blind OOD.**

H296 adds a genuine author-owned offline bilingual fictional form
GUI with 12 screenshot-based two-action tasks: focus a field, then
type a synthetic code. This was independently qualified as
**10/10 real native source-fixture browser tests on Ubuntu Chromium
and Windows Edge** at H296 exact head
`98ba9b8ac2678813bd68ab5ebb172af49baef75c`.
Fixture correct clicks/typing are provided by a deterministic
source-owned test callback, **NOT** real pretrained/adapted Qwen.

## The actual missing model grammar

H273's frozen Qwen relative 0..999 output filter allows ONLY click
or abstain. It cannot produce a second-step `type_text`.
General H199/H263 native action contracts *do* allow text input,
but only **after** a separate preceding focus action. Passing a
combined click/type or directly injecting a reference input code
is not permissible. A future honest owner-private Qwen probe
requires two genuinely different model-time token restrictions.

H297 adds `h297/two_stage_form_grammar.py`:

- Stage 1 `focus`: delegates **unchanged** H273 real LMFE
  normalized click-or-abstain token filter and prospective
  H273 relative-to-native 960x720 conversion.
- Stage 2 `type`: independent LM-format-enforcer real
  `RegexParser` / `TokenEnforcer` / `TokenEnforcerTokenizerData`
  token filter for exactly one valid H199
  `{"actions":[{"kind":"type_text","text":"FIC-xx"}]}`
  (two digits, 00–99) **or** a strict single-action abstention.
  The grammar permits **all 100 fictional values**; it never
  supplies the correct code or source DOM/gold. The user-visible
  task text is the only place where the desired value appears.
- A bounded two-stage `TwoStepFormPolicyState` advances **only
  after a strict returned model action**: click allows a later type,
  abstain ends immediately, type ends after exactly two calls.
  Type-before-focus, click/scroll/key in type stage, extra actions,
  posthoc JSON repair, a third action, nonstandard encodings,
  whitespace/raw text and unknown stages are rejected.
- The original H263 actuator, not this purely model-facing
  grammar/state object, must still independently check that a
  real editable field is currently focused before executing
  `type_text`.
- The H296 source-owned scorer and all-field postaction checks
  remain unchanged. No source reference fields, original DOM
  or trained gold are supplied to the grammar.

## Exact CPU validation

```powershell
python -m unittest discover -s tests -p test_h297_two_stage_model_grammar.py -v
python -O -m unittest discover -s tests -p test_h297_two_stage_model_grammar.py -v
python -m compileall -q h297
```

Seven original adversarial source-only tests cover normalized focus
conversion, strict 100-possible-code text grammar, all phase
transitions including abstention, invalid/combined actions,
screenshot-user-task prompt restrictions, and fake tokenizer/
model-time enforcer control to ensure the **token-time constraint**
API is used, rather than posthoc JSON rewriting.

The LMFE unit test uses an **invented tiny tokenizer stub**; it
does NOT load Qwen, execute model generations or measure learned
source-task success. Future real owner-private Qwen inference
must be separately approved, exact version pinned, use only
user-visible task+fresh PNG for EACH step, demonstrate true
model-time token enforcement and H263 actual native source-state
outcomes, and publish only sanitized aggregate receipts.

## Hala research qualification remains open

Previous genuine H291/H292 FREE T4 8-step LoRA trained adapter
runs measured **ZERO source-owned paired GUI success gain**
on already exposed original Queueboard/RouteBoard.
H293 FREE T4 24-step run terminal ERROR; private output
retrieval was blocked by tool safety, so its optimizer/model
score/failure stage are UNKNOWN. Never bypass that block.

H199 officially admitted TRAIN manifest is still EMPTY;
H173 old scored OOD consumed/sealed, H212 scored cases
prohibited, H250/H251 third-party media quarantined.
FieldPort is a new PUBLIC/EXPOSED original source, not an
independent never-seen blind OOD. Genuine source-disjoint
adapted model skill/safety/abstention across independent
rights-qualified data remain prerequisites for release.

All GitHub PRs DRAFT, no merge/deploy/release, no Mac,
no paid GPU inference, no model weights or private source data.
