# H308 — model-visible RGB identity gate for TRAIN candidates

## Concrete integrity gap

H285's original ShiftDesk source generator proves that 24 PNG-byte and
decoded **RGBA** screenshots are distinct. H307 independently checked
compressed PNG-byte uniqueness before handing 24 samples to a private
consumer. Neither condition alone proves that all 24 samples have distinct
**RGB** pixels, the representation H208 passes to the Qwen vision
processor. Different PNG ancillary metadata, PNG re-encoding, or an
alpha-only change can produce distinct byte and RGBA digests while the
model sees exactly the same RGB image.

## Narrow fix

H307's opt-in one-shot collector now decodes every bounded original
960x720 PNG, hashes only its model-visible RGB pixels in private memory,
and refuses a repeated RGB frame **before** any external consumer call.
It also rejects malformed and wrong-viewport PNGs. The existing
24-sample count, 12 EN / 12 AR source proof, strict TRAIN-candidate-only
role, and categorical fail-closed behavior remain unchanged. No hashes,
pixels, prompts, coordinates, labels or task text are returned or
committed to the public repository. The 24 candidate screenshots remain
owner-private, ephemeral, and not officially admitted.

This is a defense-in-depth check against source or producer drift, **not**
a claim that a collision has been observed in H285's real 24 screenshots.
The new tests construct invented in-memory RGBA PNGs and adversarial
alpha-only, metadata-only, invalid-size and truncated variants. They do
not touch user media or the model.

## Reproduction (source-only, no GPU)

```bash
python -m unittest discover -s tests -p 'test_h307_one_shot_train_candidate_handoff.py' -v
python -m unittest discover -s tests -p 'test_h308_model_visible_train_frame_identity.py' -v
```

## Scientific and transaction limits

The external consumer still MUST implement its own all-or-nothing
transaction; one callback and literal True are not proof of atomic
storage. H199's official admitted TRAIN manifest remains EMPTY. The
public first-party GUI sources are exposed research, never independent
blind OOD. No LoRA optimizer step, real Qwen inference, paired adapted
gain, new private Kaggle experiment, dataset admission or release is
performed. H293/H294 safety blocks remain untouched.
