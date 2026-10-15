# Colab provider detection delta review

**Verdict: pass for this narrow diagnostic repair; no actionable finding.**

Reviewed snapshot: `work/colab-provider-review.diff`, independently collected against the prior immutable training snapshot. SHA-256: `d673b6997050ad773e73f06d0011c2713ad21e00196383f73e521a044cf21c23`.

| Bar | Result | Evidence |
|---|---|---|
| Correctness | Pass | `training/scripts/baseline.py:60–65` rejects a truthy Kaggle runtime marker regardless of Colab markers or directory presence. A `/kaggle` root without a Colab release marker remains blocked; the observed Colab release marker permits the overlapping root. Detection does not depend on `/kaggle/working`. |
| Correctness, integration | Pass | Baseline calls the helper before torch/runtime imports (`:71`). Trainer calls the same helper (`train_adapter.py:73`) and still independently rejects `config.platform == 'kaggle'` (`:71–72`). Independent intercepted execution confirmed observed Colab passes these provider guards, while Kaggle marker/config and unknown-root cases fail before runtime/model work. Linux and explicit remote opt-in guards remain earlier in both entry points. |
| Product | Pass | This corrects the supplied actual Colab observation: Colab markers present, Kaggle marker absent, `/kaggle` present, `/kaggle/working` absent. It preserves the inspected Kaggle commercial-work exclusion. Provider markers are deployment diagnostics, explicitly not adversarial provider attestation. |
| Parsimony | Pass | One small shared helper replaces duplicated directory checks. The notebook remains derived from the canonical script; no dependency or provider abstraction was introduced. |

Verification: **11/11 relevant CPU tests passed** (4 provider tests, 3 baseline tests, 4 training/preflight tests). The independent helper subreview also passed the 4 provider tests and 6 additional marker/root combinations. Additional entry-point probes stopped before importing runtime/model code. The regenerated notebook's final code cell exactly equals the canonical baseline with its opt-in prefix, compiles successfully, and contains no stored execution output. No source edits or remote actions were performed by the reviewers.

Reviewed source SHA-256 values:

- Baseline: `cd18a3b55a978cd760d4b033b80f95e147e90d0e80a30112e8ed2faabe36a8e3`.
- Trainer: `4348e9f3f08d22f85bd9f5424c485abff20c085b6982756b23c865c5b1a5db5b`.
- Provider tests: `9736ea3a0b9ba39a8253d42912ef25786ad6d211773a323777451f49ccde6ecf`.
- Colab notebook: `bc33c2085f4fdffcf1835ce76cf7ede13dfb2aacac190afff12379c5dfbf467a`.

Limits: supplied T4 allocation/tensor-probe evidence is context, not a qualification result from this review. No actual processor, foundation-model inference, LoRA training, adapter quality, model fit or resumed numerical equivalence is qualified. No local foundation weights or remote jobs were used. This delta does not turn environment markers into a security boundary.

panel: narrow delta — correctness ✓ independent source/tests/integration probes · parsimony ✓ combined inline bar · product ✓ observed deployment diagnostics · security — no new authorization boundary; model qualification remains separate

## Font setup follow-up

**Verdict: pass for the dependency setup addition; no actionable finding.** The supplied actual Colab run loaded the model but stopped on a missing DejaVu font before inference. This review covers the setup repair, not that model load or any inference result.

`training/platforms/build_notebooks.py:28` retains pinned `transformers==5.16.1`, then runs fixed list arguments for `apt-get -qq update` and `apt-get -qq install -y fonts-dejavu-core`, both with `check=True` and no shell. Setup precedes the canonical baseline cell. Independent mocked execution verifies command order, update failure stopping installation, and install failure propagating. No apt or pip command was executed by the reviewers.

The regenerated notebook setup cell exactly matches the generator literal; its Python portion compiles. Its later baseline cell still exactly matches the canonical baseline plus opt-in prefix. There are no stored execution outputs. The fixed font package addresses the supplied Debian/Ubuntu Colab dependency gap without a new abstraction or paid-platform action.

Reviewed generator SHA-256: `8ebd54ff6d4c16b76c51e8e5a7ddd591ae21b2111eef2b0ec13538c6dc931d2d`.
Reviewed regenerated notebook SHA-256: `e7752f92b1b06d3c435cad153d9709f90cf90e2c8c57882f9ab881459d0c3832` (supersedes the earlier notebook hash for this follow-up).

Official repository source is supplied remote-runtime context; remote repository configuration and installation success were not verified by this reviewer. No font rendering, actual processor/model inference, LoRA step or model qualification is established by these mocked setup checks. No source edits or local foundation weights.

panel: setup delta — correctness ✓ independent source/parity/failure probes · parsimony ✓ fixed dependency addition · product ✓ supplied missing-font cause · model qualification — absent
