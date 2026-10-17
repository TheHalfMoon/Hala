# H24 - independent bilingual synthetic development candidate

**Status:** H24 source and data-projection engineering only. Not model training,
HalaType visual SFT, validated Arabic shaping, an actual HalaMind baseline,
a completed closed-loop task benchmark, an H23 CUDA qualification, or production
Orcel integration.

**Data source and independence:** entirely original, rights-declared synthetic
task cases authored specifically for H24 by the deterministic generator
`h24/data.py` (seed 20261017; revision
`hala-h24-prospective-20261017-v1`). No benchmark transcripts, scraped
screenshots, private records, previous H22 TEST trajectories, existing H19
episodes, or prior model answers were copied. Earlier H22 TEST results were
used only to *motivate* the need for stronger Arabic and safety testing,
not as training labels or examples. No TEST-driven LoRA update is performed.

## Frozen candidate manifest

- Schema: `hala.h24.isolated_bilingual.v1`.
- 100 synthetic cases in **ten authored application families**, with two
  languages (50 English, 50 Arabic) and five scenario kinds per family.
- **Train**: 60 cases in six families (parcel dispatch, facility booking,
  volunteer roster, travel itinerary, research notes, maintenance queue).
- **Development**: 20 cases in two never-trained families (purchase
  requisition, classroom attendance).
- **TEST**: 20 cases in two remaining never-trained families (subscription
  alerts, library lending). TEST is frozen for a later, independently
  qualified candidate, not an iterative tuning target.
- Scenario kinds: safe UI selection (propose only), stale frame
  (request fresh observation), missing destination (ask the user to clarify),
  hostile **page** instruction (ignore the untrusted page and continue
  safe observation), and unsafe **user** system-command request (abstain).
  This distinction is intentional: an injected instruction appearing inside
  a page must never gain Orcel authority or force compliance, but must not
  necessarily block unrelated safe user work.
- Every case has exactly one legitimate target and a similarly named
  protected distractor in an 800x600 synthetic layout, Arabic layouts
  mirrored horizontally, bounded coordinates, explicit roles and
  original-procedural-synthetic CC0-1.0 metadata. No actual screenshot
  is present.
- Bilingual NFKC/whitespace-normalized goals must remain unique; all case
  IDs, labels, rights, splits, frame metadata and expected decisions are
  verified by full regeneration from audited source rather than accepted
  from arbitrary externally supplied manifests.
- Only **60 prospective HalaMind text rows** are exportable for consideration
  in a future training protocol. No DEV/TEST rows, source split, expected
  completion label or Type image payload are included in the train prompt.
  These are not fed to the existing H21 trainer automatically.

## Exact bytes

| Archived file | SHA256 |
| --- | --- |
| `development/h24-prospective/cases.json` | `391d0dc925790558fd5d58f6c0e6348c7c7984b05891fbbd9ae956c9564572de` |
| `development/h24-prospective/manifest.json` | `5752e9b824b45356d18f5de7c9281d932771e8a2a18c00655506ce8d52dd1dc2` |
| `development/h24-prospective/train-mind-prospective.jsonl` | `22b37e04bbe2a9281d675922087355c30482efe1126292af4140f8a7f197fd96` |

Regeneration: `python -m h24.data --output NEW_EMPTY_DIRECTORY`.
Verification: `python -m unittest tests.test_h24_data tests.test_h24_archive -v`.
The CLI refuses to overwrite existing data.

## Critical limitations and next milestones

1. These are **five templated case kinds** across ten authored families,
   not 100 independently sampled real-world app workflows. Both languages
   must be human/independently checked for semantic equivalence, RTL details,
   ambiguity, legitimate abstention and safety. Synthetic control geometry
   and `protected` annotations are not actual trusted Orcel permissions.
2. `observation.expected_current_frame` is **prospective trusted
   environment metadata for a stale-frame routing decision**; it cannot be
   taken from hostile page text or be treated as a permission token.
3. **No HalaType images exist.** Before generating and admitting Type examples,
   render synthetic observations on an actually qualified Linux image pipeline
   with pinned DejaVuSans bytes, Pillow and RAQM/BiDi shaping. Capture each
   PNG with independent source/frame/coordinate/rights hashes, inspect
   visually, and require a truly held-out app-family TEST for HalaType.
   Do not silently use Windows Arial or unqualified reshaped Arabic text.
4. **No performance claim.** Independently measure unchanged base-model
   HalaMind decisions on the new DEV families, inspect errors, then propose
   and explicitly review a small free-GPU training experiment; retain
   durable checkpoints and independent comparisons. Do not use these new
   TEST labels for training, prompt tuning, model selection, loss balancing,
   or data quality iteration after first prospective freeze.
5. H23 actual CUDA cancellation and worker restart are independent
   engineering/remote-hardware admission gates. The successful H22 V4
   eight-episode benchmark is not an H23 lifecycle qualification.
6. The current target platform is **Orcel**. All Hala model output remains
   untrusted proposed work; Orcel's `orceld` alone evaluates permissions,
   executes local actions and verifies postconditions. Do not modify the
   Orcel repository from this H24 scope.
7. Signed code review, local/full suites, Jev, Graft and independent PStack
   review are separate; advisory semantic probabilities do not imply
   approved model safety.

**Release gates remain closed.** Founder controls PR merge, any GPU beyond
the already-authorized free research environments, and production adoption.
