# H237 — repair historical original-source TRAIN/DEV exclusion floor

**Source-only historical crosscheck; NOT a dataset or model release.**

## Scientific finding that triggered this repair

The prior H231–H236 handoff treated H232's Awesomplete upstream as a novel
source candidate. This was factually **incorrect**: existing public Hala
documentation shows four original app source families had already been used
in TRAIN and/or consumed DEV long before H232:

| Upstream GitHub family | Historical use | Frozen commit in public documents |
| --- | --- | --- |
| `Choices-js/Choices` | H168 TRAIN, H169 same-family consumed DEV | `e132628ac74dd427a9a454caf042c65d3065d170` |
| `leongersen/noUiSlider` | H168 TRAIN, H169 same-family consumed DEV; H137 npm source | `57033672d07e8bac527d7147721a4a2e07e0d699` |
| `flatpickr/flatpickr` | H171 TRAIN, H172 consumed DEV | `548330e6e18871d6ca1ca78746c0e89cb12a0303` |
| `LeaVerou/awesomplete` = `nudeui/awesomplete` | H171 TRAIN, H172 consumed DEV | `596bc5d5916266437970ecbf65434077d3950a3d` |

The two Awesomplete URL strings point to the SAME GitHub repo ID
`30987756` after transfer. They cannot be separate novelty families.
This follows from the repository's public original history docs:
`h168/README.md`, `h169/README.md`, `h171/README.md`,
`h172/README.md`; no owner-private media/gold or old OOD was read.

## Implemented boundary

- `h174/split_firewall.py` freezes the five source-name/alias variants in
  `FROZEN_PRIOR_ORIGINAL_REPOS` and **rejects** any proposed
  TRAIN, DEV or OOD_RESERVED source with these GitHub families, regardless
  of a newly invented app ID, Git commit or Git `.git` clone suffix.
- `h231/native_replay.py` imports this same floor in the offline source
  acquisition preflight, so renamed historical donors cannot be promoted
  as novel candidates even with a matching Git remote and freshly
  authored metadata.
- `h236/original_dragula_native.py` reuses the same shared history floor
  rather than hardcoding only Choices.js; its original Dragula QA
  remains research-only.
- H232 remains runnable for **historical library diagnostics**, not
  candidate data admission. Its documentation is corrected to acknowledge
  H171/H172 past Awesomplete usage and the now-superseded private CI note.
- `tests/test_h237_historical_source_floor.py` makes old source reuse
  mutations and allowed synthetic distinct-source cases explicit. It never
  loads old owner media, training images or model predictions.

## Limits and required next evidence

These are **publicly known minimum exclusions**, NOT the complete historical
TRAIN/DEV/OOD history. Other old original Git/fork/npm sources and all
previously consumed image/decoded-RGBA history must still be independently
reconciled from existing source/metadata ledgers before ANY new source is
admitted. No future source is qualified by merely not appearing here.

No original H173 OOD rerun, no H212 optimizer gradients, no patient data,
no use of old H168/H171 private media or reference actions. The H199
`ADMITTED_TRAIN_MANIFEST_SHAS` is still empty. Zero inference calls,
GPU work, training steps, dataset promotion, model release or merge.

H237 is a public code/docs/tests Draft only. Keep all owner-private
captures, gold and decoded-RGBA receipts outside GitHub.
