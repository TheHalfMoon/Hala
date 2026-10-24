# H113 — Original H71 TRAIN-only two independent source-bound points

**Source data admission only. No model training, checkpoint, GUI improvement
or release claim.**

H106 trained 256 original H71 TRAIN screenshots with joint visual+language
LoRA and coordinate-weighted *full drag JSON*. H107 on 64 fresh public
DragOn EVAL screens scored 0/64 strict tasks for base, H99 and H106.
H108 added pixel grid, 0/32; H110's 512px two-ROI refinement had a lone
1/16 task success but independent frozen H111 confirmation was 0/32,
even though source-target ROIs covered 24/32 scenes. H112 tested the
pretrained Qwen3-VL 0–1000 grounding coordinate contract *without new
training*: **0/16 strict full drag successes across base/H106, pixel
and normalized prompts**, 15/16 vs 12/16 H106 native JSON validity.
Every original source, prompt, raw response and unchanged H61 scorer
was owner-independently replayed. **Changing the prompt is insufficient.**

## New H113 hypothesis and source protections

Hala may need actual supervised image-to-point localization before
full start+end joint drag action. Reusing **only original H71 TRAIN**
(without a single EVAL/TEST sample), verify each original raw source:
train split, original screenshot SHA/path, original prompt contract,
immutable original full-drag completion, exact true start/end bboxes,
original screenshot dimensions and source-bbox centers.

Each admitted original image produces two **independent point examples**:
1. locate the START endpoint on the ORIGINAL full source screenshot;
2. locate the END endpoint on that same unchanged screenshot.

Each completion contains only a **single strict normalized point**
in [0,1000], endpoint role, and coordinate-space contract. Pixel-to-
normalized-to-pixel round-trip must remain inside its original source
bbox. If quantization drifts outside, quarantine that entire source
without inventing/resizing/changing labels. No original bbox, oracle
coordinates or other target leaks into either prompt. All raw source
images and labels stay private.

The planned owner-private **CPU-only** audit re-derives the immutable
H71 source SHA `9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd`
from the attached 1.76GB original private TRAIN dataset on Kaggle, and
reuses exact previously owner-qualified H105 **512 unique-source
curriculum** plan SHA
`ed69e0ea119c4e208ba6aa0a486cb29fb0c857e61fb278d6543152d022c40d0d`.
Produces a private original-source SFT JSONL plus aggregate anonymous
source coverage/digest receipt; stores no original data in GitHub.
We MUST inspect actual qualified/excluded counts and source-language/
family coverage before planning or launching another real optimizer
experiment.

This is an **opt-in dataset representation candidate**. No change to
original H71/H99/H106 weights or training code yet; never conflate
data source integrity with learned spatial success. Any future training
requires a separate pre-registered frozen protocol, free private T4
and a truly new unseen strict GUI performance test. Keep private Draft,
unmerged, no paid resources, publication, deployment or hidden TEST.
