# H106 — first real Hala model improvement hypothesis: joint grounding SFT

The original H105 candidate is now **empirically admitted**: 512 independent
original TRAIN screenshots, 25% AR, each source family >=10%, all 16 start
and end spatial regions, and max direction share 34.96% instead of H99 63.28%.
Actual 512/512 original media SHA/labels/H61 oracle and 18/18 independent
owner Windows readback passed. This **does not mean model quality improved**.

## H106 controlled, cost-free hypothesis

**Model:** exact pinned `Qwen/Qwen3-VL-2B-Instruct` original frozen base.
H99 trained 128 steps with ONLY eight late visual attention targets and no
language adapters, reaching 0/16 strict success on fresh development DragOn
scenes. A real drag action requires both visual localization and generation
of original-pixel JSON. Our tested change is a **NEW fresh 256-step optimizer**
that jointly trains the exact eight last-four QKV/output vision projections
and verified Qwen language-model q_proj/v_proj projections via rank-8 PEFT
LoRA, under H99's original source-bound coordinate-weighted loss (weight 8)
and pinned fp16/one-T4 research profile.

**TRAIN:** exact 256-prefix of the original 512-sample accepted
H105 SHA `ed69e0ea119c4e208ba6aa0a486cb29fb0c857e61fb278d6543152d022c40d0d`.
Reconstruct the original H71 10k from private owner Kaggle archive, verify
all source/media/labels, and rederive this exact candidate plan before GPU
allocation. Never train from EVAL or public benchmark labels, and don't
edit the original H71/H99 adapters, source labels or H61 scorer.

**Evidence:** all 256 actual selected source IDs in immutable verified
optimizer logs must match the pinned prefix; verify every actual step has
nonzero finite gradients and changed PEFT weights, separate visual/language
adapters, original H71 hashes, saved full optimizer and final adapter
SHA-256. Kaggle provider/private data/weights remain private; zero paid cost.

**Qualification is separate from training:** do NOT call model improved
because training ended, loss dropped or JSON validity increased. After
reliable owner-private GPU checkpoint and owner byte readback, perform a
pre-registered NEW source-disjoint DragOn EVAL comparison, e.g. public
indices 20–35 each of four task types (**64 unseen source scenes**, whereas
prior H100/H101 used indices 12–19). Same inference settings, true image
sizes, unchanged original strict scorer, paired base/H99/H106 arms. Report
exact strict drag action success, start/end-box hits, errors, degenerate
start=end, valid JSON, EN/AR where source labels support it.

**GO for further training** requires at least repeatable nonzero strict GUI
success and a trustworthy baseline advantage, verified across more than
one held-out batch. **NO-GO** if optimizer updates merely improve loss/format.

The H106 training path is opt-in; legacy H99 visual-only, H94 text-only,
production/flagship Holo4 and protected data remain unchanged.
Draft PR only; no merge, releases or publishing unapproved weights.
