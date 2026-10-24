# H115 — original strict GUI evaluation held out from new point LoRA

This is a **frozen future evaluation design**, not a successful model
result. H114 is a new private free-T4 training candidate on original
H71 TRAIN 128 source screenshots with two independent START/END role
examples each (256 optimizer updates). Its actual model checkpoint
must be owner byte-verified before H115 can allocate another GPU.

## Strict trial

Use original pinned Hcompany/DragOn public EVAL **indices 60..75**
per original domain (slider, sheet, slide_resize, text_highlight),
**64 original source screenshots, previously unseen by H93-H114**.
Never mix EVAL with H114 original H71 TRAIN or training selection.

Two paired arms on the SAME original screenshots:
1. Original byte-verified H106 frozen adapter produces full-drag pixel
   JSON in a single inference; original unchanged H61 strict scorer.
2. H114 owner byte-verified FRESH 256-step dual-point adapter receives
   the same original screenshot twice but with the independent START
   and END source-intent role prompts used during H113 TRAIN. Each
   must output exact native JSON with only role, point and normalized
   coordinate_space. No syntax repair, no inferred missing endpoint.
   Strict normalized image-coordinate point values convert into valid
   original screenshot pixels; compose full-drag JSON and apply the
   exact same untouched original H61 scorer.

Report actual native strict JSON, start-box and end-box hits, full
drag tasks, paired-source confidence and extra inference cost.
H114 uses 2 model calls/screenshot versus H106's one. These are public
EVAL diagnostic cases, not hidden TEST or deployment validation.
All actual neural outputs/screenshot paths stay owner-private.

**HARD BLOCKER:** H115 evaluation code is fail-closed with unset owner
verified H114 adapter/checkpoint SHA. A finished GPU job by itself
cannot unblock this. Only after actual H114 training receipts and
Windows owner independent checkpoint/optimizer SHA evidence can the
new exact SHA be committed in a NEW source-pinned evaluation version.
No merge, weights publication, paid resources or release.
