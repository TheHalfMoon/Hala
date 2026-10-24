# H114 — source-grounded independent dual-point LoRA pilot

H107–H112 found no repeatable full GUI drag success for the current
Qwen3-VL-2B research model. H106 joint visual/language SFT learned
JSON serialization far better than spatial localization. A pilot 1/16
in H110 did not replicate on new H111 EVAL (0/32 each).
H112 prompt-only relative coordinate conversion also had 0/16 strict
successes across four arms after owner-independent source replay.

H113 now actually qualified a different original TRAIN-only target:
512 immutable H71 source screenshots selected by the accepted H105 plan,
1,024 separate START and END normalized-point examples; source original
media SHA checked on private Kaggle, source/point-pair roles and complete
derived private JSONL independently replayed by Windows, zero quarantine.
128 Arabic and 384 English source screenshots, five original families.
No EVAL/TEST, invented center labels or public raw data.

H114 is a NEW opt-in training objective, not an update to H106.
Its original H71 curriculum/manifest/image SHA and frozen base remain
unchanged. The point overlay modifies only the SFT prompt, completion
and role-specific record ID for the model while retaining the original
media path, SHA, language, family, image format and ownership.

The frozen pilot uses the first 128 original source images from the
original accepted H105 plan, each with one START and one END task:
256 distinct role-based optimizer examples, 128 of each endpoint,
32 Arabic +96 English source images and all five original families.
Fresh rank8 FP16 joint vision/language LoRA (same bounded layer scope
as H106) on exactly one private free T4; original coordinate-digit
weight=8; no initialized H99/H106 adapter or model continuation.
Actual processor mask, source identity, gradient/update count, full
private optimizer SHA and adapter bytes must pass independent readback.

The new point-specific objective nonce, exact owner-private H113
derived-data SHA256, chosen anonymized source-schedule SHA, original
H71 manifest and new optimizer config become the checkpoint identity.
No previous training path is silently altered by default flags.

NEXT EVAL GATE, NOT YET EXECUTED: new original upstream DragOn public
EVAL indices 60..75 per four domains = 64 fresh source screenshots.
Compare original H106 full-drag model with freshly-trained H114:
H114 must localize START and END independently (two model calls),
convert its strictly valid normalized point answers into original
source pixels, compose original H61 JSON and score with the immutable
native strict H61 scorer. Report start/end hits, complete task success,
JSON failure, paired statistics and additional inference cost.
Never train on EVAL/TEST nor publish trained weights automatically.

Training completion/lower loss/valid point JSON DO NOT count as GUI
improvement. Keep Draft, private, unmerged, no public release, hidden
TEST, paid resources, or deployment.
