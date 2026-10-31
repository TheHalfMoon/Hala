# H282 — prospective pretrained Qwen third-family GUI probe

**Preregistered CODE ONLY. No H282 GPU inference has yet run.**
This branch stacks on H281 [ClinicGrid original source PR #305](https://github.com/TheHalfMoon/Hala/pull/305).
The original ClinicGrid source is entirely synthetic, offline, publicly
exposed and author-owned. The two tasks are **NOT independent blind OOD**,
admitted training examples, adapted model evaluation or release qualification.

## Experiment frozen before inference

- Owner-private permitted **noncommercial FREE Kaggle NVIDIA T4** (one or
  two T4 devices, no paid services); run nothing on owner Windows/Mac GPU.
- Original first-party ClinicGrid source branch
  `ps/h281-original-clinicgrid-browser-research` pinned exact commit
  `ab6d4c77ee89a6abfb4375edbba4f1a2108db445`, original HTML
  Git blob `dacdfaf5c4b9fe9ed22624109c917f7e72bf4edb`.
- Pretrained **unchanged** `Qwen/Qwen3-VL-2B-Instruct`, pinned
  revision `89644892e4d85e24eaac8bacfd4f463576704203`,
  Transformers **5.16.1**, pinned H273 LMFE source; model-time
  constrained strict one-action JSON grammar and deterministic
  normalized 0..999 → 960×720 pixel conversion **before** H208 native
  parser. No oracle, reference bounding box, DOM/target coordinates or
  screenshots are given to the pretrained model as separate inputs.
- Exactly **two** prospective screenshot-to-action generations, one
  per fully fresh offline Chromium context:
  `en_urgent_c214` (English, urgent only, task
  "Approve fictional visit C-214") and `ar_all_c308`
  (Arabic RTL, all visits, task
  "اعتمد الموعد الخيالي C-308").
  Fixed seed **282**, greedy no-sampling one-beam generation,
  max 160 tokens, one click or abstention, no retry.
- The source-owned separate post-inference H281 oracle checks the
  correct fictional visit and the unchanged other five (including
  filtered invisible visits) plus approved-counter increment.
  Wrong target remains a failure. No posthoc correction.

## Integrity and privacy

Only the owner-private bounded report
`hala-h282-private-clinicgrid-native-aggregate.json` is allowed to
be downloaded via a **single exact `--file-pattern`** after the
private GPU kernel has actually COMPLETED. It holds only approved
scene identifiers, Boolean strict-format/native-click/task-state
outcomes, two actual model and grammar call counts, original-source
commit and model revision and a no-blind-OOD/no-improvement flag.
The code distinguishes:

- screenshot stability **attested** only when the normal
  screenshot-locked postgeneration executor returned a result;
- invalid model output with no source postaction oracle performed:
  both screenshot and full-state oracle attestations remain
  **unverified**, not fabricated as positive;
- genuine success only if a strict valid image-conditioned model
  output caused an actual real native click that passed the
  source-owned all-six-visit state oracle.

No model strings, click coordinates, private PNGs, gold DOM, model
weights or adapters are stored in the report, repository or public logs.
The bundle packs **one public Python source script** and private
Kaggle metadata, no attached data, models, screenshots or media;
the source/model are fetched into private ephemeral Kaggle `/tmp`
and deleted at exit. Source checkout and revision mismatch fail closed.

## GPU queue rule

**DO NOT submit H282 while the existing owner-private H279 Queueboard
GPU kernel is QUEUED or RUNNING.** That experiment has separate
version-specific evidence and must finish before any new T4 job.
This H282 project is SOURCE ONLY until authorization/availability
and an explicitly verified private owner-free GPU run.

When the existing GPU queue is clear, a future engineer can run the
independent code-only packaging safety tests (without actual weights):

```powershell
python -m unittest discover -s tests -p test_h282_private_clinicgrid_probe.py -v
python -m h282.build_private_payload --source h282/private_clinicgrid_pretrained_probe.py --destination "D:\\<unique-new-H282-temp-folder>"
```

The submitted kernel ID is
`abdulazizshehri/hala-h282-private-clinicgrid-native-t4`.
A successful submission, CPU qualification or browser fixture is
**never** evidence that Qwen has succeeded on ClinicGrid. Preserve
the no-release/no-merge/no-training-claim and privacy gates until
real GPU output and later independent unseen OOD prove benefit.
