# H144 — native original drag controller A/B, new cases only

H143 froze 16 original CSS-shifted React evaluation screenshots. The
external pretrained Qwen3.5-2B H142 LoRA achieved 16/16 START and
15/16 visible destination-heading hits, yet only **8/16 actual native
browser tasks persisted after reload**. Those cases have already been
consumed once and are **not** retested, tuned, or relabeled here.

H144 creates a NEW 16-case owner-only original MIT React QA cohort at
attempt IDs **82000+**, with shifted columns and balanced Arabic/English
and Doing/Done tasks. Each attempt is tested in TWO independent fresh
browser contexts. Source and destination references are visible DOM
card/heading centers given to both controllers. Neither controller is
provided any invisible drop DOM box. All cases use unmodified original
application source revision
`fa047f7a32afb57750ed5246456dc282ea7befcd`.

Paired action policies:
- **Baseline:** move the pressed card to 12 pixels below the visible
  heading's nominal bottom border, then release.
- **Staged sweep:** drag near the heading, travel through several
  below-heading offsets, capture intermediate browser screenshots,
  then release. This policy reads no invisible target DOM.

Actual owner Windows native outcome after full reload:

| Preregistered policy | Persisted requested task success |
| --- | ---: |
| Baseline visible-heading reference | **16/16** |
| Staged sweep reference | **16/16** |

All 32 real native task drags persisted. Both succeeded together in all
16 pairs. Staged improvement cases **0**, regressions **0**. 4 Arabic
Doing, 4 English Doing, 4 Arabic Done, 4 English Done tasks were used.
Four shifted layout styles were represented (down 4, far_down 3,
left_up 4, up 5). The source collector rejected four other candidates
that could not qualify both arms.

**Interpretation:** This result does **not** show staged-sweep
superiority. It instead narrows the failure mode: the baseline actuator
is successful with perfect visible source/heading centers in these
scenes; H143's 8/16 result involved imperfect model points even though
15/16 were technically inside bounding rectangles. Point-in-box hit
alone is therefore insufficient to predict robust drag success.

**Next phase:** If improving action reliability, preregister a NEW
independent cohort with **bounded imprecise visible-point proposals**
around the true visible card/heading references, including edge-of-box
perturbations, and compare robust action control under those errors.
Do not calibrate action changes by retesting H143 consumed examples.
Then later independently evaluate frozen neural predictions in a new
cohort. This source-only experiment itself used **zero model calls and
zero optimizer steps**.

No H140 sealed BLIND neural calls, paid compute, Hala flagship 35B
update, merge to main, public weights, deployment or release. Individual
case evidence is in owner-private
`private-h144/paired-reference-action16-v1.json`.
