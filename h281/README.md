# H281 — original bilingual ClinicGrid native research family

**A third source family, not a new blind OOD test.** The fully synthetic,
standalone HTML/CSS/JS interface was written for Hala and uses fictional
visit IDs and invented service descriptions. It has no network dependencies,
patient records, appointments, real credentials or external media.

H281 differs from the public original Stockroom (inventory list) and
Queueboard (ticket/priority list) interfaces: it uses a **three-column
two-row responsive-internal card grid** within a fixed 960×720 native
browser viewport, with English/Arabic RTL language control and an
urgency filter. Only a single native click or explicit abstention is
required by each source-owned task. No OCR-derived labels, absolute
model target points, DOM or source code enter a model callback.

## Frozen two-language source-owned research tasks

| Scene | GUI | User instruction | Source visibility |
| --- | --- | --- | --- |
| `en_urgent_c214` | English, urgent-only filtered view | `Approve fictional visit C-214` | Only the two urgent visit cards rendered |
| `ar_all_c308` | Arabic RTL, unfiltered view | `اعتمد الموعد الخيالي C-308` | All six fictional visit cards rendered |

The original source is `h281/clinicgrid.html` with reviewed pinned
Git blob `dacdfaf5c4b9fe9ed22624109c917f7e72bf4edb`;
`.gitattributes` preserves identical bytes on Windows/Ubuntu.
The evaluator's fixed-source all-six visit oracle is separate from
the model-style `predict(png_bytes, task_prompt)` callback. It checks
the target visit becomes approved and *all other five* remain
unchanged, including four invisible under the urgency filter;
the visible counter must increment **exactly once**. An actually
executed wrong-visit click is a FAILURE, not a partial success or
an implicit oracle-guided retry.

`h281/clinicgrid_native_research.py` fail-closes for unexpected source
bytes, symlink/path drift, changed viewport/source URLs, unknown
task/target bindings, duplicate or incomplete source-owned visitor
states, raw malformed native action output, stale screenshot, browser
network, or an unauthorized second action. The screenshot is captured,
checked, and consumed only in memory; it is never saved. Source
oracle/coordinates must not be leaked into actual pretrained policy.

## Tests

Run synthetic source-only tests with:

```powershell
python -m unittest discover -s tests -p test_h281_clinicgrid_native_browser.py -v
```

Run **real browser** tests with an explicit environment opt-in:

```powershell
$env:HALA_H281_BROWSER_QUALIFY='1'
$env:HALA_H281_BROWSER_CHANNEL='msedge'
python -m unittest discover -s tests -p test_h281_clinicgrid_native_browser.py -v
```

The dedicated GitHub Actions workflow runs real Chromium on Ubuntu
and real Edge on Windows with pinned public browser dependencies.
Tests include correct English/Arabic fixed source fixture clicks,
wrong-neighbor click failure, abstention, malformed native coordinates,
stale-screenshot rejection, full all-visit state and task identity
guards. Source-derived clicks in these tests are **test-only
fixtures**; the tests NEVER call Qwen and must never be reported as
model-native task accuracy.

## Scientific and release limits

ClinicGrid is a public first-party research fixture, so it does
**not** establish independent hidden out-of-distribution
generalization. No declared TRAIN/DEV admission or rights review
beyond author-owned fictional fixture status, no model checkpoint
download, no LoRA optimizer update, no paired pretrained/adapted
evaluation and no release are authorized by this source test.
H199 admitted TRAIN is still empty, other prior scored/third-party
sources are subject to their own sealed/quarantine gates.
Preserve draft-only workflow, zero-cost processing, no user media,
and no Windows/Mac local model weights.
