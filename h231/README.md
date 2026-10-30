# H231 — Owner-private source-pinned native GUI replay (research only)

**Status: Windows local browser smoke verified, not dataset or model admission.** This
continues the existing H130/H137 offline Playwright pattern rather than creating
another simulated oracle score. No model weights, H173 owner OOD media/gold, H212
training examples, GPUs, private patient information, hosted inference, or paid
services are involved.

## Real advancement

`native_replay.py` accepts one **new** local original-Git working tree and a
manifest for up to 16 bounded, authored browser actions. It requires:

- Exact local Git commit and clean checkout; hash-pinned HTML and root license
  bytes (a license-file hash is NOT a complete license/dependency audit)
- Existing H173/H176/H137 known public historical source-family exclusions
- Relative, in-root, non-symlinked HTML/license files only
- Browser file URL rooted in the local checkout; network and out-of-root
  subresource requests are aborted and counted
- Real native Playwright click / text fill / HTML drag operation with a DOM
  `data-*` outcome assertion, not a fabricated provider or neural prediction
- Fresh screenshots before and after each action, SHA-256 of PNG and
  width/height-committed decoded RGBA bytes
- **Create-only** output outside the source tree and outside any Git checkout;
  any incomplete run carries `INCOMPLETE_NO_ADMISSION`

This is *authored-action/oracle native QA*. It is not proof of model ability,
donor historical disjointness, screenshot license clearance or training rights.

## Local Windows reproduction (Python 3.11+)

Run from the Hala repo root. A local Playwright module and an already-installed
Chromium or Microsoft Edge are required. No browser download is performed.

```powershell
python -m unittest discover -s tests -p test_h231_native_replay.py -v
python -m h231.native_replay --manifest D:\private-h231\candidate.json --source-root D:\approved-original-donor --output D:\Hala-H231-Private\new-capture
```

`candidate.json` has precisely these fields:

```json
{
  "schema": "hala.h231.private-native-replay.v1",
  "repo": "independent-owner/previously-unused-app",
  "git_sha": "0123456789012345678901234567890123456789",
  "license": "LICENSE",
  "license_sha256": "exact 64 lower-case hex of local LICENSE",
  "html": "examples/index.html",
  "html_sha256": "exact 64 lower-case hex of local HTML",
  "cases": [
    {
      "id": "new-click-en", "lang": "en", "action": "click",
      "selector": "#button", "destination": "", "text": "",
      "expect_selector": "#state", "expect_attribute": "data-done",
      "expect_value": "true"
    }
  ]
}
```

The README example is an illustrative schema, not pre-approved data or a
verified source. The tool deliberately fails if paths, SHAs, source history
minimum, action expectations or browser isolation differ.

## Candidate repository research, not datasets

At these **public Git source commits** the repository license root file
existed. All three still require transitive dependencies, media, actual
native application build and full historical-origin review before any use:

| Research candidate | Commit | Root license | Possible role (not admitted) |
| --- | --- | --- | --- |
| `medplum/medplum` | `33721124c9d7a1b3bcb728820a32c62e375ccec8` | Apache-2.0, `LICENSE.txt` | Synthetic-only FHIR UI TRAIN research |
| `marmelab/react-admin` | `7845c813503898db5ec25365855f84ad7d070e56` | MIT, `LICENSE.md` | Demo CRUD DEV research |
| `tabler/tabler` | `096491dea4886f78891c2e29310351d5aee6c460` | MIT, `LICENSE` | OOD_RESERVED source research, NO test images yet |

A search on Hala default branch found no textual matches for these names,
which **does not establish absence from past private experiments**. Tabler
may include ApexCharts and other differently-licensed assets; exclude those
until independently checked. Medplum sample health records may not be
synthetic; use strictly fake records, never patient data.

## Absolute non-admission boundary

This code intentionally always returns `training_approved: false`,
`ood_evaluation_approved: false`, `model_calls: 0`,
`optimizer_steps: 0` and `complete_historical_source_and_rgba_proven: false`.
It performs no training, copying donor assets, network fetch, OOD evaluation
or upload of owner-private screenshots.

H199 official TRAIN allowlist is still empty; H212 remains permanently blocked
from gradients; H173's consumed 32-case OOD stays sealed. To train real Hala
Qwen3-VL: first independently establish complete old original source/fork/npm
and decoded-RGBA history, review candidate assets/licenses, verify genuinely
disjoint newly replayed apps and recorded outcomes in a private owner workspace,
freeze clean TRAIN/DEV/OOD splits, and pass the existing H174/H176/H177/H199/
H222 admission firewalls. Only then consider free authorized compute, LoRA
training and once-only new OOD evaluation.

Repository remains PRIVATE; main SHA is left untouched. GitHub private-repo
branch Rulesets returned 403 on current plan; no paid upgrade or public revert.
Keep new code in a distinct Draft PR and do not merge or release.
