# H227 — attest five previous original-app Git repository + SHA pairings

**DRAFT ONLY**, stacked on H226 PR #247 exact head
`d70794524ca4d4b656c1b3b6ded9e63bf34699e5`.
Protected main unchanged, no merge, release, deployment, Mac, GPU,
training, historical owner images, model output, consumed H173 OOD or
private gold used.

## Actual H176 false acceptance demonstrated on the pinned parent

On owner Windows 11 using **only a fabricated test metadata fixture**,
the H176 historical attribution validator accepted a substituted repo
`fabricated-owner/fake-replacement-upstream` for prior historical
`react-dndkit-board` while preserving its frozen
`f3e129998e3140a386f1884faff314345e0eb10b` Git SHA. By rewriting
both of the supplied historical (unpaired) repository lists to the fake
name, the old check returned
`H176_PAIRED_HISTORICAL_METADATA_ONLY_NOT_ADMISSION`.
This demonstrates that previously H176 pinned the historical Git SHA
but **not the old app's actual original upstream repo**.
It does **not** demonstrate real image leakage or official TRAIN approval.

## Independently verified public GitHub commit/repository identities

Each repository's exact pinned SHA was checked via GitHub REST
`GET /repos/{owner}/{repo}/commits/{sha}`, no source cloning or
historical owner media. H176 pre-existing source-code pins were reused.

| Previous app | Verified canonical upstream GitHub repository | Previously pinned full Git SHA |
| --- | --- | --- |
| `react-dnd-kit-board` (H138) | [MSabriBoy/Kanban_Task_Board](https://github.com/MSabriBoy/Kanban_Task_Board/commit/553d4e92be3e86e0deb7045f4968152868399eae) | `553d4e92be3e86e0deb7045f4968152868399eae` |
| `react-html5-board` (H138) | [9bzero/kanban-board](https://github.com/9bzero/kanban-board/commit/83413e1f5fab4690bb46b23201a3438a427ef265) | `83413e1f5fab4690bb46b23201a3438a427ef265` |
| `react-dndkit-board` (H140) | [TomCani/kanban-dndkit-localstorage](https://github.com/TomCani/kanban-dndkit-localstorage/commit/f3e129998e3140a386f1884faff314345e0eb10b) | `f3e129998e3140a386f1884faff314345e0eb10b` |
| `todomvc-es6` (H158) | [tastejs/todomvc](https://github.com/tastejs/todomvc/commit/ff43b02e59dfa604386bb382034b2cd07c2bcd8a) | `ff43b02e59dfa604386bb382034b2cd07c2bcd8a` |
| `gridstack-original-column-demo` (H163) | [gridstack/gridstack.js](https://github.com/gridstack/gridstack.js/commit/b05b79dcbca21caca046c1e8c4da6e59546ec806) | `b05b79dcbca21caca046c1e8c4da6e59546ec806` |

The H138 commits were additionally resolved using GitHub's indexed commit
search; H140/H158/H163 had source references already recorded in Hala.
All five commit objects exist in these exact public upstreams.
Public upstream reachability does NOT itself attest what old owner
actually ran or that every historic donor has been found.

## Code and tests

`h176/source_family_attribution.py` gains `FROZEN_OTHER_REPOS`
five immutable canonical repository family names, and rejects an
historical app with a correct Git SHA attached to a different upstream
family. This check is applied after the pre-existing commit SHA check
and before permitting any historical source-only metadata receipt.

`tests/test_h176_source_family_attribution.py` uses only fabricated
source lists, Git/HTML hashes and no previous media. Updated its
positive fixture to the five independently checked public upstream
identities, added 4 tests covering:
- each of the five original app rows substituted with a fake repo while
  both unpaired source-history arrays are rewritten consistently;
- conventional `.git` clone aliases of each correct original repo
  (allowed as the **same** family, never a distinct donor);
- swapping two historical repo labels with matching unchanged frozen
  Git SHA lists (rejected);
- exact known app-to-upstream registry mapping regression.

Owner Windows pre-CI: **28/28** stdlib H176 tests normal and
**28/28** under `python -O`, total 56. Hosted H227 matrix
Linux and Windows Python 3.11/3.12 needs independent exact-head
success and existing cpu-qualification must pass before green.

## Strict scientific limitations

This proves five **known** original public source repository+commit
pairings are required by H176. It does **not** prove old historical
source-family/PNG/RGBA registry completeness, the owner actually ran
these public checkouts, authentic old app labels, independent TRAIN or
DEV admission, fresh reserved OOD, model performance or authorized
gradients. H199 official TRAIN allowlist stays EMPTY; H212 96
research-only candidates remain permanently barred from optimizer
training; original H173 was evaluated once and is permanently
consumed (2/32 baseline to 3/32, exact paired McNemar p=1, not
statistically significant). No model operations or publication.
