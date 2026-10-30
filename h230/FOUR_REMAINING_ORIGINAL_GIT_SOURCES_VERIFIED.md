# H230 — all nine KNOWN original public Git repository+SHA identities required

Private nonpublishing Draft stacked on H229 PR #250
`920d8a1542af1c0ae029f1e9266782b7534376ad`.
No main merge, release, pretrained model, old owner media, gold or OOD.

## Actual missing defense

H227 pinned five historical original app upstream repo+Git SHA pairs in
H176, but the **other four** known earlier native GUI app sources were
only guarded by the higher-level H177 audit. An owner-automation
workflow calling the lower-level H176 directly had not yet independently
required those four app's canonical original upstream and pinned Git
revision. H230 brings all nine H177-known Git sources into the H176
mandatory historical floor.

## Independently verified four additional public commit associations

Every exact GitHub commit returned successfully under the expected
original public upstream on 2026-10-30:

| Previously used source app | Original GitHub repository | Immutable full commit SHA |
| --- | --- | --- |
| `vanilla-native-board` | [anantagarwal1307/Drag-Drop-Kanban-Board](https://github.com/anantagarwal1307/Drag-Drop-Kanban-Board/commit/47cfba359a71b70a6a23e511e554416aec006474) | `47cfba359a71b70a6a23e511e554416aec006474` |
| `react-native-board` | [dandemslay/React-KanbanBoard](https://github.com/dandemslay/React-KanbanBoard/commit/fa047f7a32afb57750ed5246456dc282ea7befcd) | `fa047f7a32afb57750ed5246456dc282ea7befcd` |
| `simple-table` | [arce/simple-table](https://github.com/arce/simple-table/commit/7580270d18fb764bbf437a5e5b88ea66d63fc449) | `7580270d18fb764bbf437a5e5b88ea66d63fc449` |
| `calculator` | [ashishxa/calculator](https://github.com/ashishxa/calculator/commit/8f27a92072dde9658d8d94db8918a184ed7bd416) | `8f27a92072dde9658d8d94db8918a184ed7bd416` |

H227 previously verified five other original GUI upstreams:
MSabriBoy/Kanban_Task_Board, 9bzero/kanban-board,
TomCani/kanban-dndkit-localstorage, tastejs/todomvc,
gridstack/gridstack.js. Together H176 now requires all **nine
known** exact original Git source app-to-upstream-to-revision bindings.

## Change and synthetic negative tests

- `h176/source_family_attribution.py` extends immutable
  `KNOWN_EXTRA_PREVIOUS`, `FROZEN_OTHER_GIT`, and
  `FROZEN_OTHER_REPOS` to the four remaining H177 public origins.
  The nine known app labels are now required and each must retain
  its canonical GitHub repository family plus exact original commit.
- An owner-rewritten historical repo roster with an invented repository,
  a replaced Git SHA or a missing original app fails the H176
  independent floor (even when both owner-controlled metadata arrays
  are rewritten consistently). Replacing one original source
  app's repo with another existing historical repo is rejected.
- Conventional case-only and trailing `.git` clone aliases of the
  SAME real source remain accepted as the same source family;
  they cannot make cross-TRAIN/DEV/OOD sources independent.
- H176 and H177 existing test fixtures are updated to reflect
  stronger **earlier** H176 rejection; no test silently accepts
  a forged history simply because its expected error message moved.
- New `tests/test_h230_four_additional_original_source_pairs.py`
  independently requires the exact nine H176 source pairs equal
  H177's already recorded public nine-source registry, and uses
  only fully fabricated historical metadata for adversarial tests.
- Hosted H230 CI Linux/Windows Python 3.11/3.12 in standard and
  `-O` modes runs H176, H177, and H230 tests without npm,
  model dependencies, images or owner data. The full repo general
  `cpu-qualification` is also required at the exact final PR
  head before claiming H230 hosted PASS.

## Scientific non-claims and remaining completion gates

The nine known public Git origins are an independently checked
**minimum** for prior Hala source-family metadata. Their public
GitHub commits exist in those repositories, but that does not
prove exactly what the owner previously ran or the completeness
of all historical source/fork/npm families. Original old decoded
RGBA vs new TRAIN/DEV pixel disjointness remains unproven; owner
private historical media may never be opened to re-evaluate
consumed H173 OOD. H137 npm asset publisher SHA comparison was
completed for PUBLIC registry bytes in H229, not for previously
copied owner H137 bytes.

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains empty.
H212's research-only candidates are permanently banned from
optimizer updates. The old H173 32-case owner OOD was consumed
once (2/32 to 3/32 with exact McNemar p=1, not significant),
and must remain permanently sealed. There is still **no
admitted source-disjoint TRAIN, sealed DEV, independent
new OOD, new Qwen weights, training steps or reliable model
accuracy improvement**. No merging, publishing, owner gold access,
Mac or paid compute has occurred in H230.
