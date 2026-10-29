# H177 — known historical original-app repo pins (research-only)

**This does not produce new TRAIN/DEV/OOD, score or train a model, or qualify release.**

## Background and evidence

The earlier H176 paired owner-source check could accept a fabricated upstream
repository name for a previously used app because H176 froze certain Git SHA
values without independent app-to-upstream repository pins. A SHA-40 identifies
a commit but does not by itself validate the claimed repository name.
A future version of that same repo could therefore be misclassified as a
fresh original-app family.

H177 binds nine previously used historical apps to exact original upstream
repositories AND frozen public Git SHA values. The supporting public evidence
is from this same repository:

| Historical app ID | Verified prior upstream | Git SHA source |
| --- | --- | --- |
| react-dnd-kit-board | MSabriBoy/Kanban_Task_Board | h138/README.md |
| react-html5-board | 9bzero/kanban-board | h138/README.md |
| vanilla-native-board | anantagarwal1307/Drag-Drop-Kanban-Board | h140/freeze_train_dev.py |
| react-native-board | dandemslay/React-KanbanBoard | h140/freeze_train_dev.py |
| react-dndkit-board | TomCani/kanban-dndkit-localstorage | h140/freeze_train_dev.py |
| todomvc-es6 | tastejs/todomvc | h158/README.md |
| gridstack-original-column-demo | gridstack/gridstack.js | h163/VERIFIED_RESULTS.md |
| simple-table | arce/simple-table | h164/PILOT_VERIFIED_RESULTS.md |
| calculator | ashishxa/calculator | h164/PILOT_VERIFIED_RESULTS.md |

See h177/public_origin_pins.py for the exact Git SHA pins, copied from
those earlier frozen public evidence records. The first five cover earlier
real native drag applications, including historical TRAIN/DEV and exposed
evaluation apps; the remainder covers previous click, GridStack, table and
calculator original sources. H177 now requires the three previously missing
H140/H164 app IDs: vanilla-native-board, react-native-board and simple-table.

The already known H137 npm-only split.js@1.6.5 source remains blocked by
H176's upstream alias nathancahill/split without inventing a Git SHA.
H177 composes H176; it must never bypass the existing owner-source firewall.

## Offline owner-only run

Never put actual historical manifests, image hashes, gold, image files,
inference traces, model outputs or owner filesystem paths in GitHub.
With four existing local source-identity JSON manifests (see
h176/HISTORICAL_ATTRIBUTION_OWNER_RUNBOOK.md), run locally:

    python -m h177.audit_owner_roots --candidates C:/PRIVATE/candidates.json --prior-pixel-hashes C:/PRIVATE/prior-hashes.json --prior-source-history C:/PRIVATE/prior-source-history.json --paired-source-history C:/PRIVATE/paired-source-history.json

The command is read-only, uses the H176 capped JSON parser, prints only
aggregate counts and a source-metadata SHA-256 digest, and returns code 2
without owner source names or paths on any source-validation failure.
Actual owner manifests must contain every previously used app family.
The prior 19 known app IDs in H176 expand to **at least 22** known prior
app IDs under H177. This is only a lower bound, not completeness proof.

## Required before a new model update

1. Independently cross-reference every previous actual TRAIN, DEV and
   consumed OOD original source against its owner-private immutable source
   manifests, including old forks, npm-only apps and previously observed
   research apps. No private screenshot or gold inspection is permitted.
2. Independently confirm license, source identity, browser-native state
   changes, screenshot/decoded-RGBA unique-hash nonoverlap and frozen
   source-disjoint splits for *new* TRAIN/DEV; reserve genuinely new OOD
   source identities before training and do not inspect/capture their
   images during TRAIN/DEV.
3. Only after all admission gates: train an original fresh adapter with
   permitted free GPU and run DEV-only selection; perform exactly one
   untouched OOD after freezing both candidate and evaluation protocol.
4. Report separate baseline/adapter success counts and paired uncertainty.
   The previously consumed H173 2/32 vs 3/32, McNemar exact p=1.0,
   is not statistically meaningful improvement and must never be treated
   as fresh OOD again.

No model calls, paid compute, dataset admission, protected-main merge,
new publication or release is performed by this PR. Keep it Draft.
