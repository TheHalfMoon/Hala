# H202 — native action dispatch safety after H201

**Status:** Draft-only engineering. H201's source-original browser proof is
unchanged. This adds only executor safety and synthetic negative controls.
No model inference/training, dataset release, owner media reads/uploads,
consumed H173 OOD access, GPU use, merge, or release.

## Concrete failure modes addressed

H201 checked the source URL once before dispatching an action.
A click meant to focus a screenshot-visible control can navigate to
an unrelated page. Without a second guard, a subsequent `key_press`
would execute on that new page. H202 guards the exact canonical
`http://127.0.0.1:<1..65535>/snap/index.html` URL and 960x720
viewport between native events and after the final event. Alternate
hosts, userinfo, query strings, fragments, invalid or noncanonical
ports all fail before any browser input.

A drag that failed after `mouse.down()` could leave the browser pointer
held down. H202 attempts `mouse.up()` in `finally` even if the
`down` or intermediate `move` fails, or if the page navigates.
It refuses to report success after a detected page transition.
No generalized transaction/rollback is claimed: clicks already sent
cannot be undone, and races or release errors remain possible.

## Owner Windows source tests

At the local code candidate against H201 exact parent
`9637632d29b3808b34b498a4e52c92c0ae341536`:

- 9/9 new H202 deterministic negative-control tests PASS on
  CPython 3.11.15 and 3.12.13, both normal and `python -O`.
- 13/13 H201 baseline tests also PASS in all four environments.
- `compileall` of changed source/tests PASS.
- Tests use fake page/mouse/keyboard objects; they **do not**
  establish real-browser H202 equivalence, model accuracy,
  or any new source-trained policy.
- Dedicated no-cost GitHub Linux CPU matrix is the independent
  hosted gate; record its actual conclusion separately when available.

## Scientific boundary

The H199 real training SHA allowlist remains empty. H198's 20
source original GUI images and owner action-reference sidecars
remain strictly quarantined, unadmitted to TRAIN/DEV/OOD, and
untouched. Historical old RGBA and source-identity completeness
remain unproven; no newly reported Hala model accuracy is justified.
