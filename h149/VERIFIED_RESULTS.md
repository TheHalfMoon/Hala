# H149 — owner-only original browser event diagnosis

## Fresh genuine native React evidence

On **16 new original synthetic source tasks >=134000** from the unchanged
MIT React Kanban implementation (4 AR/Doing, 4 AR/Done, 4 EN/Doing,
4 EN/Done), the H147-style screenshot-only mouse controller received
only an exact visible source-card center and a deliberately noisy
visible heading point. Four browser-only shifted layouts were covered.

Owner-only independent capture listeners recorded true HTML5 drag events
**after** mouse-up to explain error mechanisms. They never supplied an
event type, target element, CSS class, native hidden drop rectangle, or
label to the mouse controller. Real task persistence was checked by
reloading and reading original application state.

| Measured real browser property | Cases |
|---|---:|
| dragstart event | 16/16 |
| dragenter event in intended column | 16/16 |
| screenshot pixel-change feedback | 16/16 |
| genuine drop event in destination column | **6/16** |
| requested task changed and persisted across reload | **6/16** |
| drop event but no persistence | **0/16** |
| persistence without destination drop event | **0/16** |

These are **source-reference diagnostics, not neural scores**.

## Failure mechanism observed in private event log

Each failed attempt had an intact initial dragstart and entry into the
intended column, but the pointer ultimately left the accepting native
drop area at mouse-up. The final browser events show a dragleave of
`drop_area` / column and a dragend WITHOUT a drop event. Successful
examples show `drop` on `drop_area` immediately before dragend.

The original upstream `DropArea.jsx` only invokes application
`onDrop()` on a genuine HTML5 drop event and switches between
`hide_drop` / `drop_area` via DragEnter and DragLeave. A pixel-change
signal alone is insufficient: it was present on all 16 cases even when
mouse-up missed a native drop zone.

**Conclusion:** The current computer-use bottleneck is maintaining
pointer location on an active native drop target THROUGH mouse-up,
rather than merely finding the correct destination column. A new H150
experiment should use independent task IDs and pixel-observed active
drop-area geometry to choose an actual safe interior mouse-up location;
it must never reveal target DOM geometry to the controller. Validate
real task persistence on NEW cases, not H143–H149 consumed episodes.

No LoRA changes, model inference calls, paid compute, opening of sealed
H140 BLIND, public weights, flagship Hala 35B changes, merge or release.
Private per-event receipts are owner-local at
`private-h149/native-event-diagnostic16-v1.json`.
