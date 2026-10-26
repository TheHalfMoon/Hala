# H140 candidate source registry — LICENSE + commit preflight only

**Status: research candidates, not admitted H140 screenshots.**
Checked against public GitHub source on 2026-10-26. Each listed repository had an
explicit `LICENSE` file beginning with the MIT License text and original copyright.
README/package documentation describes an application with local task state and
drag-and-drop. Git commit SHA below is an exact reproducible SOURCE candidate pin.
This does **not** mean the app was cloned, built, run offline, or passed the
actual persisted-state reference drag. Those acceptance gates remain mandatory.

| Proposed split | Repo and implementation | Exact candidate Git revision | MIT copyright identified |
| --- | --- | --- | --- |
| TRAIN + DEV | `anantagarwal1307/Drag-Drop-Kanban-Board` — standalone HTML/CSS/vanilla native HTML5 Drag-and-Drop and localStorage | `47cfba359a71b70a6a23e511e554416aec006474` | Anant Kumar Agarwal |
| TRAIN + DEV | `dandemslay/React-KanbanBoard` — Vite/React drag-and-drop without a third-party DnD library; localStorage | `fa047f7a32afb57750ed5246456dc282ea7befcd` | Danny Rodriguez |
| TRAIN + DEV | `TomCani/kanban-dndkit-localstorage` — Vite/React/TypeScript, dnd-kit sorting and cross-column cards | `f3e129998e3140a386f1884faff314345e0eb10b` | Thomas Cani |
| Independent one-time BLIND only | `N3XT3R1337/kanban-board` — Vue 3 drag-and-drop board | `5d5a433d9860cec617ab523e91447fb6513ecee8` | panaceya (N3XT3R1337) |
| Independent one-time BLIND only | `PixelPenguin94/kanban-board-vanilla-js` — dependency-free modular vanilla HTML5 board with localStorage | `652d763c61f59a23fbd64179e82bd363cda337e0` | PixelPenguin94 |
| Additional independent BLIND candidate | `aashwinshukla/FaceTask` — React/dnd-kit full Kanban/dashboard/task persistence | `0698d3c0a2a4a63b636e40c62a21f645290a4705` | aashwinshukla |

## Security, data and compute gates before using any candidate

1. Clone/checkout only the **exact** indicated revision into private owner
   Windows and prove `git rev-parse HEAD` and clean tracked diff; preserve
   `LICENSE`, the original copyright and third-party attribution. Reject
   missing or incompatible license at the frozen revision.
2. Qualify genuine fully offline browser operation: bundled source builds,
   dependencies and Playwright Chromium already present, no external network
   requests, user accounts or paid services required.
3. Make the original application perform actual pointer drag and persisted
   destination-state checks. Only owner-side observer scripts may access DOM
   bbox source truth; no production app modification or model-facing bbox.
4. Collect only NEW unique 960×720 screenshots under *multiple actual*
   source-card/destination layouts. For each app, vary legitimate initial
   board/card state, columns, card order, scroll and display layout; maintain
   the H140 24-pixel START/END positional diversity gates.
5. Train/dev apps must remain codebase-disjoint from all blind apps. Freeze
   the blind split independently and never train on H136-H139 previously
   tested screenshot content, model predictions, screenshot hashes or labels.
6. Keep datasets/images, raw model responses, adapter weights and reference
   bboxes private. No public dataset, model publication, merge, deployment,
   paid cloud or Hala flagship 35B accuracy claim.

**Do not promote a candidate to `app_sources` just because it appears in
this file.** First complete and independently verify all gates above.
