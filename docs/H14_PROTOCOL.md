# H14 experimental normalized point protocol v1

This frozen experimental protocol is separate from production contracts. The only
accepted model output is a JSON object with exactly `point`, an array of exactly two
integers in 0..1000. Whitespace is allowed. Reuse the previously calibrated explicit
wrapper contract: either bare JSON or one exact `\`\`\`json` opening followed by
a newline and JSON, then a newline and `\`\`\`` closing. This compatibility
choice predates all H14 inference. Other fences, commentary, surrounding junk,
duplicate keys, floats, booleans, null, nonfinite constants and additional keys fail closed.
Maximum output text is 4096 characters. Invalid output causes no fixture event.

For an RGB image W by H (integer dimensions 1..4096), each axis maps with
`min(size - 1, floor(coordinate * size / 1000))`. Zero maps to the first pixel;
1000 maps to the last pixel. Rectangles are `[left, top, right, bottom)` in image
pixels. Screen origins, window borders, CSS coordinates and logical DPI units are
absent. DPI metadata does not rescale the frame; no screenshot resize/crop is
performed by the scorer. The processor may internally resize visual input, but
the model's requested output frame remains the original image. Never map these
points directly to a native display without a separately qualified adapter.

The prompt contains only the user's goal and the protocol instruction. The model
receives the rendered screenshot, not target IDs, hit boxes or expected state.
A valid point creates at most one selection event in an original one-click
synthetic fixture. The independent scorer checks the selection against private
expected state. A model completion claim is invalid schema and cannot satisfy it.
Geometric hits and fixture completion are reported separately, though identical
for these deliberately narrow selection environments. Neither is native task
completion or general full-task qualification.

P2 freezes source, task specifications, protocol and prompt before inference.
P3 renders all 64 prospective heldout screenshots with actual Pillow RAQM and a
declared font and persists PNG and RGB hashes before model loading. No evaluation
prompt tuning or training is permitted on these tasks. The manifest hash must be
supplied from a prior trusted freeze record. Rendering provenance includes font
bytes, Pillow version and shaping availability. This dataset contains original
CC0 authored UI specifications without imported screenshots or personal data.
The separately installed rendering font retains its own license; CC0 applies to
the authored task specifications and text, not the font software.

Eight families are authored separately from earlier workflows: calendar dates,
transit arrivals, warehouse stock, kanban priorities, seat locations, chart peaks,
palette colors and document versions. Four English and four Arabic variants per
family vary sizes, locations, values and distractors. Variants remain dependent;
Wilson 95% intervals are descriptive, not population confidence claims.
