# H174 interact.js original drag/drop candidate — source-native only

Immutable MIT original application: `taye/interact.js` Git revision
`a993eb8cd5cbc65f3a19e45ee24b7fedc026174f`; original tracked
`examples/dropzones/index.html` SHA-256
`9bc4dbe2a8f4e63b2f78a589b54adcd822ba552ecf481528a3b1d65f79430049`,
original LICENSE SHA-256
`e805e2367aed149858263a4ae0a62168146af837a9ba3095aa01d02c8559e7fe`.

The original Vite config aliases `@interactjs/` to original source workspace
`packages/@interactjs`, solving the direct-browser bare import blocker
without inventing a replacement application. No donor files are changed:
source Git objects are exported to a disposable scratch and only there is
original `yarn.lock` installed with disabled lifecycle scripts.

A free hosted Chromium native smoke navigates original
`/dropzones/index.html`, uses actual mouse movement/down/up to move original
`#drag3` to original `#drop1` (the original demo explicitly accepts drag3
in all dropzones), and requires the original drag event handler to update
text to `Dropped`. It then compares decoded screenshot RGBA bytes from two
independent original browser contexts, without saving frames. As before,
Service Workers/downloads/WebSockets and off-loopback HTTP are blocked at
browser-context level. This does not prove an OS egress firewall.

**Eligibility:** one successful native drag event and two-context replay
would establish a narrow original-app Linux smoke only. A failed workflow
is preserved as evidence until repaired on a newer exact SHA. Neither
a positive build nor a browser smoke is a TRAIN or DEV capture or proof of
actual user task coverage. The full owner historical source-family and
SHA-only registry still has not been proven complete. No OOD family may be
viewed or captured while training. Never reuse consumed H173 data, images,
gold, scores or predictions. No weights/GPU/optimizer, paid compute, main
merge, public release or deployment. Owner Windows Edge native
requalification is independent and pending owner DC reconnect.
