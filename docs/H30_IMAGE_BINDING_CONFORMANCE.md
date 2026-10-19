# H30 image binding conformance

`tests/test_visual_binding_integration.py` exercises the authenticated v2
candidate path over the repository's loopback HTTP server. It builds nonblank
PNG fixtures in memory and uses a deterministic pixel test double that finds
the target-colored region, hashes all decoded RGB pixels, and returns that
region's center. The test checks the result against the image digest, frame and
request identities, and observed element bounds.

The English left-to-right and Arabic-locale right-to-left cases use synthetic
rectangles. Arabic is locale and accessible-name evaluation data only; these
fixtures do not render or test Arabic shaping. The engine is a deterministic
test double, not a neural model. Passing tests establish parser, binding,
redaction, and geometry behavior for these fixtures only; they do not measure
grounding quality, native UI behavior, or deployment readiness. Successful
advice remains unauthorized for execution, has no confidence value, and
requires independent verification.
