# H22 CI history repair — independent combined review

Scope: four inserted lines in `.github/workflows/ci.yml` and `docs/H22_PRECISION.md`, against worktree HEAD `21128cd1302d018128650afa75aae14febf82b14`. No runtime/test source change, broad test run, GPU execution or source edit by this reviewer.

- Correctness: clear. `fetch-depth: 0` is correctly nested under the existing immutable checkout action and provides history for the unchanged `git show 11a3a5359afa896787198e0492ccd036d0b6f4f8:...` assertions. Verified that reference is a commit and an ancestor of HEAD, so fetching full history addresses its omission from the default shallow checkout. Existing matrix, pins, permissions, checks and timeout are unchanged.
- Parsimony: clear. The checkout option directly repairs the repository prerequisite; no test-time network operation, skip or weakened assertion was added.
- Product/evidence: clear. The documentation identifies the historical-reference checkout failure and explicitly leaves actual CI rerun qualification pending. This review assesses the narrow repair; historical run counts/log details supplied in the handoff were not independently retrieved here.

Validation: `git diff --check` passed; only the two reported files are modified. Independently recomputed and verified the unchanged prospective freeze `98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b` and its listed source/config entries. No YAML-mirror test or broad rerun was performed.

Must fix: none. Worth considering: none. Actual published GitHub Actions green remains the execution gate.

Panel contribution: independent combined correctness/parsimony/product, session model; security not triggered by this delta.
