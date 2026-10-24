# H91 — Read-only Kaggle provider status without the Windows device

This phase addresses the **real evidence-access blocker**, not additional training scaffolding. The Windows Desktop Commander connection reached its monthly usage limit after it launched private H87 Kaggle v4. The H87 end state and checkpoint are therefore **unverified**, regardless of earlier synthetic CUDA and GitHub CI successes.

## Exact live GitHub Actions result

On October 24, 2026, the new [H91 read-only workflow run #40071237277](https://github.com/TheHalfMoon/Hala/actions/runs/40071237277) **executed** on GitHub Actions. Its first step found none of the accepted Kaggle credentials in the repository's GitHub Actions Secrets:

- `KAGGLE_API_TOKEN` — not configured
- `KAGGLE_USERNAME` plus `KAGGLE_KEY` — not configured

It emitted `BLOCKED_AUTH_NOT_CONFIGURED` and correctly did **not** query either private Kaggle notebook. The GitHub job itself concluded `success` because its configuration check completed; **that does not mean Kaggle access or training was successful**. Do not interpret it otherwise.

## One-time private configuration (repository owner action)

1. Open [Kaggle API token settings](https://www.kaggle.com/settings/api) and generate an API access token, ideally short-lived/revocable.
2. In the Hala repository, go to **Settings → Secrets and variables → Actions → New repository secret**. Set the name to exactly `KAGGLE_API_TOKEN` and store the Kaggle token there. Keep it OUT of source files, pull requests, public notebook logs, chat messages, and artifacts. Alternately, configure both legacy `KAGGLE_USERNAME` and `KAGGLE_KEY` as separate encrypted Actions secrets.
3. After the secret is saved, **update only `h91/recheck-trigger.txt` on branch `ps/h91-readonly-private-kaggle-status-bridge`** (for example, change its `attempt=1` number to `attempt=2`), and push/commit that change. This intentionally inert file is in the H91 GitHub Actions `push.paths` allowlist, so updating it runs the existing read-only provider check on the **Draft branch without merging**. The H91 code never checks out or executes the trigger file. Read the resulting GitHub Actions job summary. The actual workflow is `.github/workflows/kaggle-research-status.yml`. **Do not rely on the `Run workflow` button before the workflow file exists on the default branch; GitHub requires the default branch for manual `workflow_dispatch`.**

**Security:** TheHalfMoon/Hala is a public GitHub repository and its Actions logs/summaries may be public. The workflow intentionally uses a trusted-branch **push** trigger and optional manual dispatch, `permissions: contents: read`, no repository checkout or execution of PR code with credentials, no `pull_request_target`, and no model, checkpoint, private training input or raw notebook log downloads. It queries only the fixed H87/H88 kernel names with `kaggle kernels status`; for completed/failed jobs it may select a single bounded, metadata-only JSON receipt via `--file-pattern`. It emits only constrained status/diagnostic fields. **The token itself may grant broad Kaggle account permissions**, even though this workflow makes only read requests: rotate/revoke it after use and do not give it to untrusted workflows or users.

If the job produces `COMPLETE`, that means only the provider reports the notebook finished. Use H89 on real downloaded complete bytes, or H90 if an optimizer checkpoint exists but final receipts are missing. Never mark Hala trained based on provider status, synthetic tests, or unverified success JSON alone.

## Scope

- Known private research notebooks: `abdulazizshehri/hala-h87-private-original-h71-first-qwen-step` and `abdulazizshehri/hala-h88-private-original-h71-qwen-step`.
- No automated GPU allocation or Kaggle submission.
- No paid compute, merge, publish, release, or direct access to owner credentials.
- Keep Draft PR, personal noncommercial research, and model release **NO-GO** until actual optimizer checkpoint bytes and independent bilingual held-out evaluation are verified.
