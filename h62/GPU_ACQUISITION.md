# H62 — Measured free-GPU acquisition evidence

**Date: 2026-10-22.** Every row below is an observation made on this host and
this account during this session. Nothing here is an advertised limit, an
inference from a different account, or a claim about a run that did not
happen.

## 1. The single most important correction to the inherited ledger

The H61 ledger and the H62 handoff both described a "36-hour allocation
mechanism" for a free Colab T4 and reported that it had been asked for many
times and answered with HTTP 503 every time.

**That mechanism was not running when this session started.**

| Check | Command | Observed |
|---|---|---|
| Any Colab / Kaggle / training watcher process | `Get-CimInstance Win32_Process` filtered on `python|kaggle|colab` | **Only** MCP helper processes (`d1_mcp.py`) and two interactive Python shells. No watcher, no trainer, no queued job. |
| Provider-side live sessions | `colab sessions` | `No active sessions found on server.` |
| Provider-side assignments | `colab usage` | `Active assignments: 0` |
| Provider-side balance | `colab usage` | `Current balance: 0.00 compute units` |

So the frontier was a **description of a mechanism**, not a running
mechanism. A configured watcher that is not running produces no evidence and
no progress, and a 36-hour promise that is not backed by a live process is
not a plan, it is a hope. This is now fixed: H62 ships a watcher that is
actually running, over a state directory that survives the session, with a
request budget that is auditable.

## 2. What the provider actually answers for this account

| Probe | Command | Result | What it proves |
|---|---|---|---|
| Free **T4** assignment | `colab new -s hala-h62 --gpu T4` | **`Service Unavailable` (HTTP 503)** on `POST /tun/m/assign?...&variant=GPU&accelerator=T4` | The T4 accelerator variant is not being granted. |
| **CPU** assignment, same account, same CLI, same minute | `colab new -s hala-h62-cpu` | **`Session READY`** | **The assignment path works.** The 503 is specific to the T4 variant, not to the account, the OAuth token, the CLI, or the session API. |
| Remote execution in that session | `colab exec -s hala-h62-cpu -f probe.py` | `HALA_PROBE {"python": "3.13.16", "platform": "Linux-6.6.122+-x86_64-with-glibc2.39", "cwd": "/content", "exec_ok": true}` | The CLI can deliver a file to a remote Linux kernel and read real output back. The whole H60/H61 remote path is available. |
| Hardware of the live session | `colab status -s hala-h62-cpu` | `Hardware: CPU \| Shape: Standard \| Variant: DEFAULT \| Status: IDLE` | The allocation was a CPU machine, correctly reported as CPU and not as GPU. |
| Release of the session | `colab stop -s hala-h62-cpu` | `Session terminated` | Resources are released cleanly. |
| Billing after the whole probe | `colab usage` | `Current balance: 0.00 compute units`, `Usage rate: 0.00/hr`, `Active assignments: 0` | **No paid compute was consumed.** The account remained at exactly zero. |

### Why this matters, and what it does not prove

The H61 report said the 503 was "a provider capacity condition, not an
account or quota condition". The CPU probe now supports that reading with
direct evidence rather than inference: the identical assignment API, with
the identical credentials and the identical CLI, succeeds for the CPU
variant while it refuses the T4 variant.

**It still does not prove that the free T4 tier has capacity.** The provider
returns the same opaque 503 for "no T4 free right now" and for "your
account cannot have a T4", and it does not say which. H62 therefore records
the outcome as `CAPACITY_OR_BACKEND_UNAVAILABLE` and never as
`ENTITLEMENT_BLOCKED`, and it never claims that the free tier is exhausted.
The honest statement is: **the T4 variant is not available to this account
right now, and the assignment mechanism itself is healthy.**

## 3. Kaggle — real, free, and still blocked for this project

| Measurement | Command | Result |
|---|---|---|
| Weekly GPU quota | `kaggle quota` | `GPU  used 1.09h  remaining 28.91h  total 30.00h  refreshAt 2026-10-24T00:00:00` |
| Weekly TPU quota | `kaggle quota` | `TPU  used 0.00h  remaining 20.00h  total 20.00h  refreshAt 2026-10-24T00:00:00` |
| Concurrent jobs | `kaggle kernels list --mine` + `kaggle kernels status` on each Hala kernel | Every Hala kernel reports `KernelWorkerStatus.COMPLETE`. **Nothing is running.** The 28.91 h is unclaimed. |
| Authentication | `kaggle datasets list --mine` | Succeeds. 12 private Hala datasets exist, including the H60 `slider` shards. |

**The quota is real and the GPU is free. H62 does not use it, and the
reason is unchanged and already on the record.**

Kaggle's Terms of Use (effective 22 June 2025) say: *"You will only use the
Services for your own internal, personal, non-commercial use, and not on
behalf of or for the benefit of any third party."* Hala exists to power
Aladdin Pro, so a Hala training run is on behalf of and for the benefit of
a third party. `docs/FREE_GPU_RESOURCES.md` already records this as
**"Blocked for this commercial project pending provider permission"**, and
H60 and H61 both wrote in their ledgers that **"the guards stay, not
weakened"**.

`training/scripts/train_adapter.py` and `training/scripts/baseline.py` both
refuse a Kaggle runtime, and those refusals are untouched by this work.
Turning a measured 28.91 h of free GPU into a training run by moving that
guard would be a terms violation dressed as progress.

**What the founder can do, and what it buys:**

1. Obtain explicit Kaggle permission for commercial research use. That
   converts 28.91 h of guaranteed, queue-free GPU into a real option, and
   it is the single highest-leverage unblock in this project.
2. Leave it blocked. Then the free Colab T4 remains the only compliant
   path, and the watcher in `h62/colab_t4_watcher.py` is the mechanism.

H62 does not assume either answer. It records the quota as a measured fact
and keeps the guard.

**Kaggle private storage is unaffected.** The H60 shards stay published and
readable; H62 does not republish, delete or modify any of them, and the
concurrent H60 corpus work is untouched.

## 4. The watcher that is now actually running

`h62/colab_t4_watcher.py` is the replacement for the configured-but-dead
36-hour mechanism. Its properties are enforced by
`tests/test_h62_colab_watcher.py` (19 tests) and are not aspirations:

| Property | Enforcement |
|---|---|
| One request at a time, never a second session while one exists | `run_watcher` checks `colab sessions` first and **adopts** a live GPU session |
| Never spends paid compute | `ProviderClient.balance_is_zero()` reads the provider's own balance before every allocation and refuses unless it is exactly `0.00` |
| Bounded request rate | Exponential backoff from 120 s to a 1800 s cap with ±25 % jitter; a hard request budget of 80 and a hard 36 h deadline |
| Honest failure classification | `classify_provider_error` names `CAPACITY_OR_BACKEND_UNAVAILABLE`, `ENTITLEMENT_BLOCKED`, `QUOTA_BLOCKED`, `AUTHENTICATION_FAILED`, `PROVIDER_TIMEOUT`, `UNKNOWN_FAILURE` — and a 503 is never recorded as an entitlement block |
| Durable across a restart | An append-only, `fsync`-ed JSONL receipt stream; `ReceiptLog.restore()` rebuilds the attempt and request counters so a restart continues instead of restarting the clock |
| Terminal conditions stop it | An entitlement, quota or authentication failure ends the loop instead of hammering the provider |
| Secrets do not leak into the trail | `redact()` strips token, secret, password, api-key and authorization values, including a `Bearer` scheme |

The watcher writes to `h62/STATE.md` and its receipt stream, and it is
launched detached so that it survives the end of an agent session.

## 5. What is NOT true and is not claimed

* **No T4 has been obtained.** No GPU training has run in this session.
* **No H61 checkpoint exists.** The H61 pipeline is verified and ready; it
  has never executed on a GPU.
* **No H61 performance number exists.** The only measured GUI result in the
  project is H60's negative one: strict bare JSON 0 → 17 of 48, geometric
  grounding **0 of 24** on both public DEV sets, real GUI drag task success
  **0 of 48**.
* **A 503 is not proof of capacity**, and a zero balance is not proof that
  the session that materialises will be free forever. Both are checked at
  the moment they matter, by the code, not by this document.
* **The Colab free tier can disappear mid-run.** The remote entry point
  therefore publishes an immutable checkpoint bundle at every optimizer
  step and resumes from a SHA-verified private checkpoint, so a lost session
  costs time and not the run.
