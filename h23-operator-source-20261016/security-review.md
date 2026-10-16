# H23 two-group operator independent security review

Scope: operator preparation grain only: `outputs/h23-operator.py`, its frozen TEST retention helper, two pending hardware/retention templates and operator tests. No H23 scientific source re-audit or operational run.

Reviewed bytes: operator SHA256 `cb48b0ee3824027b597c64c38b6b0dd389e50333e7436964c2279efb6f3ef034`; helper SHA256 `d8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e` independently match.

Verdict: one must-fix before operator execution.

| Class | Location | Finding | Concrete fix |
|---|---|---|---|
| must-fix (P2) | `outputs/h23-operator.py:242` | Exceptions after `Popen` from selector setup/read or evidence-log writes skip both stdout closure and owned-process wait/cleanup. `run()` catches the exception and immediately archives while the model invocation may remain active under its separate 1200s GNU timeout. Its incremental result can continue changing after export, losing final failure/reap evidence and leaving model work running after the wrapper reports failure. CPU-only injection of `os.read` raising `OSError` confirmed zero `wait`, `terminate`, `kill` or stdout-close calls. | Guard the entire post-spawn monitor with `try/finally`; on monitor failure stop only the new owned invocation group, apply bounded grace/kill/reap, record confirmed or unconfirmed cleanup explicitly, close the pipe, then retain evidence. Add a monitor-exception regression demonstrating bounded owned cleanup before export; no external PID/name kill or automatic rerun. |

Clean checks:

- Original DEV rows are literal immutable rows, bound to experiment `60a098...`; reviewed helper binds source `07591...`, original experiment/GPU identity and exact DEV mode/index/path. Old DEV local paths are unnecessary.
- New TEST has a separate immutable manifest and experiment SHA; its local terminal status and raw archive must match the public raw archive SHA, archived status/manifest and frozen physical GPU. Exactly six ordered DEV0..2/TEST0..2 rows are required. All six public receipts and raw ZIPs are checked before clone/cache/model preparation.
- Helper bytes are SHA-checked before import. Metadata and evidence URLs have fixed public host/repository, immutable commit paths, external digests and read bounds; redirects reject. ZIP validation covers unique/bounded inventory, per-member/total size, required raw evidence, hashes/CRC on consumed members and source/mode/result identity; no extraction or arbitrary execution occurs.
- Hardware amendment and retention table are separate external commit/SHA inputs. Their present null TEST/hardware fields fail closed. Original H23 source `18ad45...`, freeze `4c522...`, original prospective bytes, pins and scientific scope remain fixed. H23 expected GPU equals the frozen TEST GPU; no cross-GPU paired-cost claim.
- Linux Kaggle identity, exact clean source, reviewed cache traversal, exact non-symlink old-cache rename into a nonexistent new cache, package/font/RAQM/logical+physical GPU checks and fixed official GNU timeout invocation are present. The normal launch uses one new owned session/group, no shell, caller program, external PID or native-action scope.
- Output log prefix, member/total/ZIP bounds and failure status remain unqualified. No credentials, telemetry, billing or external service execution were introduced. Templates and source preparation do not establish remote execution or GPU qualification.

Validation: `rtk proxy python outputs/test_h23_operator.py -v` passed all 10 tests in 1.411s. Additional stdlib mock reproduced the monitoring-exception cleanup gap; no actual subprocess, network fetch, clone, cache move, model/GPU operation or publication occurred. Actual TEST/hardware fields, public artifacts and H23 evidence remain root-owned pending gates.

panel: operator security ✓ independent session-model judge; correctness/Jev/Graft/publication and actual hardware qualification are separate gates.
