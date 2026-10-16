# H23 two-group operator combined review

Scope: operational wrapper and two metadata templates only; no rerun of reviewed H23 scientific-module review. Operator SHA `cb48b0ee3824027b597c64c38b6b0dd389e50333e7436964c2279efb6f3ef034` and helper SHA `d8914643a539a059b6d7f5e4d4d940717ed2020b936948410fed2cd5ef58a42e` match inspected bytes. No operator/GPU/remote execution or source edits.

## Must-fix

- **P2 — `outputs/h23-operator.py:invoke` post-Popen exception path / `run` failure retention:** selector creation/registration/select, log open/write or stdout read exceptions bypass stdout close and process wait. `run` catches the exception and immediately creates/announces the evidence ZIP while the owned H23 producer may still be running and modifying scientific evidence. The independent GNU timeout eventually bounds that group, but export no longer represents a settled terminal attempt and may precede final task/fault/reap records. Put post-launch cleanup/settlement in a bounded finally path: close owned capture resources, let or cause only this owned invocation group to terminate within its fixed budget/grace, record explicit termination/reap outcome, and package after settlement (or explicitly record inability to settle). Add a CPU post-Popen selector/read/log failure regression proving the process is handled before retention. CPU mock reproduction: `selector.select` raises OSError; `invoke` exits with zero process.wait calls and zero stdout.close calls.

## Clean / intentional pending state

- Exact hash check precedes helper import. External immutable commit/SHA checks bind original H23 prospective bytes, unchanged H23 source/freeze, explicit hardware amendment and separate retention table. Null TEST manifest/GPU/table fields reject before source/cache/model work.
- Six fixed rows are ordered DEV0..2 then TEST0..2 with fp16/int8/nf4 mapping; original DEV rows are literal and verified publicly without requiring expired old local files. New TEST rows bind the new manifest/attempt/GPU and require both published raw receipt/ZIP validation and matching local terminal status/raw archive. All twelve public reads and all three new local checks precede clone/cache movement/ML work.
- Same expected physical/logical GPU UUID as the frozen TEST group is rechecked before H23 invocation. Cache movement is restricted to the fixed reviewed old cache into a nonexistent fixed H23 cache after a bounded cache scan.
- Normal invocation uses one fixed official GNU timeout command, 1200 seconds plus 5-second kill grace, in a newly owned session/group. Capture stops waiting after 1208 seconds; raw log prefix is bounded and overflow remains incomplete. The issue above is the exception branch, not the normal timer branch.
- Evidence preserves original prospective metadata plus the amendment, table and incremental scientific results; no cross-GPU paired cost or actual CUDA/native/production qualification is asserted. Pending immutable TEST/hardware metadata remains root-owned work before execution.
- No additional parsimony or product-scope issue found.

Validation: `rtk python -m unittest discover -s outputs -p test_h23_operator.py -v` — 10 CPU tests passed in 1.136 s. Exact operator/helper hashes checked. Additional CPU exception-path mock performed without subprocess launch. No actual H23 inference, clone/cache move, model download or publication.

panel: operator combined contribution — correctness ✓ fresh-context Codex · parsimony ✓ combined · product ✓ combined · security separate/root-owned
