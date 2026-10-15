# Aladdin compatibility register

| Adapter | Planning/source revision | State |
|---|---|---|
| `hala.request.v1` / proposal/plan/decision | Aladdin `0c3c86c8acfb57e3dd3ce83c83ee8cfd3b5035c1`, MODEL_INTEGRATION_CONTRACTS.md | Implemented strict Hala-side mock. Adds required task/session bindings; not a literal wire drop-in or kernel capability qualification |
| Live reference recheck | Aladdin `00a6dba9dab8759b175dd410cc036cb0ba8d2b70` | Five commits ahead; browser/MCP/kernel source changed. Observed diff did not change AI planning schema documents. No Aladdin checkout, branch, worktree, PR or source was modified |
| Native kernel/relay adapter | Current qualified Aladdin vocabulary | NOT IMPLEMENTED/UNQUALIFIED. Requires exact current capability mapping and independently approved local policy/execution tests |

Aladdin's architecture documents are adopted planning direction with proposed model wire shapes. PR #328 was open at inspected head `d796f2f74e5390076a2d2828bea5dc3cc06b990`; it is research, not deployed inference. Native computer tools remain governed by Aladdin. Hala's logical labels cannot invent native operations, lower ceilings, mint targets/leases/approvals/receipts, or convert `done_proposed` into success.

Mock outputs are revalidated against request/task/tenant/device/session/observation bindings. Compatibility changes should add a reviewed versioned adapter rather than silently changing Hala v1 or relying on unfinished Aladdin features.
