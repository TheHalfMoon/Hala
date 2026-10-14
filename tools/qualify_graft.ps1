$ErrorActionPreference = 'Continue'
$graftEvidence = 'docs/reviews'
graft build --no-ignore --no-gitignore . *> "$graftEvidence/graft-build.log"
$graftBuildExit = $LASTEXITCODE
graft check --json . | Set-Content -LiteralPath "$graftEvidence/graft-check.json" -Encoding utf8
$graftCheckExit = $LASTEXITCODE
graft blast --format json . | Set-Content -LiteralPath "$graftEvidence/graft-impact.json" -Encoding utf8
$graftBlastExit = $LASTEXITCODE
$graftState = Get-Content -LiteralPath "$graftEvidence/graft-check.json" -Raw | ConvertFrom-Json
$graftRecord = [pscustomobject]@{date='2026-10-14';version='0.21.1';build_exit=$graftBuildExit;check_exit=$graftCheckExit;impact_exit=$graftBlastExit;graph_ok=$graftState.graph.ok;nodes=$graftState.graph.nodes;deep_context_built=(-not $graftState.context.missing);scope='Source graph freshness and change impact, not correctness or isolated execution';telemetry='disabled'}
$graftRecord | ConvertTo-Json | Set-Content -LiteralPath "$graftEvidence/graft-summary.json" -Encoding utf8
$graftRecord | ConvertTo-Json
if ($graftBuildExit -ne 0 -or $graftCheckExit -ne 0 -or $graftBlastExit -ne 0) {exit 1}
