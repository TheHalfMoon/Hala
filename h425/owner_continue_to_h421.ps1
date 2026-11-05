# Owner only: H425 genuinely qualified source model -> PRIVATE Kaggle H421 paired eval.
[CmdletBinding()]
param(
  [string]$Results = "D:\Hala-H425-Completed-Private",
  [string]$Stage = "D:\Hala-H425-H421-Qualified-Dataset-Private",
  [string]$EvalFolder = "D:\Hala-H425-H421-Eval-Private",
  [string]$Sources = "D:\Hala-H419-Source-Preflight",
  [string]$Paired = "D:\Hala-H422-Paired-Private",
  [string]$H421 = "D:\Hala-H421-Counterfactual-Private"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VerifyScript = Join-Path $Root "h425\owner_verify_private_gpu.ps1"
$StageScript = Join-Path $Root "h425\stage_qualified_h421.py"
$ExpectedId = "abdulazizshehri/hala-h425-genuine-trained-parent-private"
$EvalMetadata = Join-Path $Root "h425\kaggle\h421-eval-kernel-metadata.json"
$EvalCode = Join-Path $Root "h425\kaggle\h425_h421_paired_eval.py"
# Verify first, BEFORE any model staging or upload. Provider RUNNING/ERROR => exit1.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $VerifyScript -OutputRoot $Results -SourceRoot $Sources -PairedRoot $Paired -H421Root $H421
if ($LASTEXITCODE -ne 0) { throw "H425 actual source/LoRA/checkpoints not approved; stop" }
if (Test-Path -LiteralPath $Stage) { throw "H425 owner model stage collision" }
if (Test-Path -LiteralPath $EvalFolder) { throw "H421 evaluation folder collision" }
$ReceiptRoot = Join-Path $Results "h425-owner-private-h417-replay-paired-native"
& python $StageScript --receipt-root $ReceiptRoot --sources $Sources --paired $Paired --h421 $H421 --output-root $Stage
if ($LASTEXITCODE -ne 0) { throw "H425 source-only strong gate failed; stop without upload" }
$meta = Get-Content (Join-Path $Stage "dataset-metadata.json") -Raw | ConvertFrom-Json
if ($meta.id -cne $ExpectedId -or $meta.isPrivate -ne $true) { throw "Private exact Kaggle dataset metadata invalid" }
# Kaggle 'datasets create' is PRIVATE by default; NEVER use -u.
& kaggle datasets create -p $Stage -r skip
if ($LASTEXITCODE -ne 0) { throw "Private H425 model dataset creation failed" }
$MetaProofDir = Join-Path (Split-Path -Parent $Stage) "Hala-H425-H421-Kaggle-Metadata-Private"
if (Test-Path -LiteralPath $MetaProofDir) { throw "Kaggle metadata proof path collision" }
New-Item -ItemType Directory -Path $MetaProofDir | Out-Null
& kaggle datasets metadata $ExpectedId -p $MetaProofDir
if ($LASTEXITCODE -ne 0) { throw "Kaggle PRIVATE metadata readback failed" }
$provider = Get-Content (Join-Path $MetaProofDir "dataset-metadata.json") -Raw | ConvertFrom-Json
if ($provider.info.isPrivate -ne $true -or $provider.info.datasetSlug -cne "hala-h425-genuine-trained-parent-private") { throw "Refuse inference: Kaggle dataset not private" }
$eval = Get-Content -LiteralPath $EvalMetadata -Raw | ConvertFrom-Json
if ($eval.is_private -ne $true -or $eval.enable_gpu -ne $true -or $eval.machine_shape -cne "NvidiaTeslaT4" -or $eval.dataset_sources.Count -ne 8 -or $ExpectedId -notin $eval.dataset_sources) { throw "H421 FREE PRIVATE T4 evaluation manifest invalid" }
New-Item -ItemType Directory -Path $EvalFolder | Out-Null
Copy-Item -LiteralPath $EvalMetadata -Destination (Join-Path $EvalFolder "kernel-metadata.json")
Copy-Item -LiteralPath $EvalCode -Destination (Join-Path $EvalFolder "h425_h421_paired_eval.py")
if ((Get-FileHash $EvalCode -Algorithm SHA256).Hash -cne (Get-FileHash (Join-Path $EvalFolder "h425_h421_paired_eval.py") -Algorithm SHA256).Hash) { throw "Frozen H421 evaluation code SHA mismatch" }
& kaggle kernels push -p $EvalFolder
if ($LASTEXITCODE -ne 0) { throw "Private H421 384 genuine-case GPU model evaluation push failed" }
Write-Host "H425 genuinely source-qualified, PRIVATE H421 original 384-case model evaluation submitted. NOT external OOD/release."
