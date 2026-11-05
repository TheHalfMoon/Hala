# Owner-only H425 authentic free Kaggle model receipt verification.
# Never publishes weights or images; no push, merge, or paid GPU.
[CmdletBinding()]
param(
    [string]$OutputRoot = "D:\Hala-H425-Completed-Private",
    [string]$SourceRoot = "D:\Hala-H419-Source-Preflight",
    [string]$PairedRoot = "D:\Hala-H422-Paired-Private",
    [string]$H421Root = "D:\Hala-H421-Counterfactual-Private",
    [switch]$StatusOnly
)
$ErrorActionPreference = "Stop"
$Slug = "abdulazizshehri/hala-h425-private-h417-replay-paired-training"
$CodeRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$StatusText = & kaggle kernels status $Slug 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "H425 authenticated Kaggle status retrieval failed" }
Write-Host $StatusText.Trim()
if ($StatusOnly) { return }
if ($StatusText -notmatch 'KernelWorkerStatus\.COMPLETE') {
    throw "H425 Kaggle provider is not COMPLETE; refusing download/qualification"
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
& kaggle kernels output $Slug --path $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "H425 Kaggle provider output download failed" }
$Output = Join-Path $OutputRoot "h425-owner-private-h417-replay-paired-native"
$ReceiptPath = Join-Path $Output "receipt.json"
$SidecarPath = Join-Path $Output "receipt.sha256"
if (!(Test-Path -LiteralPath $ReceiptPath -PathType Leaf) -or !(Test-Path -LiteralPath $SidecarPath -PathType Leaf)) {
    throw "H425 authentic private GPU receipt or SHA sidecar absent"
}
$Digest = (Get-FileHash -LiteralPath $ReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$Sidecar = (Get-Content -LiteralPath $SidecarPath -Raw).Trim()
if ($Digest -cne $Sidecar) { throw "H425 Kaggle receipt SHA sidecar does not match provider bytes" }
$Data = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
if ($Data.status -ne "COMPLETE_PRIVATE_H425_H417_REPLAY_TRUE_NATIVE_PAIRED") {
    throw "H425 private GPU did not complete true training; status: $($Data.status)"
}
$Adapter = Join-Path $Output "private-h417-replay-paired-native-adapter\adapter_model.safetensors"
$Checkpoints = Join-Path $Output "private-checkpoints"
$Validator = Join-Path $CodeRoot "h425\validate_receipt.py"
foreach ($File in @($Adapter, $Validator)) {
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) {
        throw "H425 trained LoRA or verifier missing: $File"
    }
}
if (!(Test-Path -LiteralPath $Checkpoints -PathType Container)) {
    throw "H425 eight intermediate actual LoRA checkpoints missing"
}
foreach ($Root in @($SourceRoot, $PairedRoot, $H421Root)) {
    if (!(Test-Path -LiteralPath $Root -PathType Container)) {
        throw "H425 owner original source-root missing: $Root"
    }
}
# Normal owner Python sees user-site Pillow; -I disables it unless venv owns Pillow.
& python $Validator --receipt-root $Output --adapter $Adapter --checkpoints-root $Checkpoints --sources $SourceRoot --paired $PairedRoot --h421 $H421Root
if ($LASTEXITCODE -ne 0) {
    throw "H425 trained LoRA, 240 source JPEGs, or checkpoints failed authentic verification"
}
Write-Host "H425 genuine private training evidence replayed; NOT external OOD or release approval."
