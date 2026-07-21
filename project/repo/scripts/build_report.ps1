$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toolRoot = Join-Path $repoRoot "tmp\pdfs\tectonic"
$archivePath = Join-Path $repoRoot "tmp\pdfs\tectonic-0.16.9-x86_64-pc-windows-msvc.zip"
$tectonicPath = Join-Path $toolRoot "tectonic.exe"
$buildRoot = Join-Path $repoRoot "tmp\pdfs\build"
$reportSource = Join-Path $repoRoot "report\main.tex"
$finalPdf = Join-Path $repoRoot "report.pdf"

$tectonicUrl = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-pc-windows-msvc.zip"
$expectedSha256 = "131A24604785A9600989A3D91225F597DF52AC06F00AEFFE86FD529F99EE5CDD"

New-Item -ItemType Directory -Force -Path (Split-Path $archivePath), $toolRoot, $buildRoot | Out-Null

if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -Uri $tectonicUrl -OutFile $archivePath
}

$actualSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
if ($actualSha256 -ne $expectedSha256) {
    throw "Tectonic archive SHA-256 mismatch: expected $expectedSha256, got $actualSha256"
}

if (-not (Test-Path -LiteralPath $tectonicPath)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $toolRoot -Force
}

$version = & $tectonicPath --version
if ($LASTEXITCODE -ne 0 -or $version -ne "Tectonic 0.16.9") {
    throw "Unexpected Tectonic version: $version"
}

Push-Location (Join-Path $repoRoot "report")
try {
    & $tectonicPath --print --keep-logs --outdir $buildRoot $reportSource
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic compilation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$builtPdf = Join-Path $buildRoot "main.pdf"
if (-not (Test-Path -LiteralPath $builtPdf)) {
    throw "Tectonic completed without producing $builtPdf"
}

Copy-Item -LiteralPath $builtPdf -Destination $finalPdf -Force
Write-Output "Built $finalPdf with $version; compiler archive SHA-256 $actualSha256"
