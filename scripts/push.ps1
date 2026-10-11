# Send the code (and, with -Data, the assembled dataset) to the node. Run on the PC:
#     powershell -ExecutionPolicy Bypass -File "C:\Affective Computing\affect\scripts\push.ps1"
#     powershell -ExecutionPolicy Bypass -File "C:\Affective Computing\affect\scripts\push.ps1" -Data
# Packs src\, notebooks\, scripts\, tests\ and the top-level files into one archive with its
# SHA-256, and copies it with unpack.sh to ~/affect_incoming on the node in one scp (one
# password prompt). Then prints the command to run on the node. Nothing on the node changes
# until that command runs.
param([switch]$Data)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$NodeFile = Join-Path $Repo "node.local.txt"   # the node's login (user@host), one line; kept out of git and off the node
if (-not (Test-Path $NodeFile)) { throw "node.local.txt is missing beside README.md: write the node's login (user@host) in it, one line" }
$Node = (Get-Content $NodeFile -TotalCount 1).Trim()
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
# Windows' own tar (bsdtar). A GNU tar from Git for Windows, if first on the PATH, would read
# "C:\..." as a remote host and fail, so the full path is used.
$Tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $Tar)) { throw "Windows tar not found at $Tar" }
$Stage = Join-Path $env:TEMP "affect_push_$Stamp"
$Inc = Join-Path $Stage "affect_incoming"
New-Item -ItemType Directory -Force -Path $Inc | Out-Null

function Write-Hash([string]$Path) {
    $name = Split-Path -Leaf $Path
    $hash = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLower()
    [System.IO.File]::WriteAllText("$Path.sha256", "$hash  $name`n")
    return $hash
}

$Code = "affect_code_$Stamp.tar.gz"
$Items = @("src", "notebooks", "scripts", "sweeps", "tests", "README.md", "HANDOFF.md", "VERIFICATION_LOG.md",
           "findings.csv", "requirements.in", ".gitignore", ".gitattributes") |
         Where-Object { Test-Path (Join-Path $Repo $_) }
& $Tar -czf (Join-Path $Inc $Code) --exclude=__pycache__ --exclude=*.pyc -C $Repo @Items
if ($LASTEXITCODE -ne 0) { throw "tar failed for the code archive" }
$CodeHash = Write-Hash (Join-Path $Inc $Code)
Copy-Item (Join-Path $PSScriptRoot "unpack.sh") $Inc
Write-Host "code archive $Code  sha256 $CodeHash"

$DataName = $null
if ($Data) {
    $Assembled = Join-Path $Repo "data\iemocap\assembled"
    if (-not (Test-Path (Join-Path $Assembled "ASSEMBLY.json"))) { throw "no finished assembly at $Assembled" }
    $DataName = "affect_data_iemocap_$Stamp.tar"
    & $Tar -cf (Join-Path $Inc $DataName) -C (Join-Path $Repo "data\iemocap") assembled
    if ($LASTEXITCODE -ne 0) { throw "tar failed for the data archive" }
    $DataHash = Write-Hash (Join-Path $Inc $DataName)
    Write-Host "data archive $DataName  sha256 $DataHash"
}

scp -r $Inc "${Node}:"
if ($LASTEXITCODE -ne 0) { throw "scp failed; nothing was installed on the node" }
Write-Host ""
Write-Host "Copied to ~/affect_incoming on the node. There, run:"
Write-Host "    bash ~/affect_incoming/unpack.sh $Code"
if ($DataName) { Write-Host "    bash ~/affect_incoming/unpack.sh $DataName" }
Write-Host "Local copy of what was sent: $Stage"
