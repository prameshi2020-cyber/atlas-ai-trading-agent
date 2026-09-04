<#
.SYNOPSIS
  ATLAS pre-commit verification. Stops on the first failure. Never commits.
.EXAMPLE
  .\scripts\verify.ps1                 # full gate
  .\scripts\verify.ps1 -Quick          # compile + diff check only
  .\scripts\verify.ps1 -Focus tests/test_decision_engine.py, tests/test_walk_forward.py
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [string[]]$Focus = @()
)

$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { Join-Path $PSScriptRoot ".." } else { Get-Location }
Set-Location $root
$py = ".\.venv\Scripts\python.exe"

function Step($name, [scriptblock]$body) {
    Write-Host "`n== $name ==" -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "ok: $name" -ForegroundColor Green
}

Step "python present" { & $py --version }

Step "compile changed python files" {
    $files = @(git diff --name-only HEAD -- '*.py') + @(git ls-files --others --exclude-standard -- '*.py') |
        Where-Object { $_ -and (Test-Path $_) } | Sort-Object -Unique
    if ($files) { & $py -m py_compile @files } else { Write-Host "(no changed .py files)"; $global:LASTEXITCODE = 0 }
}

Step "whitespace / conflict markers (git diff --check)" { git --no-pager diff --check HEAD }

Step "no tabs in app/ or tests/" {
    $hits = git --no-pager grep -n -P "\t" -- 'app/*.py' 'tests/*.py'
    if ($hits) { $hits; $global:LASTEXITCODE = 1 } else { $global:LASTEXITCODE = 0 }
}

Step "no stray artefacts in repo root" {
    $junk = git ls-files --others --exclude-standard | Where-Object { $_ -notmatch '/' -and $_ -notmatch '\.(md|txt|toml|yml|yaml|ps1|py)$' }
    if ($junk) { Write-Host "untracked root files: $junk"; $global:LASTEXITCODE = 1 } else { $global:LASTEXITCODE = 0 }
}

if ($Focus.Count -gt 0) {
    Step "focused tests" { & $py -m pytest -q --tb=short @Focus }
}

if (-not $Quick) {
    Step "architecture boundary" { & $py -m pytest -q --tb=short tests/test_ai_boundary_imports.py }
    Step "full suite" { & $py -m pytest -q --tb=short }
}

Write-Host "`nALL CHECKS PASSED - safe to stage and commit." -ForegroundColor Green
Write-Host "Reminder: stage only the files of this task; leave other agents' work unstaged." -ForegroundColor Yellow
