# Local pre-deploy verification.
#
# The rule this enforces: never push to find out whether it works. Compile, run the
# test suite, and boot the real Streamlit entry point headlessly - all locally, in
# seconds. Deployment is then a check, not the first test.
#
#   .\scripts\verify.ps1                 # full check
#   .\scripts\verify.ps1 -SkipTests      # quick: compile + boot only
#   .\scripts\verify.ps1 -Port 8505      # if 8504 is taken
#
# Exit code 0 = safe to deploy. Anything else = do not deploy.

param(
    [switch]$SkipTests,
    [int]$Port = 8504,
    [int]$BootTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root ".cache"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$failures = [System.Collections.Generic.List[string]]::new()

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "FAIL: no interpreter at $python (create .venv first)" -ForegroundColor Red
    exit 1
}

# --- 1. compile ------------------------------------------------------------
Step "compile (src, tests, evals)"
& $python -m compileall -q src tests evals
if ($LASTEXITCODE -ne 0) { $failures.Add("compile failed") } else { Write-Host "ok" }

# --- 2. tests --------------------------------------------------------------
if (-not $SkipTests) {
    Step "tests"
    $testOut = Join-Path $logDir "verify-pytest.log"
    $testErr = Join-Path $logDir "verify-pytest.err"
    # Start-Process with file redirection: piped stdio can be blocked in some
    # sandbox modes, and a test run must never depend on that.
    $pytest = Start-Process -FilePath $python `
        -ArgumentList "-m", "pytest", "-q", "--tb=short", "--basetemp=.pytest-tmp-verify" `
        -WorkingDirectory $root -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $testOut -RedirectStandardError $testErr
    Get-Content $testOut -Tail 6 | Write-Host
    if ($pytest.ExitCode -ne 0) {
        $failures.Add("tests failed (exit $($pytest.ExitCode)); see $testOut")
    }
}

# --- 3. headless boot of the real entry point ------------------------------
# What production runs: streamlit_app.py at the repo root. Booting it here catches
# import errors, port/config mistakes and startup crashes before a deploy does.
Step "headless boot on port $Port"
$bootOut = Join-Path $logDir "verify-boot.log"
$bootErr = Join-Path $logDir "verify-boot.err"
$streamlit = Start-Process -FilePath $python `
    -ArgumentList "-m", "streamlit", "run", "streamlit_app.py",
        "--server.headless", "true", "--server.port", "$Port",
        "--browser.gatherUsageStats", "false" `
    -WorkingDirectory $root -PassThru -NoNewWindow `
    -RedirectStandardOutput $bootOut -RedirectStandardError $bootErr

$healthy = $false
$deadline = (Get-Date).AddSeconds($BootTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1500
    try {
        $health = Invoke-WebRequest "http://127.0.0.1:$Port/_stcore/health" `
            -UseBasicParsing -TimeoutSec 4
        if ($health.Content -match "ok") { $healthy = $true; break }
    } catch { }
    if ($streamlit.HasExited) { break }
}

if ($healthy) {
    try {
        $page = Invoke-WebRequest "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 10
        Write-Host ("ok: health=ok, page HTTP " + $page.StatusCode)
        if ($page.StatusCode -ne 200) { $failures.Add("page returned $($page.StatusCode)") }
    } catch {
        $failures.Add("health ok but page failed: $($_.Exception.Message)")
    }
} else {
    $failures.Add("app did not become healthy within ${BootTimeoutSeconds}s; see $bootErr")
    Get-Content $bootErr -Tail 10 -ErrorAction SilentlyContinue | Write-Host
}

# Always release the port, including a child process that outlived the wrapper.
Stop-Process -Id $streamlit.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 800
Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

# --- summary ---------------------------------------------------------------
Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "VERIFY PASSED - safe to deploy" -ForegroundColor Green
    exit 0
}
Write-Host "VERIFY FAILED - do not deploy:" -ForegroundColor Red
$failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
exit 1
