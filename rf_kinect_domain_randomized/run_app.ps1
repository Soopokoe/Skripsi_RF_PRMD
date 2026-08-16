$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$CodexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonExe = $null

if (Test-Path $VenvPython) {
    try {
        & $VenvPython --version *> $null
        if ($LASTEXITCODE -eq 0) { $PythonExe = $VenvPython }
    } catch {}
}

if (-not $PythonExe -and (Test-Path $CodexPython)) {
    $PythonExe = $CodexPython
    $env:PYTHONPATH = (Join-Path $ProjectDir ".venv\Lib\site-packages")
}

if (-not $PythonExe) {
    throw "Python tidak ditemukan. Install Python 3.12, lalu jalankan setup_venv_windows.ps1 dari root project."
}

$env:RF_APP_PORT = "5001"
Set-Location $ProjectDir
Write-Host "MotionLens RF berjalan di http://127.0.0.1:5001"
Write-Host "Tekan Ctrl+C untuk menghentikan aplikasi."
& $PythonExe (Join-Path $PSScriptRoot "app.py")
