$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt

Write-Host ""
Write-Host "VENV selesai dibuat dan package sudah di-install."
Write-Host "Jalankan aplikasi dengan:"
Write-Host ".\.venv\Scripts\python.exe app.py"
