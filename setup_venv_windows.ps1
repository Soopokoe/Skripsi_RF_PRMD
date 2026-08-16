Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "VENV selesai dibuat dan package sudah di-install."
Write-Host "Aktifkan ulang dengan:"
Write-Host ".\.venv\Scripts\activate"
