$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Environment belum tersedia. Jalankan .\setup_venv_windows.ps1 terlebih dahulu."
}

Set-Location $ProjectDir
Write-Host "MotionLens Streamlit berjalan di http://127.0.0.1:8501"
Write-Host "Tekan Ctrl+C untuk menghentikan aplikasi."
& $PythonExe -m streamlit run streamlit_app.py --server.port 8501
