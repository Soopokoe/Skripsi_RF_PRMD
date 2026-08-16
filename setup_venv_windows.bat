@echo off
if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo VENV selesai dibuat dan package sudah di-install.
echo Aktifkan ulang dengan:
echo .venv\Scripts\activate.bat
pause
