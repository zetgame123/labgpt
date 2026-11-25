@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] Creating Python virtual environment...
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\pip.exe" (
    echo [ERROR] Failed to create .venv. Make sure Python 3.10-3.12 is installed.
    pause
    exit /b 1
)

echo [2/3] Installing dependencies...
".venv\Scripts\pip.exe" install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt

echo [3/3] Preparing .env...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example. Add your GROQ_API_KEY in Settings (gear icon) or in .env.
)

echo.
echo Setup complete! Run run.bat to start the overlay.
pause
