@echo off
chcp 936 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM Add the bundled ffmpeg to PATH for this session
if exist "bin\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%~dp0bin\ffmpeg\bin;%PATH%"
)

if not exist ".venv\Scripts\kakure.exe" (
    echo [ERROR] Kakure not found. Please run install.bat first.
    echo.
    pause
    exit /b 1
)

echo Starting Kakure, your browser will open automatically...
echo If it does not open, visit http://127.0.0.1:7860 manually.
echo.
echo Closing this window stops Kakure.
echo.

.venv\Scripts\kakure.exe

if errorlevel 1 (
    echo.
    echo [ERROR] Kakure failed to start. See the messages above.
    echo You can re-run install.bat to fix the installation.
    pause
)
