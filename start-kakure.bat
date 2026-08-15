@echo off
chcp 936 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM Add the bundled ffmpeg to PATH for this session
if exist "bin\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%~dp0bin\ffmpeg\bin;%PATH%"
)

REM Portable package (整合包) mode: embedded Python inside python\
if exist "python\python.exe" (
    echo Starting Kakure, your browser will open automatically...
    echo If it does not open, visit http://127.0.0.1:7530 manually.
    echo.
    echo Closing this window stops Kakure.
    echo.
    "python\python.exe" -m kakure.cli %*
    goto :end
)

REM Developer mode: virtual environment created by install.bat
if exist ".venv\Scripts\kakure.exe" (
    echo Starting Kakure, your browser will open automatically...
    echo If it does not open, visit http://127.0.0.1:7530 manually.
    echo.
    echo Closing this window stops Kakure.
    echo.
    .venv\Scripts\kakure.exe %*
    goto :end
)

echo [ERROR] Kakure not found.
echo If this is a fresh clone, run install.bat first.
echo If this is a portable package, re-download it (python\ is missing).
echo.
pause
exit /b 1

:end
if errorlevel 1 (
    echo.
    echo [ERROR] Kakure failed to start. See the messages above.
    pause
)
