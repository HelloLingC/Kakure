@echo off
chcp 936 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo   Kakure One-Click Installer (Windows)
echo   This script installs Python if needed; the rest of the
echo   installation is handled by install.py
echo ============================================================
echo.

REM -----------------------------------------------------------
REM Step 1: Find/install Python (>= 3.10)
REM -----------------------------------------------------------
set "PYTHON_EXE="

REM Prefer the py launcher (Python 3.11 required)
py -3 -c "import sys;raise SystemExit(0 if (3,11)<=sys.version_info<(3,12) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=py -3"

if not defined PYTHON_EXE (
    REM Fall back to the python command
    python -c "import sys;raise SystemExit(0 if (3,11)<=sys.version_info<(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if defined PYTHON_EXE (
    echo [1/6] Found a usable Python
) else (
    echo [1/6] Python not found or too old (3.11 required), installing Python 3.11 ...
    echo.

    REM Prefer winget
    where winget >nul 2>nul
    if !errorlevel!==0 (
        echo Installing Python 3.11 via winget, this may take a few minutes...
        winget install --id Python.Python.3.11 -e --silent --accept-source-agreements --accept-package-agreements >nul 2>nul
        if !errorlevel!==0 (
            echo winget installation succeeded.
        ) else (
            echo winget installation failed, falling back to the official installer...
        )
    )

    REM If winget failed or is unavailable, fall back to the official silent installer
    set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
    if not exist "!PYTHON_EXE!" (
        echo Downloading the official Python 3.11 installer...
        curl.exe -L -o "%TEMP%\kakure_python_setup.exe" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        if errorlevel 1 (
            echo [ERROR] Failed to download Python, please check your network.
            echo You can also download and install Python 3.11 manually from
            echo https://www.python.org/downloads/ and then re-run this script.
            pause
            exit /b 1
        )
        echo Running silent install (no admin rights needed)...
        "%TEMP%\kakure_python_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
        if errorlevel 1 (
            echo [ERROR] Silent Python install failed, please install Python 3.11
            echo manually and then re-run this script.
            pause
            exit /b 1
        )
    )
    if not exist "!PYTHON_EXE!" (
        echo [ERROR] Python installation failed, please install Python 3.11 manually
        echo and then re-run this script.
        pause
        exit /b 1
    )
    echo [1/6] Python 3.11 installed.
    echo.
)

REM -----------------------------------------------------------
REM Step 2: Run install.py to finish the installation
REM -----------------------------------------------------------
echo.
echo Running install.py to complete the installation...
"!PYTHON_EXE!" "%~dp0install.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed, see the messages above.
    pause
    exit /b 1
)
