@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM OriginMCP Server v10.2 - Portable Launcher
REM ============================================================================
REM 
REM Configuration is loaded from (in order of priority):
REM 1. Environment variables
REM 2. origin_mcp.ini in current directory
REM 3. origin_mcp.ini in script directory
REM 4. Default values in server.py
REM
REM To transfer to another server:
REM 1. Copy this BAT, server.py, and origin_mcp.ini
REM 2. Edit origin_mcp.ini with new paths
REM 3. Run setup-origin-mcp.ps1 or create venv manually
REM 4. Double-click this BAT to start
REM ============================================================================

REM If double-clicked, re-launch in a persistent cmd window and exit this stub
if /i "%~1" NEQ "KEEP" (
    start "OriginMCP Server" cmd.exe /k ""%~f0" KEEP"
    exit /b
)

REM Get script directory (where the BAT file is located)
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to change to %SCRIPT_DIR%
    pause
    exit /b 1
)

echo ================================================
echo       OriginMCP Server v10.2
echo ================================================
echo.
echo Script Directory: %SCRIPT_DIR%
echo.

REM ==========================================================================
REM Configuration - Edit these or use origin_mcp.ini
REM ==========================================================================
set "OWUI_URL=https://webui.forthought.cc"
set "OWUI_PUBLIC_URL=https://webui.forthought.cc"

REM Load JWT from secrets file if not set
if "%JWT_TOKEN%"=="" (
    if exist "%SCRIPT_DIR%\secrets\owui_jwt.txt" (
        set /p JWT_TOKEN=<"%SCRIPT_DIR%\secrets\owui_jwt.txt"
        echo [config] JWT loaded from secrets\owui_jwt.txt
    )
)

if "%JWT_TOKEN%"=="" (
    echo WARNING: JWT_TOKEN not set - file operations will fail
    echo.
    echo To fix: Create secrets\owui_jwt.txt with your Open WebUI JWT token
    echo.
)

REM ==========================================================================
REM Python Environment Setup
REM ==========================================================================
set "PYTHONUNBUFFERED=1"
set "EXPORT_MODE=owui"
set "EXPORT_DIR=%SCRIPT_DIR%\exports"

if not exist "%EXPORT_DIR%" mkdir "%EXPORT_DIR%"


REM Find Python - check venv first, then system Python
set "VENV_PY=%SCRIPT_DIR%\venv\Scripts\python.exe"
set "SYSTEM_PY=python"

if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    echo [python] Using venv: %VENV_PY%
) else (
    set "PY=%SYSTEM_PY%"
    echo [python] Using system Python
    echo WARNING: Virtual environment not found at %SCRIPT_DIR%\venv
    echo Run setup-origin-mcp.ps1 to create it
)

REM Server script
set "SV=%SCRIPT_DIR%\server.py"

if not exist "%SV%" (
    echo ERROR: server.py not found at %SV%
    pause
    exit /b 1
)

REM ==========================================================================
REM Pre-flight Checks
REM ==========================================================================
echo.
echo [checks] Running pre-flight checks...

REM Check Python
"%PY%" --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found or not working
    pause
    exit /b 1
)

REM Check required packages
"%PY%" -c "import fastapi, uvicorn, numpy, pandas, scipy, matplotlib, requests" 2>nul
if errorlevel 1 (
    echo WARNING: Some Python packages may be missing
    echo Run: pip install fastapi uvicorn numpy pandas scipy matplotlib requests
)

REM Check originpro
"%PY%" -c "import originpro" 2>nul
if errorlevel 1 (
    echo WARNING: originpro package not found
    echo Origin features will be unavailable
    echo Install with: pip install originpro
) else (
    echo [checks] originpro found
)

REM ==========================================================================
REM Start Server
REM ==========================================================================
echo.
echo ================================================
echo Starting OriginMCP Server...
echo ================================================
echo.
echo Config file: %SCRIPT_DIR%\origin_mcp.ini (if exists)
echo Export mode: %EXPORT_MODE%
echo Export dir:  %FILE_EXPORT_DIR%
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

REM Start the server
"%PY%" -u "%SV%"

REM If we get here, the server stopped
echo.
echo ================================================
echo Server stopped
echo ================================================
pause
