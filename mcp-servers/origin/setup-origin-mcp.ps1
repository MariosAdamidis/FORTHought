# OriginMCP Setup Script v10.2 - Portable Edition
# Run as Administrator on any Windows machine with Origin 2021+
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs required packages
# 3. Configures Windows Firewall
# 4. Creates launcher files
#
# To transfer to a new server:
# 1. Copy this script, server.py, and origin_mcp.ini to the new server
# 2. Edit the configuration below
# 3. Run this script as Administrator
# 4. Start the server with start-origin-mcp.bat

param(
    [string]$BaseDir = "",
    [int]$ServerPort = 12009
)

Write-Host "=== OriginMCP Setup v10.2 (Portable) ===" -ForegroundColor Cyan

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Please run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    pause
    exit 1
}

# =============================================================================
# CONFIGURATION - Edit these for your server
# =============================================================================
# Base directory - defaults to script location or E:\OriginPro
if (-not $BaseDir) {
    $scriptPath = $PSScriptRoot
    if ($scriptPath -and (Test-Path $scriptPath)) {
        $BaseDir = $scriptPath
    } else {
        $BaseDir = "E:\OriginPro"
    }
}

$VENV_DIR = "$BaseDir\venv"
$SERVER_FILE = "$BaseDir\server.py"
$CONFIG_FILE = "$BaseDir\origin_mcp.ini"
$SECRETS_DIR = "$BaseDir\secrets"
$EXPORTS_DIR = "$BaseDir\exports"
$SHORTCUT_PATH = "$BaseDir\OriginMCP.lnk"
$BATCH_FILE = "$BaseDir\start-origin-mcp.bat"

# Try to get Tailscale IP automatically
try {
    $tailscaleIP = (Get-NetIPAddress -InterfaceAlias "Tailscale*" -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress | Select-Object -First 1
} catch {
    $tailscaleIP = $null
}

if (-not $tailscaleIP) {
    $tailscaleIP = "YOUR_TAILSCALE_IP"
    Write-Host "NOTE: Could not detect Tailscale IP. Update MetaMCP config manually." -ForegroundColor Yellow
}

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  Base Directory: $BaseDir"
Write-Host "  Server Port:    $ServerPort"
Write-Host "  Tailscale IP:   $tailscaleIP"

# =============================================================================
# Create directories
# =============================================================================
Write-Host "`n[1/6] Creating directories..." -ForegroundColor Yellow

$dirsToCreate = @($BaseDir, $SECRETS_DIR, $EXPORTS_DIR)
foreach ($dir in $dirsToCreate) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

Set-Location $BaseDir

# =============================================================================
# Check Python
# =============================================================================
Write-Host "`n[2/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    pause
    exit 1
}

# =============================================================================
# Create/update virtual environment
# =============================================================================
Write-Host "`n[3/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "  Creating venv at $VENV_DIR..." -ForegroundColor Cyan
    python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "  Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  Virtual environment already exists" -ForegroundColor Green
}

# =============================================================================
# Install packages
# =============================================================================
Write-Host "`n[4/6] Installing Python packages..." -ForegroundColor Yellow
$pipExe = "$VENV_DIR\Scripts\pip.exe"

Write-Host "  Upgrading pip..."
& "$VENV_DIR\Scripts\python.exe" -m pip install --upgrade pip --quiet

$packages = @(
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "requests>=2.31",
    "originpro"
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg..."
    & $pipExe install --quiet --upgrade $pkg
}

Write-Host "  Packages installed" -ForegroundColor Green

# =============================================================================
# Configure Windows Firewall
# =============================================================================
Write-Host "`n[5/6] Configuring firewall..." -ForegroundColor Yellow

$ruleName = "OriginMCP Server (Port $ServerPort)"
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existingRule) {
    Remove-NetFirewallRule -DisplayName $ruleName
    Write-Host "  Removed old firewall rule" -ForegroundColor Gray
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $ServerPort `
    -Profile Any | Out-Null

Write-Host "  Firewall rule added for port $ServerPort" -ForegroundColor Green

# =============================================================================
# Create config file if it doesn't exist
# =============================================================================
Write-Host "`n[6/6] Creating configuration files..." -ForegroundColor Yellow

if (-not (Test-Path $CONFIG_FILE)) {
    $configContent = @"
; OriginMCP Server Configuration v10.2
; Generated by setup script

[network]
owui_url = http://100.117.144.23:8081
owui_public_url = https://files.forthought.cc
server_host = 0.0.0.0
server_port = $ServerPort

[authentication]
jwt_token = 
secrets_dir = $SECRETS_DIR

[paths]
export_dir = $EXPORTS_DIR

[options]
export_mode = owui
debug_traceback = 0
max_error_chars = 300
max_tool_text_chars = 4000
"@
    Set-Content -Path $CONFIG_FILE -Value $configContent -Encoding UTF8
    Write-Host "  Created: $CONFIG_FILE" -ForegroundColor Green
} else {
    Write-Host "  Config file already exists: $CONFIG_FILE" -ForegroundColor Gray
}

# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($SHORTCUT_PATH)
$Shortcut.TargetPath = $BATCH_FILE
$Shortcut.WorkingDirectory = $BaseDir
$Shortcut.IconLocation = "C:\Windows\System32\SHELL32.dll,21"
$Shortcut.Description = "Launch OriginMCP Server v10.2"
$Shortcut.Save()

Write-Host "  Created shortcut: $SHORTCUT_PATH" -ForegroundColor Green

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

Write-Host "`nServer Information:" -ForegroundColor Yellow
Write-Host "  Base Directory: $BaseDir"
Write-Host "  Config File:    $CONFIG_FILE"
Write-Host "  Server Port:    $ServerPort"
if ($tailscaleIP -ne "YOUR_TAILSCALE_IP") {
    Write-Host "  Tailscale URL:  http://${tailscaleIP}:$ServerPort/mcp"
}

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "  1. Ensure OriginPro 2021+ is installed"
Write-Host "  2. Place your JWT token in: $SECRETS_DIR\owui_jwt.txt"
Write-Host "  3. Copy server.py to: $BaseDir"
Write-Host "  4. Double-click start-origin-mcp.bat to start"

if ($tailscaleIP -ne "YOUR_TAILSCALE_IP") {
    Write-Host "`nMetaMCP Configuration:" -ForegroundColor Cyan
    Write-Host @"

Add this to your MetaMCP config:

{
  "name": "spec",
  "call_template_type": "mcp",
  "config": {
    "mcpServers": {
      "spec": {
        "transport": "http",
        "timeout": 300000,
        "url": "http://${tailscaleIP}:$ServerPort/mcp"
      }
    }
  },
  "timeout": 300000
}

"@
}

Write-Host "`nTo transfer to another server:" -ForegroundColor Yellow
Write-Host "  1. Copy: server.py, start-origin-mcp.bat, origin_mcp.ini, setup-origin-mcp.ps1"
Write-Host "  2. Edit origin_mcp.ini with new paths"
Write-Host "  3. Run setup-origin-mcp.ps1 on the new server"
Write-Host "  4. Copy your JWT token to secrets\owui_jwt.txt"

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
