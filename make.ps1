<#
.SYNOPSIS
    Anima — Multi-Agent LSD Detection. Windows entrypoint.

.DESCRIPTION
    The native-Windows equivalent of the Makefile, with the same target names,
    so the documentation reads the same on every platform:

        .\make.ps1              list the targets
        .\make.ps1 setup        install EVERYTHING and get the pipeline runnable
        .\make.ps1 dev          run backend + frontend
        .\make.ps1 test         run the test suite

    Or, from cmd.exe, use the wrapper and drop the extension entirely:

        make setup

    This exists because the Makefile needs GNU make and bash, which Windows does
    not ship. Rather than asking the owner to install Git Bash or WSL first, this
    script does the same work with what Windows already has: PowerShell plus one
    of winget, Chocolatey or Scoop for system packages.

    It is safe to re-run — anything already present is detected and skipped.

.PARAMETER Target
    The target to run. Defaults to 'help'.

.PARAMETER Python
    Path to a specific Python 3.9-3.12 interpreter, when the one you want is not
    on PATH. Equivalent to `make setup PYTHON=...`.

.EXAMPLE
    .\make.ps1 setup
.EXAMPLE
    .\make.ps1 setup -Python "C:\Python312\python.exe"
.EXAMPLE
    .\make.ps1 dev
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Target = 'help',

    [string] $Python,

    [int] $BackendPort = 8000,
    [int] $FrontendPort = 5173,

    # Print the install commands instead of running them.
    [switch] $DryRun,

    # Skip the confirmation prompt before installing system packages.
    [switch] $Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Everything is resolved relative to the project root, so the script works no
# matter where it is invoked from.
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$Backend       = Join-Path $Root 'backend'
$Frontend      = Join-Path $Root 'frontend'
$Venv          = Join-Path $Backend '.venv'
$VenvBin       = Join-Path $Venv 'Scripts'          # Windows venv layout
$VenvPython    = Join-Path $VenvBin 'python.exe'
$VenvUvicorn   = Join-Path $VenvBin 'uvicorn.exe'
$ModelPath     = Join-Path $Root 'vision\cattle_lsd_detector.keras'
$IndexPath     = Join-Path $Root 'vision\visual_index.npz'

# TensorFlow publishes wheels for these only.
$PySupported   = @('3.12', '3.11', '3.10', '3.9')
$NodeMinMajor  = 18

# ---------------------------------------------------------------- output ----

function Write-Step { param([string] $Text) Write-Host ''; Write-Host $Text -ForegroundColor White }
function Write-Ok   { param([string] $Text) Write-Host '  ok       ' -ForegroundColor Green -NoNewline; Write-Host $Text }
function Write-Miss { param([string] $Text) Write-Host '  missing  ' -ForegroundColor Yellow -NoNewline; Write-Host $Text }
function Write-Bad  { param([string] $Text) Write-Host '  failed   ' -ForegroundColor Red -NoNewline; Write-Host $Text }
function Write-Dim  { param([string] $Text) Write-Host $Text -ForegroundColor DarkGray }

function Test-Command {
    param([string] $Name)
    return [bool] (Get-Command $Name -ErrorAction SilentlyContinue)
}

# ------------------------------------------------------------------- env ----

<#
    Read backend\.env so model names here always match what the service uses.
    Mirrors backend/app/config.py: real environment variables win, quotes are
    stripped, and trailing ` # comment` is dropped.
#>
function Get-DotEnv {
    $settings = @{}
    $envFile = Join-Path $Backend '.env'
    if (-not (Test-Path -LiteralPath $envFile)) { return $settings }

    foreach ($rawLine in Get-Content -LiteralPath $envFile) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }
        if ($line.StartsWith('export ')) { $line = $line.Substring(7).TrimStart() }

        $split = $line.IndexOf('=')
        if ($split -lt 1) { continue }

        $key = $line.Substring(0, $split).Trim()
        $value = $line.Substring($split + 1).Trim()
        if ($value.Length -ge 2 -and $value[0] -eq $value[-1] -and ($value[0] -eq '"' -or $value[0] -eq "'")) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        elseif ($value.Contains(' #')) {
            $value = $value.Split(' #')[0].Trim()
        }
        $settings[$key] = $value
    }
    return $settings
}

$DotEnv = Get-DotEnv

function Get-Setting {
    param([string] $Key, [string] $Default)
    $fromEnv = [Environment]::GetEnvironmentVariable($Key)
    if ($fromEnv) { return $fromEnv }
    if ($DotEnv.ContainsKey($Key) -and $DotEnv[$Key]) { return $DotEnv[$Key] }
    return $Default
}

$VisionModel = Get-Setting 'LSD_VISION_MODEL' 'gemma3:4b'
$TextModel   = Get-Setting 'LSD_TEXT_MODEL'   'gemma3:4b'
$OllamaUrl   = (Get-Setting 'LSD_LLM_BASE_URL' 'http://127.0.0.1:11434').TrimEnd('/')
# The two roles usually point at one multimodal model; pull it once.
$Models      = @($VisionModel, $TextModel) | Sort-Object -Unique

# --------------------------------------------------- package management ----

function Get-PackageManager {
    if (Test-Command 'winget') { return 'winget' }
    if (Test-Command 'choco')  { return 'choco' }
    if (Test-Command 'scoop')  { return 'scoop' }
    return $null
}

function Install-Package {
    param([string] $Manager, [string[]] $Ids)

    switch ($Manager) {
        'winget' { $exe = 'winget'; $argv = @('install', '--accept-source-agreements',
                                              '--accept-package-agreements', '-e', '--id') + $Ids }
        'choco'  { $exe = 'choco';  $argv = @('install', '-y') + $Ids }
        'scoop'  { $exe = 'scoop';  $argv = @('install') + $Ids }
        default  { return $false }
    }

    if ($DryRun) {
        Write-Dim "  [dry-run] $exe $($argv -join ' ')"
        return $true
    }

    Write-Dim "  running: $exe $($argv -join ' ')"
    & $exe @argv
    # winget returns non-zero for "already installed"; treat only a hard failure
    # as fatal, and let the follow-up check decide whether it really worked.
    return ($LASTEXITCODE -eq 0)
}

function Confirm-Install {
    param([string] $What)
    if ($Yes -or $DryRun) { return $true }
    Write-Host ''
    $answer = Read-Host "Install $What now? [Y/n]"
    return ($answer -eq '' -or $answer -match '^[Yy]')
}

function Show-Manual {
    param([string] $What, [string] $Url)
    Write-Host ''
    Write-Host "Cannot install automatically on this system." -ForegroundColor Yellow
    Write-Host "Install $What manually:"
    Write-Host "  $Url"
}

# ---------------------------------------------------------------- python ----

<#
    Return the path of a supported interpreter, or $null.

    Tries the `py` launcher first because it is the reliable way to reach a
    specific version on Windows, then bare names on PATH. The Microsoft Store
    stub `python.exe` is skipped: it exits 9009 and opens the Store instead of
    running, which would otherwise look like a working interpreter.
#>
function Find-Python {
    if ($Python) {
        if (Test-Path -LiteralPath $Python) {
            $version = Get-PythonVersion $Python
            if ($version -and $PySupported -contains $version) { return $Python }
            throw "The interpreter at '$Python' is Python $version. TensorFlow needs one of: $($PySupported -join ', ')."
        }
        throw "No interpreter found at '$Python'."
    }

    if (Test-Command 'py') {
        foreach ($version in $PySupported) {
            $resolved = & py "-$version" -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) { return $resolved.Trim() }
        }
    }

    foreach ($candidate in @('python3.12', 'python3.11', 'python3.10', 'python3.9', 'python3', 'python')) {
        if (-not (Test-Command $candidate)) { continue }
        $source = (Get-Command $candidate).Source
        if ($source -and $source -like '*WindowsApps*') { continue }   # Store stub
        $version = Get-PythonVersion $candidate
        if ($version -and $PySupported -contains $version) { return $source }
    }
    return $null
}

function Get-PythonVersion {
    param([string] $Exe)
    try {
        $out = & $Exe -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        if ($LASTEXITCODE -eq 0) { return $out.Trim() }
    }
    catch { }
    return $null
}

function Test-PythonDep {
    $found = Find-Python
    if ($found) {
        Write-Ok "python    $(& $found --version 2>&1)  $found"
        return $true
    }
    Write-Miss "python    not found (need one of: $($PySupported -join ', '))"
    return $false
}

function Install-PythonDep {
    param([string] $Manager)
    switch ($Manager) {
        'winget' { return (Install-Package $Manager @('Python.Python.3.12')) }
        'choco'  { return (Install-Package $Manager @('python312')) }
        'scoop'  { return (Install-Package $Manager @('python')) }
        default  { Show-Manual 'Python 3.12' 'https://www.python.org/downloads/windows/'; return $false }
    }
}

# ------------------------------------------------------------------ node ----

function Test-NodeDep {
    if (Test-Command 'node') {
        $raw = (& node --version) -replace '^v', ''
        $major = [int] ($raw.Split('.')[0])
        if ($major -ge $NodeMinMajor) {
            Write-Ok "node      v$raw"
            return $true
        }
        Write-Miss "node      v$raw is older than v$NodeMinMajor"
        return $false
    }
    Write-Miss "node      not found (need v$NodeMinMajor+)"
    return $false
}

function Install-NodeDep {
    param([string] $Manager)
    switch ($Manager) {
        'winget' { return (Install-Package $Manager @('OpenJS.NodeJS.LTS')) }
        'choco'  { return (Install-Package $Manager @('nodejs-lts')) }
        'scoop'  { return (Install-Package $Manager @('nodejs-lts')) }
        default  { Show-Manual "Node $NodeMinMajor+" 'https://nodejs.org/en/download'; return $false }
    }
}

# ---------------------------------------------------------------- ollama ----

function Test-OllamaDep {
    if (Test-Command 'ollama') {
        Write-Ok "ollama    $((& ollama --version 2>&1) | Select-Object -First 1)"
        return $true
    }
    Write-Miss 'ollama    not found'
    return $false
}

function Install-OllamaDep {
    param([string] $Manager)
    switch ($Manager) {
        'winget' { return (Install-Package $Manager @('Ollama.Ollama')) }
        'choco'  { return (Install-Package $Manager @('ollama')) }
        'scoop'  { return (Install-Package $Manager @('ollama')) }
        default  { Show-Manual 'Ollama' 'https://ollama.com/download/windows'; return $false }
    }
}

function Test-OllamaRunning {
    try {
        $null = Invoke-WebRequest -Uri "$OllamaUrl/api/tags" -TimeoutSec 3 -UseBasicParsing
        return $true
    }
    catch { return $false }
}

function Get-InstalledModels {
    if (-not (Test-Command 'ollama')) { return @() }
    $lines = & ollama list 2>$null
    if (-not $lines -or $lines.Count -lt 2) { return @() }
    # Skip the header row; the model name is the first whitespace-delimited field.
    return $lines | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ }
}

function Start-OllamaServer {
    if (Test-OllamaRunning) {
        Write-Ok "ollama server already running at $OllamaUrl"
        return $true
    }
    if (-not (Test-Command 'ollama')) {
        Write-Bad 'ollama is not installed — run: .\make.ps1 setup'
        return $false
    }

    Write-Step 'Starting the Ollama server'
    $logDir = Join-Path $Root '.ollama'
    if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

    # Detached and windowless so it outlives this script, matching the `nohup`
    # behaviour of the bash bootstrap.
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'server.log') `
        -RedirectStandardError  (Join-Path $logDir 'server.err.log') | Out-Null

    for ($waited = 1; $waited -le 30; $waited++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaRunning) {
            Write-Ok "ollama server up after ${waited}s  (log: .ollama\server.log)"
            return $true
        }
    }
    Write-Bad 'ollama did not come up within 30s — see .ollama\server.err.log'
    return $false
}

# --------------------------------------------------------------- targets ----

function Invoke-Help {
    Write-Host 'Anima — Multi-Agent LSD Detection' -ForegroundColor White
    Write-Dim 'All inference runs locally. No API keys required.'
    Write-Host ''
    Write-Host 'First time:  ' -NoNewline; Write-Host '.\make.ps1 setup' -ForegroundColor White -NoNewline
    Write-Dim '   (installs everything, ~10 min)'
    Write-Host 'Then:        ' -NoNewline; Write-Host '.\make.ps1 dev' -ForegroundColor White
    Write-Host ''

    $targets = [ordered] @{
        'setup'          = 'Install everything needed to run the pipeline'
        'setup-offline'  = 'Setup without system packages or downloads'
        'system-deps'    = 'Install Python, Node and Ollama via winget/choco/scoop'
        'setup-backend'  = 'Create the Python venv and install backend dependencies'
        'setup-frontend' = 'Install frontend dependencies'
        'models'         = 'Download the local model weights (several GB, one time)'
        'index'          = 'Build the precedent index for the visual evidence panel'
        'serve-ollama'   = 'Start the local model server in the background'
        'dev'            = 'Run backend + frontend together'
        'backend'        = "Run the API only (port $BackendPort)"
        'frontend'       = "Run the Vue dev server only (port $FrontendPort)"
        'stop'           = 'Stop stray dev servers'
        'test'           = 'Run the backend test suite (needs no models)'
        'evaluate'       = 'Held-out evaluation of the trained CNN'
        'health'         = 'Report which components are live'
        'doctor'         = 'Diagnose the setup'
        'clean'          = 'Remove caches and build artefacts'
        'clean-all'      = 'Also remove the venv and node_modules'
    }
    foreach ($name in $targets.Keys) {
        Write-Host ('  {0,-16}' -f $name) -ForegroundColor Green -NoNewline
        Write-Host $targets[$name]
    }
    Write-Host ''
    Write-Dim "platform: windows · models: $VisionModel (vision), $TextModel (text)"
    Write-Dim 'From cmd.exe you can also just type:  make <target>'
}

function Invoke-SystemDeps {
    Write-Step 'System dependencies'
    $manager = Get-PackageManager
    if ($manager) { Write-Dim "  package manager: $manager" }
    else { Write-Dim '  package manager: none found (winget, choco or scoop)' }

    $missing = @()
    if (-not (Test-PythonDep)) { $missing += 'python' }
    if (-not (Test-NodeDep))   { $missing += 'node' }
    if (-not (Test-OllamaDep)) { $missing += 'ollama' }

    if ($missing.Count -eq 0) { return $true }
    if ($env:SKIP_SYSTEM_DEPS -eq '1') {
        Write-Dim '  skipping installation (SKIP_SYSTEM_DEPS=1)'
        return $false
    }
    if (-not $manager) {
        Show-Manual 'the tools listed above' 'https://www.python.org/downloads/windows/ , https://nodejs.org/en/download , https://ollama.com/download/windows'
        return $false
    }
    if (-not (Confirm-Install ($missing -join ', '))) {
        Write-Dim '  skipped by request'
        return $false
    }

    foreach ($tool in $missing) {
        Write-Step "Installing $tool"
        switch ($tool) {
            'python' { Install-PythonDep $manager | Out-Null }
            'node'   { Install-NodeDep   $manager | Out-Null }
            'ollama' { Install-OllamaDep $manager | Out-Null }
        }
    }

    # Installers extend PATH for *new* processes only, so re-read it here rather
    # than telling the user to open a new terminal.
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')

    Write-Step 'Re-checking'
    $ok = (Test-PythonDep) -and (Test-NodeDep) -and (Test-OllamaDep)
    if (-not $ok) {
        Write-Host ''
        Write-Host 'Some tools are still missing. If they were just installed, close this' -ForegroundColor Yellow
        Write-Host 'window, open a new one, and run the command again.' -ForegroundColor Yellow
    }
    return $ok
}

function Assert-Venv {
    if (Test-Path -LiteralPath $VenvPython) { return }

    $python = Find-Python
    if (-not $python) {
        Write-Host 'No supported Python found.' -ForegroundColor Red -NoNewline
        Write-Host " Need one of: $($PySupported -join ', ')"
        Write-Host 'Run .\make.ps1 system-deps, or pass one: .\make.ps1 setup -Python C:\path\to\python.exe'
        throw 'python missing'
    }
    Write-Step "Creating virtualenv ($python, Python $(Get-PythonVersion $python))"
    & $python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}

function Invoke-SetupBackend {
    Assert-Venv
    Write-Step 'Installing backend dependencies'
    Write-Dim '  (TensorFlow is large — a few minutes)'
    & $VenvPython -m pip install --quiet --upgrade pip
    & $VenvPython -m pip install --quiet -r (Join-Path $Backend 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

    $envFile = Join-Path $Backend '.env'
    if (-not (Test-Path -LiteralPath $envFile)) {
        Copy-Item (Join-Path $Backend '.env.example') $envFile
        Write-Dim '  created backend\.env from the example'
    }
    Write-Ok "backend ($(& $VenvPython --version 2>&1))"
}

function Invoke-SetupFrontend {
    Write-Step 'Installing frontend dependencies'
    if (-not (Test-Command 'npm')) {
        Write-Bad 'npm not found — run .\make.ps1 system-deps'
        throw 'npm missing'
    }
    Push-Location $Frontend
    try {
        & npm install --silent --no-fund --no-audit
        if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
    }
    finally { Pop-Location }
    Write-Ok "frontend (node $(& node --version))"
}

function Invoke-Models {
    Write-Step 'Local model weights'
    if (-not (Test-Command 'ollama')) {
        Write-Bad 'ollama not installed — run .\make.ps1 system-deps'
        throw 'ollama missing'
    }
    Start-OllamaServer | Out-Null

    $installed = Get-InstalledModels
    foreach ($model in $Models) {
        if ($installed -contains $model) {
            Write-Ok "$model (already downloaded)"
        }
        else {
            Write-Dim "  pulling  $model"
            & ollama pull $model
        }
    }
}

function Invoke-Index {
    Assert-Venv
    Write-Step 'Precedent index'
    Write-Dim '  (embeds the dataset — about 20 s)'
    Push-Location $Backend
    try { & $VenvPython (Join-Path 'scripts' 'build_visual_index.py') }
    finally { Pop-Location }
}

function Invoke-Setup {
    Write-Host 'Anima setup' -ForegroundColor White -NoNewline
    Write-Dim '  platform: windows'

    if (-not (Invoke-SystemDeps)) {
        Write-Host ''
        Write-Host 'Setup stopped:' -ForegroundColor Red -NoNewline
        Write-Host ' the system dependencies above are missing.'
        Write-Host 'Install them (see the links above), then re-run .\make.ps1 setup'
        Write-Dim 'Already have them elsewhere? Try: .\make.ps1 setup -Python C:\path\to\python.exe'
        exit 1
    }

    Invoke-SetupBackend
    Invoke-SetupFrontend
    Invoke-Models
    try { Invoke-Index }
    catch { Write-Miss 'precedent index — run .\make.ps1 index later' }

    Write-Host ''
    Write-Host 'Setup complete.' -ForegroundColor Green
    Invoke-Doctor
    Write-Host ''
    Write-Host 'Start it with: ' -NoNewline
    Write-Host '.\make.ps1 dev' -ForegroundColor White
}

function Invoke-SetupOffline {
    $env:SKIP_SYSTEM_DEPS = '1'
    try { Invoke-SystemDeps | Out-Null } catch { }
    Invoke-SetupBackend
    Invoke-SetupFrontend
    Write-Host ''
    Write-Host 'Code dependencies installed.' -ForegroundColor Green
    Write-Dim 'Run .\make.ps1 models when you have network to pull the weights.'
}

function Assert-VenvPresent {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host 'No virtualenv.' -ForegroundColor Red -NoNewline
        Write-Host ' Run .\make.ps1 setup first.'
        exit 1
    }
}

function Invoke-Dev {
    Assert-VenvPresent
    Write-Host 'Backend  ' -ForegroundColor White -NoNewline
    Write-Host "http://127.0.0.1:$BackendPort  " -NoNewline
    Write-Dim '(API docs at /docs)'
    Write-Host 'Frontend ' -ForegroundColor White -NoNewline
    Write-Host "http://localhost:$FrontendPort"
    Write-Dim 'Ctrl-C stops both.'
    Write-Host ''

    $processes = @()
    try {
        $processes += Start-Process -FilePath $VenvUvicorn `
            -ArgumentList 'app.main:app', '--reload', '--port', $BackendPort `
            -WorkingDirectory $Backend -NoNewWindow -PassThru

        # npm is a .cmd shim, so it has to be launched through cmd.exe.
        $processes += Start-Process -FilePath 'cmd.exe' `
            -ArgumentList '/c', 'npm', 'run', 'dev', '--', '--port', $FrontendPort `
            -WorkingDirectory $Frontend -NoNewWindow -PassThru

        Wait-Process -Id ($processes | ForEach-Object { $_.Id })
    }
    finally {
        # Ctrl-C lands here, so neither server is left orphaned holding a port.
        foreach ($process in $processes) {
            if ($process -and -not $process.HasExited) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
        Invoke-Stop
    }
}

function Invoke-Backend {
    Assert-VenvPresent
    Push-Location $Backend
    try { & $VenvUvicorn 'app.main:app' '--reload' '--port' $BackendPort }
    finally { Pop-Location }
}

function Invoke-Frontend {
    Push-Location $Frontend
    try { & npm run dev -- --port $FrontendPort }
    finally { Pop-Location }
}

function Invoke-Stop {
    # Match on the command line rather than the image name: the servers run as
    # python.exe and node.exe, and killing those wholesale would take out
    # unrelated work the user has open.
    $stopped = 0
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction Stop |
                 Where-Object { $_.CommandLine -and (
                     ($_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'app\.main:app') -or
                     $_.CommandLine -match 'vite'
                 ) }
        foreach ($proc in $procs) {
            if ($proc.ProcessId -eq $PID) { continue }
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
    catch {
        # Win32_Process is Windows-only. Never let a failed cleanup sweep look
        # like the command itself broke.
        Write-Dim "  could not enumerate processes ($($_.Exception.Message))"
        return
    }
    if ($stopped -gt 0) { Write-Host "stopped $stopped dev server process(es)" }
    else { Write-Dim '  no dev servers running' }
}

function Invoke-Test {
    Assert-VenvPresent
    Push-Location $Backend
    try { & $VenvPython -m pytest tests -q }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Evaluate {
    Assert-VenvPresent
    Push-Location $Backend
    try { & $VenvPython (Join-Path 'scripts' 'evaluate_model.py') }
    finally { Pop-Location }
}

function Invoke-Health {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/api/health" `
                                      -TimeoutSec 5 -UseBasicParsing
        $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 6
    }
    catch {
        Write-Host "Backend not responding on port $BackendPort." -ForegroundColor Yellow -NoNewline
        Write-Host ' Start it with: .\make.ps1 backend'
    }
}

function Invoke-Doctor {
    Write-Step 'System dependencies'
    $manager = Get-PackageManager
    if ($manager) { Write-Dim "  package manager: $manager" }
    Test-PythonDep | Out-Null
    Test-NodeDep   | Out-Null
    Test-OllamaDep | Out-Null

    Write-Step 'Project'
    if (Test-Path -LiteralPath $VenvPython) { Write-Ok "venv ($(& $VenvPython --version 2>&1))" }
    else { Write-Miss 'venv — run .\make.ps1 setup' }

    if (Test-Path -LiteralPath (Join-Path $Frontend 'node_modules')) { Write-Ok 'frontend dependencies' }
    else { Write-Miss 'frontend dependencies — run .\make.ps1 setup' }

    if (Test-Path -LiteralPath $ModelPath) { Write-Ok 'trained CNN' }
    else { Write-Bad "missing: vision\cattle_lsd_detector.keras" }

    if (Test-Path -LiteralPath $IndexPath) { Write-Ok 'precedent index' }
    else { Write-Miss 'precedent index — run .\make.ps1 index (optional)' }

    Write-Step "Local model server ($OllamaUrl)"
    if (Test-OllamaRunning) {
        Write-Ok 'running'
        $installed = Get-InstalledModels
        foreach ($model in $Models) {
            if ($installed -contains $model) { Write-Ok $model }
            else { Write-Miss "$model — run .\make.ps1 models" }
        }
    }
    else {
        Write-Miss 'stopped — start with .\make.ps1 serve-ollama'
    }
}

function Invoke-Clean {
    # Walk only the backend, and skip .venv: __pycache__ exists nowhere else,
    # and recursing through node_modules or site-packages turns a one-second
    # clean into a minute of directory enumeration on Windows.
    Get-ChildItem -LiteralPath $Backend -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike "$Venv*" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($path in @((Join-Path $Backend '.pytest_cache'), (Join-Path $Frontend 'dist'))) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
    Write-Host 'cleaned' -ForegroundColor Green
}

function Invoke-CleanAll {
    Invoke-Clean
    foreach ($path in @($Venv, (Join-Path $Frontend 'node_modules'))) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
    Write-Host 'removed venv and node_modules' -ForegroundColor Green
}

# -------------------------------------------------------------- dispatch ----

# Everything below runs inside one handler so a failure reaches the owner as a
# single readable line rather than a PowerShell stack trace. Use -Verbose for
# the full record when something needs actual debugging.
try {

switch ($Target.ToLowerInvariant()) {
    'help'           { Invoke-Help }
    '--help'         { Invoke-Help }
    '-h'             { Invoke-Help }
    'setup'          { Invoke-Setup }
    'setup-offline'  { Invoke-SetupOffline }
    'system-deps'    { Invoke-SystemDeps | Out-Null }
    'setup-backend'  { Invoke-SetupBackend }
    'setup-frontend' { Invoke-SetupFrontend }
    'models'         { Invoke-Models }
    'index'          { Invoke-Index }
    'serve-ollama'   { Start-OllamaServer | Out-Null }
    'dev'            { Invoke-Dev }
    'backend'        { Invoke-Backend }
    'frontend'       { Invoke-Frontend }
    'stop'           { Invoke-Stop }
    'test'           { Invoke-Test }
    'evaluate'       { Invoke-Evaluate }
    'health'         { Invoke-Health }
    'doctor'         { Invoke-Doctor }
    'clean'          { Invoke-Clean }
    'clean-all'      { Invoke-CleanAll }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Write-Host ''
        Invoke-Help
        exit 2
    }
}

}
catch {
    Write-Host ''
    Write-Host 'Failed: ' -ForegroundColor Red -NoNewline
    Write-Host $_.Exception.Message
    Write-Dim "  target: $Target"
    Write-Verbose ($_ | Out-String)
    Write-Dim '  run .\make.ps1 doctor to check the setup'
    exit 1
}
