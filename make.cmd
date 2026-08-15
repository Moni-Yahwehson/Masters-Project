@echo off
REM Anima — Windows entrypoint shim.
REM
REM Lets a Windows user type the same command the docs give everyone else:
REM
REM     make setup
REM     make dev
REM
REM Windows resolves `make` to this file, which hands off to make.ps1.
REM
REM -ExecutionPolicy Bypass applies to this one invocation only. It changes
REM nothing system-wide, and it is what stops a fresh Windows install from
REM refusing to run the script at all — the single most common reason a
REM PowerShell script "does nothing" when double-clicked.
REM
REM Note for anyone editing: %~dp0 already ends with a backslash.

setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make.ps1" %*
exit /b %ERRORLEVEL%
