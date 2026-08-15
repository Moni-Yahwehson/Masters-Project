#!/usr/bin/env bash
#
# Cross-platform system dependency bootstrap for the Anima pipeline.
#
# Ensures Python (3.9-3.12), Node 18+ and Ollama are present, using whichever
# package manager the host provides. Runs on macOS, Linux (apt/dnf/yum/pacman/
# zypper/apk) and Windows via Git Bash / MSYS2 / WSL (winget/choco/scoop).
#
#   ./scripts/system-deps.sh check          report status, install nothing
#   ./scripts/system-deps.sh ensure         install whatever is missing
#   ./scripts/system-deps.sh ensure python  install just one tool
#   ./scripts/system-deps.sh serve-ollama   start the model server, wait for it
#
# Honours:
#   SKIP_SYSTEM_DEPS=1   report what's missing, install nothing, exit non-zero
#   ASSUME_YES=1         don't prompt before installing (CI)
#   DRY_RUN=1            print the install commands instead of running them
#   FORCE_PLATFORM=...   override detection (macos|linux|windows) — for testing
#   FORCE_PKG=...        override package-manager detection — for testing
#
# Never installs silently over an existing tool, and never guesses: if no
# supported package manager is found it prints the exact manual step instead.

set -euo pipefail

PY_SUPPORTED=("3.9" "3.10" "3.11" "3.12")
NODE_MIN_MAJOR=18

# ---------------------------------------------------------------- output ----
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[32m'
  YLW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
else
  BOLD=""; DIM=""; GRN=""; YLW=""; RED=""; OFF=""
fi

info() { printf '%s\n' "$*"; }
ok()   { printf '  %sok%s       %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '  %smissing%s  %s\n' "$YLW" "$OFF" "$*"; }
bad()  { printf '  %sfailed%s   %s\n' "$RED" "$OFF" "$*"; }
step() { printf '\n%s%s%s\n' "$BOLD" "$*" "$OFF"; }

have() { command -v "$1" >/dev/null 2>&1; }

# -------------------------------------------------------------- platform ----
detect_platform() {
  case "$(uname -s 2>/dev/null || echo unknown)" in
    Darwin) echo macos ;;
    Linux)
      # WSL is Linux for our purposes — the Linux installers work there.
      echo linux ;;
    MINGW*|MSYS*|CYGWIN*) echo windows ;;
    *)
      # `make` on native Windows sets OS=Windows_NT even without uname.
      if [ "${OS:-}" = "Windows_NT" ]; then echo windows; else echo unknown; fi ;;
  esac
}

PLATFORM="${FORCE_PLATFORM:-$(detect_platform)}"

detect_package_manager() {
  case "$PLATFORM" in
    macos)
      have brew && { echo brew; return; } ;;
    linux)
      # Homebrew on Linux takes precedence when present: no sudo needed.
      have brew     && { echo brew;     return; }
      have apt-get  && { echo apt;      return; }
      have dnf      && { echo dnf;      return; }
      have yum      && { echo yum;      return; }
      have pacman   && { echo pacman;   return; }
      have zypper   && { echo zypper;   return; }
      have apk      && { echo apk;      return; } ;;
    windows)
      have winget   && { echo winget;   return; }
      have choco    && { echo choco;    return; }
      have scoop    && { echo scoop;    return; } ;;
  esac
  echo none
}

PKG="${FORCE_PKG:-$(detect_package_manager)}"

# Package managers that need root, when we are not already root.
needs_sudo() {
  case "$PKG" in
    apt|dnf|yum|pacman|zypper|apk) [ "$(id -u 2>/dev/null || echo 1)" != "0" ] ;;
    *) return 1 ;;
  esac
}

SUDO=""
if needs_sudo; then
  if have sudo; then SUDO="sudo"; else
    printf '%sNo sudo available and %s needs root.%s\n' "$YLW" "$PKG" "$OFF"
  fi
fi

confirm() {
  [ "${ASSUME_YES:-0}" = "1" ] && return 0
  [ -t 0 ] || return 0   # non-interactive (CI, piped): proceed
  printf '%s [Y/n] ' "$1"
  read -r reply || reply=""
  case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

run_pkg() {
  step "Installing $* with $PKG"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '  %s[dry-run]%s %s install %s\n' "$DIM" "$OFF" "$PKG" "$*"
    return 0
  fi
  case "$PKG" in
    brew)   brew install "$@" ;;
    apt)    $SUDO apt-get update -qq && $SUDO apt-get install -y "$@" ;;
    dnf)    $SUDO dnf install -y "$@" ;;
    yum)    $SUDO yum install -y "$@" ;;
    pacman) $SUDO pacman -Sy --noconfirm "$@" ;;
    zypper) $SUDO zypper install -y "$@" ;;
    apk)    $SUDO apk add "$@" ;;
    winget) winget install --accept-source-agreements --accept-package-agreements -e --id "$@" ;;
    choco)  choco install -y "$@" ;;
    scoop)  scoop install "$@" ;;
    *)      return 1 ;;
  esac
}

manual() {
  printf '\n%sCannot install automatically on this system.%s\n' "$YLW" "$OFF"
  printf 'Install %s manually:\n  %s\n' "$1" "$2"
  return 1
}

# ---------------------------------------------------------------- python ----
# Print the path of a supported interpreter, if one exists.
find_python() {
  local candidate version
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
    have "$candidate" || continue
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    for supported in "${PY_SUPPORTED[@]}"; do
      if [ "$version" = "$supported" ]; then command -v "$candidate"; return 0; fi
    done
  done
  return 1
}

check_python() {
  local found
  if found="$(find_python)"; then
    ok "python    $("$found" --version 2>&1)  $found"
    return 0
  fi
  if have python3; then
    warn "python    $(python3 --version 2>&1) found, but TensorFlow needs ${PY_SUPPORTED[*]}"
  else
    warn "python    not found (need one of: ${PY_SUPPORTED[*]})"
  fi
  return 1
}

install_python() {
  case "$PKG" in
    brew)   run_pkg python@3.12 ;;
    apt)    run_pkg python3 python3-venv python3-pip ;;
    dnf|yum) run_pkg python3 python3-pip ;;
    pacman) run_pkg python python-pip ;;
    zypper) run_pkg python3 python3-pip ;;
    apk)    run_pkg python3 py3-pip ;;
    winget) run_pkg Python.Python.3.12 ;;
    choco)  run_pkg python312 ;;
    scoop)  run_pkg python ;;
    *) manual "Python 3.12" "https://www.python.org/downloads/" ;;
  esac
}

# ------------------------------------------------------------------ node ----
check_node() {
  if have node; then
    local major
    major="$(node --version 2>/dev/null | sed 's/^v//; s/\..*//')"
    if [ -n "$major" ] && [ "$major" -ge "$NODE_MIN_MAJOR" ] 2>/dev/null; then
      ok "node      $(node --version)"
      return 0
    fi
    warn "node      $(node --version) is older than v$NODE_MIN_MAJOR"
    return 1
  fi
  warn "node      not found (need v$NODE_MIN_MAJOR+)"
  return 1
}

install_node() {
  case "$PKG" in
    brew)   run_pkg node ;;
    apt)    run_pkg nodejs npm ;;
    dnf|yum) run_pkg nodejs npm ;;
    pacman) run_pkg nodejs npm ;;
    zypper) run_pkg nodejs npm ;;
    apk)    run_pkg nodejs npm ;;
    winget) run_pkg OpenJS.NodeJS.LTS ;;
    choco)  run_pkg nodejs-lts ;;
    scoop)  run_pkg nodejs-lts ;;
    *) manual "Node $NODE_MIN_MAJOR+" "https://nodejs.org/en/download" ;;
  esac
}

# ---------------------------------------------------------------- ollama ----
check_ollama() {
  if have ollama; then
    ok "ollama    $(ollama --version 2>&1 | head -1)"
    return 0
  fi
  warn "ollama    not found"
  return 1
}

install_ollama() {
  case "$PLATFORM" in
    macos)
      if [ "$PKG" = "brew" ]; then run_pkg ollama
      else manual "Ollama" "https://ollama.com/download/mac"; fi ;;
    linux)
      # The official script is the supported path on Linux and handles every
      # distro; distro packages lag badly.
      if have curl || [ "${DRY_RUN:-0}" = "1" ]; then
        step "Installing Ollama (official installer)"
        if [ "${DRY_RUN:-0}" = "1" ]; then
          printf '  %s[dry-run]%s curl -fsSL https://ollama.com/install.sh | sh\n' "$DIM" "$OFF"
          return 0
        fi
        curl -fsSL https://ollama.com/install.sh | sh
      else
        manual "Ollama" "https://ollama.com/download/linux"
      fi ;;
    windows)
      case "$PKG" in
        winget) run_pkg Ollama.Ollama ;;
        choco)  run_pkg ollama ;;
        scoop)  run_pkg ollama ;;
        *) manual "Ollama" "https://ollama.com/download/windows" ;;
      esac ;;
    *) manual "Ollama" "https://ollama.com/download" ;;
  esac
}

# ------------------------------------------------------- ollama lifecycle ----
OLLAMA_URL="${LSD_LLM_BASE_URL:-http://127.0.0.1:11434}"

ollama_running() {
  if have curl; then
    curl -sf -m 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1
  else
    # Fall back to Python: curl is absent on some minimal Windows shells.
    python_probe
  fi
}

python_probe() {
  local py
  py="$(find_python 2>/dev/null || command -v python3 2>/dev/null || command -v python 2>/dev/null)" || return 1
  "$py" - "$OLLAMA_URL" <<'PY' >/dev/null 2>&1
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=3)
except Exception:
    sys.exit(1)
PY
}

serve_ollama() {
  if ollama_running; then
    ok "ollama server already running at $OLLAMA_URL"
    return 0
  fi
  have ollama || { bad "ollama is not installed — run: make setup"; return 1; }

  step "Starting the Ollama server"
  mkdir -p .ollama
  # Detached so it survives this script; log kept for debugging.
  nohup ollama serve >.ollama/server.log 2>&1 &
  disown 2>/dev/null || true

  local waited=0
  while [ "$waited" -lt 30 ]; do
    sleep 1
    waited=$((waited + 1))
    if ollama_running; then
      ok "ollama server up after ${waited}s  ${DIM}(log: .ollama/server.log)${OFF}"
      return 0
    fi
  done
  bad "ollama did not come up within 30s — see .ollama/server.log"
  return 1
}

# ----------------------------------------------------------------- flows ----
ensure_one() {
  local tool="$1"
  case "$tool" in
    python) check_python || { install_python && check_python; } ;;
    node)   check_node   || { install_node   && check_node; } ;;
    ollama) check_ollama || { install_ollama && check_ollama; } ;;
    *) printf 'Unknown tool: %s\n' "$tool"; return 1 ;;
  esac
}

cmd_check() {
  step "System dependencies  ${DIM}(platform: $PLATFORM, package manager: $PKG)${OFF}"
  local missing=0
  check_python || missing=$((missing + 1))
  check_node   || missing=$((missing + 1))
  check_ollama || missing=$((missing + 1))
  return "$missing"
}

cmd_ensure() {
  local tools=("$@")
  [ "${#tools[@]}" -eq 0 ] && tools=(python node ollama)

  step "System dependencies  ${DIM}(platform: $PLATFORM, package manager: $PKG)${OFF}"

  local missing=()
  for tool in "${tools[@]}"; do
    case "$tool" in
      python) check_python || missing+=(python) ;;
      node)   check_node   || missing+=(node) ;;
      ollama) check_ollama || missing+=(ollama) ;;
    esac
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    info "\nEverything needed is already installed."
    return 0
  fi

  if [ "${SKIP_SYSTEM_DEPS:-0}" = "1" ]; then
    printf '\n%sSKIP_SYSTEM_DEPS=1 — not installing: %s%s\n' "$YLW" "${missing[*]}" "$OFF"
    return 1
  fi

  if [ "$PKG" = "none" ] && [ "$PLATFORM" != "linux" ]; then
    printf '\n%sNo supported package manager found on this system.%s\n' "$YLW" "$OFF"
    printf 'Install these manually, then re-run `make setup`:\n'
    for tool in "${missing[@]}"; do
      case "$tool" in
        python) printf '  python  https://www.python.org/downloads/  (3.9-3.12)\n' ;;
        node)   printf '  node    https://nodejs.org/en/download     (18+)\n' ;;
        ollama) printf '  ollama  https://ollama.com/download\n' ;;
      esac
    done
    return 1
  fi

  printf '\nWill install: %s%s%s\n' "$BOLD" "${missing[*]}" "$OFF"
  confirm "Continue?" || { info "Aborted."; return 1; }

  local failed=0
  for tool in "${missing[@]}"; do
    ensure_one "$tool" || { bad "could not install $tool"; failed=$((failed + 1)); }
  done
  return "$failed"
}

case "${1:-check}" in
  check)        cmd_check ;;
  ensure)       shift; cmd_ensure "$@" ;;
  serve-ollama) serve_ollama ;;
  find-python)  find_python ;;
  platform)     printf '%s %s\n' "$PLATFORM" "$PKG" ;;
  *) printf 'Usage: %s {check|ensure [tool...]|serve-ollama|find-python|platform}\n' "$0"; exit 2 ;;
esac
