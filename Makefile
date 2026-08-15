# Anima — Multi-Agent LSD Detection
#
# One entrypoint for the whole pipeline. Everything runs on this machine:
# no hosted model APIs, no API keys.
#
#   make            list the targets
#   make setup      install EVERYTHING and get the pipeline runnable
#   make dev        run backend + frontend
#   make test       run the test suite
#
# Portable across macOS, Linux (incl. WSL) and Windows via Git Bash / MSYS2.

SHELL := /bin/bash
.DEFAULT_GOAL := help
.SHELLFLAGS := -eu -o pipefail -c

# ----------------------------------------------------------- platform ------
# `OS` is set to Windows_NT on native Windows regardless of shell; `uname`
# covers everything else, including Git Bash (MINGW*) and WSL (Linux).
UNAME_S := $(shell uname -s 2>/dev/null || echo unknown)

ifeq ($(OS),Windows_NT)
  PLATFORM := windows
else ifeq ($(UNAME_S),Darwin)
  PLATFORM := macos
else ifeq ($(UNAME_S),Linux)
  PLATFORM := linux
else ifneq (,$(findstring MINGW,$(UNAME_S)))
  PLATFORM := windows
else ifneq (,$(findstring MSYS,$(UNAME_S)))
  PLATFORM := windows
else ifneq (,$(findstring CYGWIN,$(UNAME_S)))
  PLATFORM := windows
else
  PLATFORM := unknown
endif

ifeq ($(PLATFORM),windows)
  EXE := .exe
else
  EXE :=
endif

# ------------------------------------------------------------- layout ------
BACKEND  := backend
FRONTEND := frontend
VENV     := $(BACKEND)/.venv
SCRIPTS  := scripts/system-deps.sh

# A venv puts its executables in Scripts/ on Windows, bin/ everywhere else.
# Must come after VENV is defined.
ifeq ($(PLATFORM),windows)
  VENV_BIN := $(VENV)/Scripts
else
  VENV_BIN := $(VENV)/bin
endif

PY      := $(VENV_BIN)/python$(EXE)
PIP     := $(VENV_BIN)/pip$(EXE)
UVICORN := $(VENV_BIN)/uvicorn$(EXE)

BACKEND_PORT  ?= 8000
FRONTEND_PORT ?= 5173

# Model names come from backend/.env when present, so `make models` always
# pulls exactly what the service is configured to use. `-` = tolerate absence.
-include $(BACKEND)/.env

# Fallbacks only — backend/.env is the source of truth and is included above.
LSD_VISION_MODEL ?= gemma3:4b
LSD_TEXT_MODEL   ?= gemma3:4b
LSD_LLM_BASE_URL ?= http://127.0.0.1:11434

# The two roles usually point at one multimodal model; pull it once.
MODELS := $(sort $(LSD_VISION_MODEL) $(LSD_TEXT_MODEL))

# TensorFlow publishes wheels for these only; setup refuses anything else.
PY_SUPPORTED := 3.9 3.10 3.11 3.12
# Discovered via the bootstrap script so make and the script always agree.
PYTHON ?= $(shell bash $(SCRIPTS) find-python 2>/dev/null)

BOLD := \033[1m
DIM  := \033[2m
GRN  := \033[32m
YLW  := \033[33m
RED  := \033[31m
OFF  := \033[0m

.PHONY: help
help: ## Show this help
	@printf "$(BOLD)Anima — Multi-Agent LSD Detection$(OFF)\n"
	@printf "$(DIM)All inference runs locally. No API keys required.$(OFF)\n\n"
	@printf "$(BOLD)First time:$(OFF)  make setup    $(DIM)(installs everything, ~10 min)$(OFF)\n"
	@printf "$(BOLD)Then:$(OFF)        make dev\n\n"
	@# -h: MAKEFILE_LIST also contains backend/.env (pulled in by `-include`
	@# above), and with more than one file grep prefixes every line with the
	@# filename — which awk then reports as the target name.
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GRN)%-18s$(OFF) %s\n", $$1, $$2}'
	@printf "\n$(DIM)platform: $(PLATFORM) · models: $(LSD_VISION_MODEL) (vision), $(LSD_TEXT_MODEL) (text)$(OFF)\n"

# ============================================================== SETUP =======

.PHONY: setup
setup: ## Install everything needed to run the pipeline (system deps, venv, node, models)
	@printf "$(BOLD)Anima setup$(OFF)  $(DIM)platform: $(PLATFORM)$(OFF)\n"
	@$(MAKE) --no-print-directory system-deps || { \
		printf "\n$(RED)Setup stopped:$(OFF) the system dependencies above are missing.\n"; \
		printf "Install them (see the links printed above), then re-run $(BOLD)make setup$(OFF).\n"; \
		printf "$(DIM)Already have them elsewhere? Try: make setup PYTHON=/path/to/python3.12$(OFF)\n"; \
		exit 1; }
	@$(MAKE) --no-print-directory setup-backend
	@$(MAKE) --no-print-directory setup-frontend
	@$(MAKE) --no-print-directory models
	@$(MAKE) --no-print-directory index || \
		printf "  $(YLW)skipped$(OFF)  precedent index — run $(BOLD)make index$(OFF) later\n"
	@printf "\n$(GRN)$(BOLD)Setup complete.$(OFF)\n"
	@$(MAKE) --no-print-directory doctor
	@printf "\nStart it with: $(BOLD)make dev$(OFF)\n"

.PHONY: setup-offline
setup-offline: ## Setup without touching system packages or downloading models
	@SKIP_SYSTEM_DEPS=1 $(MAKE) --no-print-directory system-deps || true
	@$(MAKE) --no-print-directory setup-backend
	@$(MAKE) --no-print-directory setup-frontend
	@printf "\n$(GRN)Code dependencies installed.$(OFF)\n"
	@printf "$(DIM)Run `make models` when you have network to pull the weights.$(OFF)\n"

.PHONY: system-deps
system-deps: ## Install Python, Node and Ollama using the host's package manager
	@bash $(SCRIPTS) ensure

.PHONY: setup-backend
setup-backend: $(VENV) ## Create the Python venv and install backend dependencies
	@printf "\n$(BOLD)Installing backend dependencies$(OFF) $(DIM)(TensorFlow is large — a few minutes)$(OFF)\n"
	@$(PY) -m pip install --quiet --upgrade pip
	@$(PY) -m pip install --quiet -r $(BACKEND)/requirements.txt
	@test -f $(BACKEND)/.env || { cp $(BACKEND)/.env.example $(BACKEND)/.env; \
		printf "  $(DIM)created backend/.env from the example$(OFF)\n"; }
	@printf "  $(GRN)ok$(OFF)       backend ($$($(PY) --version 2>&1))\n"

$(VENV):
	@test -n "$(PYTHON)" || { \
		printf "$(RED)No supported Python found.$(OFF) Need one of: $(PY_SUPPORTED)\n"; \
		printf "Run $(BOLD)make system-deps$(OFF), or pass one: $(BOLD)make setup PYTHON=/path/to/python3.12$(OFF)\n"; \
		exit 1; }
	@ver=$$($(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
	if ! printf '%s\n' $(PY_SUPPORTED) | grep -qx "$$ver"; then \
		printf "$(RED)Python $$ver is not supported.$(OFF) TensorFlow needs $(PY_SUPPORTED).\n"; \
		printf "Run $(BOLD)make system-deps$(OFF), or: $(BOLD)make setup PYTHON=/path/to/python3.12$(OFF)\n"; \
		exit 1; \
	fi; \
	printf "\n$(BOLD)Creating virtualenv$(OFF) $(DIM)($(PYTHON), Python $$ver)$(OFF)\n"; \
	$(PYTHON) -m venv $(VENV)

.PHONY: setup-frontend
setup-frontend: ## Install frontend dependencies
	@printf "\n$(BOLD)Installing frontend dependencies$(OFF)\n"
	@command -v npm >/dev/null 2>&1 || { \
		printf "  $(RED)failed$(OFF)   npm not found — run $(BOLD)make system-deps$(OFF)\n"; exit 1; }
	@cd $(FRONTEND) && npm install --silent --no-fund --no-audit
	@printf "  $(GRN)ok$(OFF)       frontend (node $$(node --version))\n"

.PHONY: models
models: ## Download the local model weights (several GB, one time)
	@printf "\n$(BOLD)Local model weights$(OFF)\n"
	@command -v ollama >/dev/null 2>&1 || { \
		printf "  $(RED)failed$(OFF)   ollama not installed — run $(BOLD)make system-deps$(OFF)\n"; exit 1; }
	@bash $(SCRIPTS) serve-ollama
	@for model in $(MODELS); do \
		if ollama list 2>/dev/null | awk 'NR>1 {print $$1}' | grep -qx "$$model"; then \
			printf "  $(GRN)ok$(OFF)       $$model $(DIM)(already downloaded)$(OFF)\n"; \
		else \
			printf "  $(DIM)pulling$(OFF)  $$model\n"; \
			ollama pull "$$model"; \
		fi; \
	done

.PHONY: index
index: check-venv ## Build the precedent index for the visual evidence panel
	@printf "\n$(BOLD)Precedent index$(OFF) $(DIM)(embeds the dataset — about 20 s)$(OFF)\n"
	@cd $(BACKEND) && ../$(PY) scripts/build_visual_index.py

.PHONY: serve-ollama
serve-ollama: ## Start the local model server in the background
	@bash $(SCRIPTS) serve-ollama

# ============================================================ RUNNING =======

.PHONY: dev
dev: check-venv ## Run backend + frontend together
	@printf "$(BOLD)Backend$(OFF)  http://127.0.0.1:$(BACKEND_PORT)  $(DIM)(API docs at /docs)$(OFF)\n"
	@printf "$(BOLD)Frontend$(OFF) http://localhost:$(FRONTEND_PORT)\n"
	@printf "$(DIM)Ctrl-C stops both.$(OFF)\n\n"
	@trap 'kill 0' EXIT INT TERM; \
		( cd $(BACKEND) && ../$(UVICORN) app.main:app --reload --port $(BACKEND_PORT) ) & \
		( cd $(FRONTEND) && npm run dev -- --port $(FRONTEND_PORT) ) & \
		wait

.PHONY: backend
backend: check-venv ## Run the API only (port 8000)
	@cd $(BACKEND) && ../$(UVICORN) app.main:app --reload --port $(BACKEND_PORT)

.PHONY: frontend
frontend: ## Run the Vue dev server only (port 5173)
	@cd $(FRONTEND) && npm run dev -- --port $(FRONTEND_PORT)

.PHONY: stop
stop: ## Stop stray dev servers
	@-pkill -f "uvicorn app.main:app" 2>/dev/null && printf "stopped backend\n" || true
	@-pkill -f "vite" 2>/dev/null && printf "stopped frontend\n" || true

# ============================================================= CHECKS =======

.PHONY: test
test: check-venv ## Run the backend test suite (needs no models)
	@cd $(BACKEND) && ../$(PY) -m pytest tests -q

.PHONY: evaluate
evaluate: check-venv ## Held-out evaluation of the trained CNN
	@cd $(BACKEND) && ../$(PY) scripts/evaluate_model.py

.PHONY: health
health: ## Report which components are live
	@curl -sf -m 5 http://127.0.0.1:$(BACKEND_PORT)/api/health 2>/dev/null \
		| $(PY) -m json.tool 2>/dev/null \
		|| printf "$(YLW)Backend not responding on port $(BACKEND_PORT).$(OFF) Start it with: make backend\n"

.PHONY: doctor
doctor: ## Diagnose the setup
	@bash $(SCRIPTS) check || true
	@printf "\n$(BOLD)Project$(OFF)\n"
	@if [ -d "$(VENV)" ]; then printf "  $(GRN)ok$(OFF)       venv ($$($(PY) --version 2>&1))\n"; \
		else printf "  $(YLW)missing$(OFF)  venv — run $(BOLD)make setup$(OFF)\n"; fi
	@if [ -d "$(FRONTEND)/node_modules" ]; then printf "  $(GRN)ok$(OFF)       frontend dependencies\n"; \
		else printf "  $(YLW)missing$(OFF)  frontend dependencies — run $(BOLD)make setup$(OFF)\n"; fi
	@if [ -f "vision/cattle_lsd_detector.keras" ]; then \
		printf "  $(GRN)ok$(OFF)       trained CNN\n"; \
		else printf "  $(RED)missing$(OFF)  vision/cattle_lsd_detector.keras\n"; fi
	@if [ -f "vision/visual_index.npz" ]; then \
		printf "  $(GRN)ok$(OFF)       precedent index\n"; \
		else printf "  $(YLW)missing$(OFF)  precedent index — run $(BOLD)make index$(OFF) $(DIM)(optional)$(OFF)\n"; fi
	@printf "\n$(BOLD)Local model server$(OFF) $(DIM)($(LSD_LLM_BASE_URL))$(OFF)\n"
	@if curl -sf -m 3 $(LSD_LLM_BASE_URL)/api/tags >/dev/null 2>&1; then \
		printf "  $(GRN)ok$(OFF)       running\n"; \
		for model in $(MODELS); do \
			if ollama list 2>/dev/null | awk 'NR>1 {print $$1}' | grep -qx "$$model"; then \
				printf "  $(GRN)ok$(OFF)       $$model\n"; \
			else printf "  $(YLW)missing$(OFF)  $$model — run $(BOLD)make models$(OFF)\n"; fi; \
		done; \
	else \
		printf "  $(YLW)stopped$(OFF)  start with $(BOLD)make serve-ollama$(OFF)\n"; \
	fi

.PHONY: check-venv
check-venv:
	@test -x "$(PY)" || { \
		printf "$(RED)No virtualenv.$(OFF) Run $(BOLD)make setup$(OFF) first.\n"; exit 1; }

# ======================================================= HOUSEKEEPING =======

.PHONY: clean
clean: ## Remove caches and build artefacts
	@find . -type d -name __pycache__ -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf $(BACKEND)/.pytest_cache $(FRONTEND)/dist
	@printf "$(GRN)cleaned$(OFF)\n"

.PHONY: clean-all
clean-all: clean ## Also remove the venv and node_modules
	@rm -rf $(VENV) $(FRONTEND)/node_modules
	@printf "$(GRN)removed venv and node_modules$(OFF)\n"
