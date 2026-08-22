# Makefile — run sdesim scripts via `python -m scripts.<name>`
#
# Usage:
#   make run_optimal_protocol
#   make run_bath_model_eq run_bath_model_non_eq run_optimal_protocol   # runs in the order given
#   make -k run_bath_model_eq run_optimal_protocol                      # continue even if one fails
#   make list                                                           # show all discoverable scripts
#   make pipeline                                                       # run the full dependency-ordered chain

PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || command -v py 2>/dev/null)
SCRIPTS_DIR := scripts

# Auto-discover every scripts/run_*.py and turn it into a phony target name,
# e.g. scripts/run_optimal_protocol.py -> run_optimal_protocol
SCRIPTS := $(patsubst $(SCRIPTS_DIR)/%.py,%,$(wildcard $(SCRIPTS_DIR)/run_*.py))

ifeq ($(strip $(SCRIPTS)),)
$(error No scripts found under $(SCRIPTS_DIR)/run_*.py — run make from the repo root)
endif

.PHONY: all list help pipeline $(SCRIPTS)

all: $(SCRIPTS)

list help:
	@echo "Available scripts (run with 'make <name>', or several at once):"
	@for s in $(SCRIPTS); do echo "  $$s"; done

# One rule matches every discovered script name; $@ is the target name that
# was invoked, e.g. `make run_optimal_protocol` -> $@ = run_optimal_protocol
$(SCRIPTS):
	$(PYTHON) -m $(SCRIPTS_DIR).$@

# Fixed, dependency-correct order for the whole pipeline (edit names/order
# to match your actual files). Prerequisites of a single target are built
# left-to-right in serial (non -j) make, so this is deterministic.
pipeline: run_bath_model_eq run_bath_model_non_eq run_optimal_protocol run_optimal_protocol_3_measurement run_initial_condition_convergence 