PYTHON ?= python3
CONFIG ?= configs/full.json
OUT ?= results/$(notdir $(basename $(CONFIG)))
PERF ?= results/performance
WORKERS ?= 4
REPEATS ?= 3
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib

.PHONY: all simulate analyze figures clean test regression profile complexity benchmark parallel stability-check baseline

all: figures

simulate:
	$(PYTHON) -m src.simulation --config "$(CONFIG)" --output "$(OUT)"

analyze: simulate
	$(PYTHON) -m src.analyze --config "$(CONFIG)" --output "$(OUT)"

figures: analyze
	$(PYTHON) -m src.figures --config "$(CONFIG)" --output "$(OUT)"

test:
	$(PYTHON) -m pytest -q

regression:
	$(PYTHON) -m pytest -q tests/test_regression.py

baseline:
	$(PYTHON) -m baseline.src.simulation --config "$(CONFIG)" --output results/baseline-$(notdir $(basename $(CONFIG))) --force

profile:
	$(PYTHON) -m src.performance profile --config configs/pilot.json --out "$(PERF)"

complexity:
	$(PYTHON) -m src.performance complexity --out "$(PERF)" --repeats "$(REPEATS)"

benchmark:
	$(PYTHON) -m src.performance benchmark --config "$(CONFIG)" --out "$(PERF)" --repeats "$(REPEATS)"

parallel:
	$(PYTHON) -m src.performance parallel --config "$(CONFIG)" --out "$(PERF)" --workers 2 "$(WORKERS)" --repeats "$(REPEATS)"

stability-check:
	$(PYTHON) -m src.stability_check --out "$(PERF)"

clean:
	$(PYTHON) -m src.clean --config "$(CONFIG)" --output "$(OUT)"
