VENV=.venv
PY=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

.PHONY: venv install run80 repeat results

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run80:
	$(PY) -m src.run_80_20

repeat:
	$(PY) -m src.run_80_20_repeat_v2metrics

results:
	cat outputs/RESULTS_SUMMARY.md
