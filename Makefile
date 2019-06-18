VENV = venv

PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python
RUN_GROBID_TAGGER = $(PYTHON) grobidTagger.py
USE_GPU = 0

ARGS =


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	virtualenv --system-site-packages -p python3 $(VENV)


dev-install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements.dev.txt
	@if [ "$(USE_GPU)" == "1" ]; then \
		$(PIP) install -r requirements.gpu.txt; \
	else \
		$(PIP) install -r requirements.cpu.txt; \
	fi


dev-venv: venv-create dev-install


flake8-syntax:
  # stop the build if there are Python syntax errors or undefined names
	$(PYTHON) -m flake8 delft *.py --count --select=E901,E999,F821,F822,F823 --show-source --statistics


flake8-warning-only:
  # exit-zero treats all errors as warnings.
	$(PYTHON) -m flake8 delft *.py --count --exit-zero --statistics


flake8:
	$(PYTHON) -m flake8 delft *.py


pylint:
	$(PYTHON) -m pylint delft *.py


tests: flake8 pylint


grobid-tagger-train-header:
	$(RUN_GROBID_TAGGER) header train $(ARGS)
