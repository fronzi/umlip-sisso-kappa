# --------------------------------------------------------------------------
# Convenience wrappers around the seven pipeline scripts.
# Usage:  make dataset  |  make T0  |  make T1 UMLIP=mace-mp-0  |  ...
# --------------------------------------------------------------------------
PYTHON    ?= python
CFG       ?= configs/default.yaml
UMLIP     ?= mace-mp-0

.PHONY: all dataset T0 T1 T2 sisso validate figures clean

all: figures

dataset:
	$(PYTHON) scripts/01_assemble_dataset.py -c $(CFG)

T0:
	$(PYTHON) scripts/02_compute_T0.py -c $(CFG)

T1:
	$(PYTHON) scripts/03_compute_T1.py -c $(CFG) -u $(UMLIP)

T2:
	$(PYTHON) scripts/04_compute_T2.py -c $(CFG) -u $(UMLIP)

sisso:
	$(PYTHON) scripts/05_run_sisso.py -c $(CFG) -t T0 -u $(UMLIP)
	$(PYTHON) scripts/05_run_sisso.py -c $(CFG) -t T1 -u $(UMLIP)
	$(PYTHON) scripts/05_run_sisso.py -c $(CFG) -t T2 -u $(UMLIP)

validate:
	$(PYTHON) scripts/06_validate_softening.py -c $(CFG) -u $(UMLIP)

figures:
	$(PYTHON) scripts/07_make_figures.py -c $(CFG) -u $(UMLIP)

clean:
	rm -rf outputs/sr figs
