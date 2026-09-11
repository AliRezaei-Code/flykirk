.PHONY: help install test demo debate brain fetch-annotations fetch-flywire clean

PY ?= python3
export PYTHONPATH := src

help:
	@echo "make install              install into the active environment (editable)"
	@echo "make test                 run the test suite (no model, no network)"
	@echo "make demo                 offline debate, scripted speaker, no dependencies beyond numpy"
	@echo "make brain                stimulate one fly brain with a sentence"
	@echo "make debate               debate using a local model (ollama/llama.cpp must be running)"
	@echo "make fetch-annotations    download the FlyWire 783 annotation table (~32 MB)"
	@echo "make fetch-flywire        download and build the real proofread connectome (~852 MB)"
	@echo "make clean                remove caches"

install:
	$(PY) -m pip install -e .

test:
	$(PY) tests/run.py

demo:
	$(PY) -m flykirk debate --offline --rounds 3 --neurons 4000 --seed 7 --show-brain

brain:
	$(PY) -m flykirk brain --text "The banana is OBVIOUSLY the superior fruit!" --neurons 4000

debate:
	$(PY) -m flykirk debate --random-topic --rounds 3 --neurons 4000 --show-brain

fetch-annotations:
	$(PY) -m flykirk fetch annotations

fetch-flywire:
	$(PY) -m flykirk fetch connectome --source zenodo --min-synapses 5

clean:
	rm -rf .pytest_cache **/__pycache__ src/flykirk.egg-info
	rm -rf $${FLYKIRK_DATA:-$$HOME/.cache/flykirk}
