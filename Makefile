.PHONY: all install train experiments ablations figures tables report test clean

all: install train experiments ablations figures tables test
	@echo "All done. Results in results/, figures in figures/, tables in docs/tables.md"

install:
	pip install -r requirements.txt

train:
	python scripts/train_drl.py --config configs/default.yaml --output models/

experiments:
	python scripts/run_experiments.py --config configs/default.yaml --output results/

ablations:
	python scripts/run_ablations.py --config configs/default.yaml --output results/

figures:
	python scripts/generate_figures.py --results results/ --output figures/

tables:
	python scripts/generate_report_tables.py --results results/ --output docs/

test:
	python -m pytest tests/ -v

clean:
	rm -rf results/ figures/ models/ docs/tables.md
