.PHONY: env-init env-update test

env-init:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml

test:
	python -m unittest discover