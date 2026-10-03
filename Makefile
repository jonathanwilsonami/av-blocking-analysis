# Reproduce the full analysis and website.  Requires uv and Quarto (see README).
NB := notebooks/av_blocking_analysis.ipynb

.PHONY: all setup test analysis notebook-html site preview zeroshot clean

all: test analysis site

setup:            ## create .venv and install the package + dev tools
	uv sync

test:             ## unit tests
	uv run pytest -q

analysis:         ## execute the notebook; regenerates project-site/assets and _variables.yml
	uv run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 $(NB)

notebook-html:    ## static HTML copy of the executed notebook for the website
	uv run jupyter nbconvert --to html $(NB) --output-dir project-site --output notebook

site: notebook-html  ## render paper, slides, and site into project-site/docs
	cd project-site && uv run quarto render

preview:          ## live-preview the website
	cd project-site && uv run quarto preview

zeroshot:         ## (optional) re-run the zero-shot transformer and refresh its cache
	uv sync --extra nlp
	uv run python -c "from av_blocking import data, nlp; nlp.run_zeroshot(data.build_incidents())"

clean:
	rm -rf project-site/docs project-site/.quarto project-site/notebook.html
