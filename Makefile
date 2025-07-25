# --- --- ---

.PHONY: check
check: 
#   uvx ty check
	uvx ruff check --fix
	uvx ruff format

# --- --- ---

.PHONY: test
test:
	uv run pytest


.PHONY: docs
docs:
	uv run sphinx-build -M html docs/ .docs
	open -a "Google Chrome" .docs/html/index.html
	read  -n 1 -p "Remove Docs after Input:" mainmenuinput
	rm -rf .docs