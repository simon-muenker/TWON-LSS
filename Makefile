# --- --- ---

.PHONY: check
check: 
	uvx ty check
	uvx ruff check --fix
	uvx ruff format

# --- --- ---

.PHONY: test
test:
	uv run pytest