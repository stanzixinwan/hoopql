.PHONY: dev test eval

dev:
	docker compose up --build

test:
	uv run pytest

eval:
	@echo "Eval runner lands in Phase 4 (T4.3)."
