# ML Resources Hub — common tasks
#
# Usage:
#   make            # build all topic pages (alias for `make build`)
#   make build      # build all topic pages from topics/_src/*.html
#   make watch      # rebuild on source / template change (Ctrl+C to stop)
#   make serve      # run a local server at http://localhost:8000
#   make setup      # one-time: point git at our pre-commit hook

.PHONY: build watch serve setup

build:
	@python3 build.py

watch:
	@python3 build.py --watch

serve:
	@echo "→ http://localhost:8000"
	@python3 -m http.server 8000

setup:
	@git config core.hooksPath hooks
	@chmod +x hooks/pre-commit
	@echo "✓ git will now run hooks/pre-commit before every commit"
