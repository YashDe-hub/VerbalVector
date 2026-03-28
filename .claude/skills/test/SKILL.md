---
name: test
description: Run pytest with coverage and highlight untested functions
---
Run `pytest tests/ -v --tb=short --cov --cov-report=term-missing`.
If tests fail, diagnose and fix. If tests pass, report coverage and list 0% functions.
