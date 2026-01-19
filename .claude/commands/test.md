---
name: test
description: Run tests with options for coverage, watch mode, and parallel fixing
---

# Run Tests

## Quick Run (default)
```bash
pytest tests/ -q
```

## With Coverage
```bash
pytest tests/ --cov=agents --cov=config --cov=data --cov=database --cov=execution --cov=risk --cov=monitoring --cov-report=term-missing
```

## Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_risk_manager.py -v
pytest tests/unit/test_claude_agent.py -v
pytest tests/unit/test_indicators.py -v
```

## Filter by Test Name
```bash
pytest tests/ -k "test_buy" -v
pytest tests/ -k "risk" -v
```

## Parallel Execution (faster)
```bash
pytest tests/ -n auto
```

## On Failure: Spawn Parallel Agents

If tests fail, analyze failures and spawn parallel agents:

1. **Parse failures** - Group by module/domain
2. **Spawn agents** - One per domain in a SINGLE response:
   - "risk-fixer" for risk_manager failures
   - "agent-fixer" for claude_agent/grok_agent failures
   - "indicator-fixer" for indicators failures
   - "execution-fixer" for order_executor failures
   - "integration-fixer" for integration test failures

Each agent receives:
- List of failing tests
- Error messages
- Instructions to fix and re-run

## Verify All Pass
After fixes, run full suite:
```bash
pytest tests/ -q --tb=short
```

All 416 tests must pass before completing.
