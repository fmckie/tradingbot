---
name: update-app
description: Update dependencies, fix deprecations and warnings
---

# Dependency Update & Deprecation Fix

## Step 1: Check for Updates

```bash
pip list --outdated
```

Review which packages have newer versions available.

## Step 2: Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

For specific packages with major updates, update manually:
```bash
pip install --upgrade <package-name>
```

## Step 3: Check for Deprecations & Warnings

Run installation and check output:
```bash
pip install -r requirements.txt 2>&1
```

Read ALL output carefully. Look for:
- DeprecationWarning messages
- FutureWarning messages
- Security vulnerabilities
- Dependency conflicts
- Import warnings

## Step 4: Fix Issues

For each warning/deprecation:
1. Research the recommended replacement or fix
2. Update code/dependencies accordingly
3. Re-run installation
4. Verify no warnings remain

Common fixes:
- Replace deprecated imports with new ones
- Update function calls to new API signatures
- Pin versions to avoid breaking changes

## Step 5: Run Quality Checks

```bash
python -m py_compile main.py
python test_setup.py
python test_learning_system.py
python test_simulation.py
```

Fix all errors before completing.

## Step 6: Verify Clean Install

Ensure a fresh install works:
```bash
pip freeze > requirements-current.txt
pip install -r requirements.txt --dry-run
```

Review requirements-current.txt for any version conflicts.

## Step 7: Update requirements.txt

If versions changed, update requirements.txt with tested versions:
```bash
pip freeze | grep -E "^(anthropic|alpaca-py|pandas|numpy|ta|httpx|asyncpg|modal|schedule|pytz|rich|python-dotenv|sqlite-utils)==" > requirements-updated.txt
```

Review and merge into requirements.txt as needed.
