---
name: fix
description: Run typechecking and linting, then spawn parallel agents to fix all issues
---

# Project Code Quality Check

This command runs all linting and typechecking tools for this project, collects errors, groups them by domain, and spawns parallel agents to fix them.

## Step 1: Run Linting and Typechecking

Run these commands and collect ALL output:

```bash
# Syntax check all Python files
python -m py_compile main.py modal_app.py test_setup.py test_learning_system.py test_simulation.py
find agents config data database execution monitoring reports risk tools -name "*.py" -exec python -m py_compile {} \;

# Type checking (if mypy installed)
mypy . --ignore-missing-imports 2>&1 || echo "mypy not installed, skipping"

# Linting (if ruff installed)
ruff check . 2>&1 || echo "ruff not installed, skipping"

# Alternative lint (if pylint installed)
pylint **/*.py --output-format=text 2>&1 || echo "pylint not installed, skipping"
```

## Step 2: Collect and Parse Errors

Parse the output from the linting and typechecking commands. Group errors by domain:
- **Syntax errors**: Python compilation failures
- **Type errors**: Issues from mypy
- **Lint errors**: Issues from ruff or pylint
- **Import errors**: Missing or circular imports

Create a list of all files with issues and the specific problems in each file.

## Step 3: Spawn Parallel Agents

For each domain that has issues, spawn an agent in parallel using the Task tool:

**IMPORTANT**: Use a SINGLE response with MULTIPLE Task tool calls to run agents in parallel.

Example agent assignments:
- Spawn a "syntax-fixer" agent for syntax/compilation errors
- Spawn a "type-fixer" agent for type annotation issues
- Spawn a "lint-fixer" agent for code style and lint errors
- Spawn an "import-fixer" agent for import issues

Each agent should:
1. Receive the list of files and specific errors in their domain
2. Fix all errors in their domain
3. Run the relevant check command to verify fixes
4. Report completion

## Step 4: Verify All Fixes

After all agents complete, run the full check again:

```bash
python -m py_compile main.py
python test_setup.py
```

Ensure all issues are resolved before completing.

## Step 5: Run Tests

Verify fixes didn't break functionality:

```bash
python test_setup.py
python test_learning_system.py
python test_simulation.py
```

Fix any test failures before completing.
