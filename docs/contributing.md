# Contributing Guide

This guide details code quality standards, tools, and testing commands used in development.

## Quality Stack

Before committing code, ensure it adheres to linting and style specifications using:
- **Black**: Auto-format python modules to standard 88 character styles.
- **isort**: Organize python module import blocks.
- **Ruff**: Enforce Python style guide rules.
- **mypy**: Enforce static typing.
- **pytest**: Confirm code health using unit tests.

## Helper Commands

Our codebase maps testing and verification to the project `Makefile`:

1. **Auto-formatting code**:
   ```bash
   make format
   ```
2. **Linting and verification checks**:
   ```bash
   make lint
   ```
3. **Running the unit tests**:
   ```bash
   make test
   ```

## Creating Pull Requests
- Keep files modular and stick to single responsibility principles.
- Code must have complete PEP 484 type hints.
- Functions require google-style docstrings.
