# Contributing to AtomicRAG

Thanks for your interest in contributing! This guide will help you get set up.

## Development Setup

```bash
# 1. Fork and clone the repo
git clone https://github.com/<your-username>/atomicrag.git
cd atomicrag

# 2. Install in development mode
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install
```

Once installed, pre-commit will automatically run on every `git commit` to check formatting, linting, and common issues.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality. The following hooks run automatically on each commit:

| Hook | What it does |
|------|-------------|
| `trailing-whitespace` | Removes trailing whitespace |
| `end-of-file-fixer` | Ensures files end with a newline |
| `check-yaml` / `check-toml` | Validates YAML and TOML syntax |
| `check-added-large-files` | Prevents files larger than 500KB |
| `check-merge-conflict` | Catches unresolved merge conflict markers |
| `debug-statements` | Flags leftover `print()` / `breakpoint()` / `pdb` |
| `ruff` | Lints Python code and auto-fixes issues |
| `ruff-format` | Formats Python code consistently |

### Running hooks manually

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only (same as what happens on commit)
pre-commit run
```

If a hook modifies a file, the commit will be aborted. Just `git add` the fixed files and commit again.

## Making Changes

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-change
   ```

2. **Make your changes** and ensure pre-commit hooks pass.

3. **Run tests** (if applicable):
   ```bash
   pytest
   ```

4. **Commit** your changes:
   ```bash
   git add .
   git commit -m "feat: describe your change"
   ```

5. **Push** and open a Pull Request:
   ```bash
   git push origin feature/my-change
   ```

## Commit Message Style

Use clear, descriptive commit messages:

- `feat: add new feature`
- `fix: resolve bug in retrieval`
- `docs: update README`
- `style: apply formatting`
- `refactor: restructure module`
- `test: add unit tests`

## Code Style

- Python 3.10+
- Formatting and linting handled by [ruff](https://docs.astral.sh/ruff/) via pre-commit
- Line length: 100 characters
- No need to manually format -- pre-commit handles it for you

## Questions?

Open an [issue](https://github.com/Rahuljangs/atomicrag/issues) to discuss before starting large changes.
