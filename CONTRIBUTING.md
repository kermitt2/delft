# Contributing to DeLFT

Thank you for your interest in contributing to DeLFT! This guide will help you get started.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork:
   ```sh
   git clone https://github.com/<your-username>/delft
   cd delft
   ```
3. Create a branch for your changes:
   ```sh
   git checkout -b my-feature
   ```

## Development Setup

**Requirements:** Python 3.10 or 3.11

Set up a virtual environment and install in editable mode:

```sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install pip

# macOS
uv pip install -e ".[dev]"

# Linux with CUDA 12.1 (GPU)
uv pip install -e ".[dev,gpu]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. The configuration is in `pyproject.toml`.

Key settings:
- Line length: 120 characters
- Double quotes for strings
- Imports are sorted automatically

Before committing, run:

```sh
ruff check .
ruff format .
```

To auto-fix linting issues:

```sh
ruff check --fix .
```

## Running Tests

```sh
# Run all tests
pytest

# Skip slow tests
pytest -m "not slow"

# Run specific test directory
pytest tests/sequence_labelling/
```

## CI Checks

All pull requests must pass:
1. **Linting:** `ruff check .`
2. **Formatting:** `ruff format --check .`
3. **Tests:** `pytest`

CI runs on both Ubuntu and macOS with Python 3.10 and 3.11.

## Pull Request Process

1. Ensure there is an open issue to refer to
2. Branch from `master`
2. Make your changes and ensure all CI checks pass locally
3. Write a clear description of what your changes do and why
4. Submit your pull request against the `master` branch
5. Squash and merge


## License

By contributing to DeLFT, you agree to share your contribution under the [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).
