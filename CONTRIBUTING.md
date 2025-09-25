# Contributing to AIVault

We welcome contributions to AIVault! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/vahidnourbakhsh/AIVault.git
cd AIVault
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate aivault
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- We use Black for code formatting
- We use isort for import sorting  
- We use flake8 for linting
- We use mypy for type checking

Run all checks with:
```bash
black aivault tests examples
isort aivault tests examples
flake8 aivault tests examples
mypy aivault
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

For coverage reports:
```bash
pytest tests/ --cov=aivault --cov-report=html
```

## Adding New Features

1. Create a new branch from main
2. Add your feature with appropriate tests
3. Update documentation if needed
4. Ensure all tests pass
5. Submit a pull request

## Documentation

- Use clear docstrings for all functions and classes
- Follow NumPy docstring style
- Include examples in docstrings when helpful
- Update README.md if adding major features

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Write a clear pull request description

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Full error traceback
- Minimal example to reproduce the issue

## Code of Conduct

Be respectful and inclusive in all interactions. We want AIVault to be welcoming to contributors of all backgrounds and experience levels.
