# Contributing to LayerLens

Thank you for your interest in contributing to LayerLens! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ErenAta16/LayerLens.git
   cd LayerLens
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Add integration tests for new pipeline features

## Error Handling

- Use custom exceptions from `layerlens.exceptions`
- Add validation using `layerlens.utils.validation`
- Include logging for debugging
- Provide clear, actionable error messages

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update CHANGELOG.md for significant changes
- Add examples for new features

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

## Questions?

Open an issue for questions or discussions.

