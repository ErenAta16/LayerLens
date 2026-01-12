# LayerLens Test Suite

This directory contains the test suite for LayerLens, organized into unit tests and integration tests.

## Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_profiling_aggregators.py
│   ├── test_profiling_analyzers.py
│   └── test_optimization_solver.py
├── integration/             # Integration tests for full pipelines
│   └── test_pipeline_e2e.py
└── fixtures/               # Test fixtures and sample data
```

## Running Tests

```bash
# Install test dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=layerlens --cov-report=html

# Run specific test file
pytest tests/unit/test_profiling_analyzers.py

# Run with verbose output
pytest -v
```

## Test Categories

### Unit Tests
Test individual components in isolation:
- Profiling aggregators
- Profiling analyzers
- Optimization solver
- Configuration classes
- Model specifications

### Integration Tests
Test full pipeline workflows:
- End-to-end pipeline execution
- Manifest generation
- Latency profiling
- Multi-model scenarios

## Writing New Tests

1. **Unit tests**: Add to `tests/unit/` with descriptive names like `test_<component>_<behavior>.py`
2. **Integration tests**: Add to `tests/integration/` for full workflow tests
3. **Fixtures**: Add reusable fixtures to `tests/conftest.py`
4. **Follow pytest conventions**: Use descriptive test function names starting with `test_`

## Coverage Goals

- Unit test coverage: >80%
- Integration test coverage: >60%
- Critical path coverage: 100%

