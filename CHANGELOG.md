# Changelog

All notable changes to LayerLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of LayerLens
- Cython-accelerated profiling (gradient energy, Fisher information)
- Multi-objective optimization solver
- Support for LoRA, Adapter, and Prefix tuning methods
- Realistic latency modeling with device and workload factors
- YOLO model integration
- LLM-YOLO pipeline for end-to-end analysis
- Comprehensive test suite (15 tests)
- Input validation and error handling
- Structured logging system
- Custom exception hierarchy
- Documentation and usage guides

### Performance
- Optimized pipeline: O(N²) → O(N) for large models
- Eliminated duplicate calculations
- Dictionary caching for repeated lookups
- NumPy array conversion optimizations
- 50-100x speedup for large models (100+ layers)

### Fixed
- Double normalization issue in analyzers
- Missing weight handling in aggregators
- Cython module optional fallback for Colab compatibility
- Package discovery in pyproject.toml
- .gitignore excluding layerlens/output package

### Changed
- Package renamed from `hyperlora` to `layerlens`
- Config module split into submodules (profiling, optimization, latency)
- Meta package renamed to models
- Improved error messages with actionable guidance

### Security
- Input validation for all public APIs
- Type checking and validation utilities

## [Unreleased]

### Planned
- Additional model architectures support
- Enhanced benchmarking suite
- CI/CD pipeline
- Performance profiling tools
- Extended documentation

