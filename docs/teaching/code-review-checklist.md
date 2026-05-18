# Code Review Checklist

## General
- [ ] Code follows naming conventions (ADR-003)
- [ ] No hardcoded paths (use paths.py)
- [ ] Type hints present
- [ ] Docstrings for public APIs

## Configuration
- [ ] Uses Pydantic models
- [ ] Config can be loaded from YAML
- [ ] Default values are sensible

## Data Pipeline
- [ ] Handles missing files gracefully
- [ ] Validates input format
- [ ] Progress indicators for long operations

## Testing
- [ ] Unit tests for pure functions
- [ ] Integration tests for workflows
- [ ] Tests use fixtures from conftest.py

## Error Handling
- [ ] Meaningful error messages
- [ ] Exceptions caught at appropriate level
- [ ] Logging for debugging
