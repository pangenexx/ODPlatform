# ADR-003: Naming Conventions

## Status
Accepted

## Context
We needed consistent naming conventions across the project.

## Decision

### Python Packages
- Use snake_case: `odp_platform`, `data_validation`
- Mirror directory structure in package hierarchy

### Modules
- Use snake_case with `_utils` suffix for utility modules
- Use `_config` suffix for configuration modules

### Classes
- Use PascalCase: `TrainConfig`, `PascalVOCConverter`
- Suffix with purpose: `Service`, `Converter`, `Validator`

### Functions
- Use snake_case: `convert_voc_to_yolo`, `setup_logger`
- Verb-first naming for actions

### CLI Commands
- Prefix with `odp-`: `odp-train`, `odp-validate`
- Use hyphens, not underscores

## Consequences
- Consistent naming across all modules
- Easy to guess module/class/function names
- Clear separation of concerns through naming
