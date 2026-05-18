# D1: Architecture Overview

## Monorepo Structure

ODPlatform uses a monorepo structure with multiple apps:

```
ODPlatform/
├── apps/
│   ├── platform/      # Core engine (current focus)
│   ├── web-backend/   # Future V1.1
│   ├── web-frontend/  # Future V1.1
│   └── desktop/       # Future V2.0
├── docs/              # Documentation
├── data/              # Shared datasets
├── models/            # Shared model weights
└── runs/              # Training outputs
```

## Layer Architecture (platform app)

```
CLI Layer          ← Entry points (trans, validate, train, val, infer)
    ↓
Service Layer      ← Orchestration (training, evaluation, inference)
    ↓
Core Layer         ← Business logic (data conversion, validation)
    ↓
Config Layer       ← Configuration management
    ↓
Common Layer       ← Utilities (paths, logging, system, string, performance)
```

## Key Design Decisions

1. **src/ layout**: Follows PyPA recommendations
2. **Pydantic configs**: Type-safe configuration with validation
3. **Workspace marker**: `.odp-workspace` file for root detection
4. **CLI entry points**: Defined in pyproject.toml scripts
