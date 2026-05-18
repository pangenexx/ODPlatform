# ADR-002: Paths Strategy

## Status
Accepted

## Context
We needed a reliable way to locate the workspace root from anywhere in the project.

## Decision
We use a `.odp-workspace` marker file that `paths.py` searches for by walking up the directory tree.

## Implementation
```python
def find_workspace_root(start: Path | None = None) -> Path:
    current = start or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".odp-workspace").exists():
            return parent
    raise RuntimeError("Cannot find workspace root.")
```

## Consequences
- Works regardless of current working directory
- No hardcoded paths
- Easy to understand and debug
