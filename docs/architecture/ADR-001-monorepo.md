# ADR-001: Monorepo Structure

## Status
Accepted

## Context
We needed to decide between monorepo and polyrepo for the ODPlatform project.

## Decision
We chose a monorepo structure with the following benefits:
- Single source of truth
- Easier cross-module refactoring
- Simplified dependency management
- Better for teaching purposes

## Consequences
- All apps share the same repository
- Workspace-level tooling (pyproject.toml) coordinates builds
- Large files (data, models) are gitignored or use LFS
