# ODPlatform Project Rules for AI

## Commands
- `pip install -e ./apps/platform` — install core package
- `pip install -e ./apps/platform --force-reinstall --no-deps` — reinstall after pyproject.toml changes
- `odp-webui` — start Gradio WebUI at localhost:7860
- `odp-webui --port 7861` — start on different port
- `odp-backend` — start FastAPI backend only
- `odp-train --model yolo11n.pt --data configs/datasets/<dataset>.yaml` — train model
- `odp-infer detect --model best.pt --input <path>` — single image/folder/video inference
- `odp-infer live --model best.pt --source 0` — live camera inference
- `odp-init` — create runtime directories
- `odp-reset --yes --force` — clean runtime data

## Coding Conventions
- All imports at top; standard lib → third-party → local
- Type hints required for all function signatures
- String quotes: use single quotes for JS/Python strings in HTML templates to avoid escaping issues
- matplotlib: always set `matplotlib.use("Agg")` before import pyplot
- Gradio: use `elem_classes` for CSS hooks, no inline styles

## Key File Locations
- WebUI entry: `apps/platform/src/odp_platform/webui/app.py`
- User tabs: `apps/platform/src/odp_platform/webui/user_tabs.py`
- Training tab: `apps/platform/src/odp_platform/webui/training_tab.py`
- CLI entry: `apps/platform/src/odp_platform/cli/`
- Inference engine: `apps/platform/src/odp_platform/inference/engine.py`
- Visualizer: `apps/platform/src/odp_platform/inference/visualizer.py`

## Branch Strategy
- Only `main` branch is active. All work happens on main.
- No feature branches unless explicitly requested.
