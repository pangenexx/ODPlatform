# ODPlatform - Object Detection Platform

A monorepo-based object detection platform for training, evaluating, and deploying YOLO models.

## Project Structure

```
ODPlatform/
├── apps/                    # Business applications
│   ├── platform/            # Core engine (D1-D10)
│   ├── web-backend/         # Web backend (V1.1)
│   ├── web-frontend/        # Web frontend (V1.1)
│   └── desktop/             # Desktop app (V2.0)
├── docs/                    # Project documentation
│   ├── architecture/        # Architecture Decision Records
│   ├── srs/                 # Software Requirements Specification
│   ├── teaching/            # Teaching materials
│   └── api/                 # API documentation
├── data/                    # Shared datasets (.gitignore)
├── models/                  # Shared model weights (Git LFS)
├── runs/                    # Training outputs (.gitignore)
└── scripts/                 # Workspace-level maintenance scripts
```

## Quick Start

```bash
# Create conda environment
conda create -n odp-gpu python=3.12 -y
conda activate odp-gpu

# Install platform package
cd apps/platform
pip install -e .
```

## License

See [LICENSE](LICENSE) for details.
