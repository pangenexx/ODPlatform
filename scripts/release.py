"""Release script for ODPlatform."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests."""
    print("Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "apps/platform/tests", "-v"],
        cwd=Path(__file__).parent.parent,
    )
    return result.returncode == 0


def build_package():
    """Build the platform package."""
    print("Building package...")
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=Path(__file__).parent.parent / "apps" / "platform",
    )
    return result.returncode == 0


def main():
    """Main release workflow."""
    if not run_tests():
        print("Tests failed. Aborting release.")
        sys.exit(1)

    if not build_package():
        print("Build failed. Aborting release.")
        sys.exit(1)

    print("Release build completed successfully.")


if __name__ == "__main__":
    main()
