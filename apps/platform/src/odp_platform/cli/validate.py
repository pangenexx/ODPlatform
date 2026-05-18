"""Data validation CLI."""

import argparse
from pathlib import Path

from odp_platform.data_validation.service import (
    validate_dataset,
    analyze_dataset,
    clean_dataset,
    visualize_dataset,
    generate_report,
)


def main():
    """Main entry point for data validation."""
    parser = argparse.ArgumentParser(description="Validate and analyze YOLO datasets")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--clean", action="store_true", help="Clean invalid samples")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")

    args = parser.parse_args()

    print("Validating dataset...")
    results = validate_dataset(args.dataset, args.classes)

    if results["valid"]:
        print("Dataset is valid!")
    else:
        print(f"Found {len(results['errors'])} errors:")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["warnings"]:
        print(f"Found {len(results['warnings'])} warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        print("\nAnalyzing dataset...")
        analysis = analyze_dataset(args.dataset)
        generate_report(analysis, args.output / "report.md")

        if args.clean:
            print("\nCleaning dataset...")
            clean_dir = args.output / "cleaned"
            clean_dataset(args.dataset, clean_dir)
            print(f"Cleaned dataset saved to: {clean_dir}")

        if args.visualize:
            print("\nGenerating visualizations...")
            visualize_dataset(args.dataset, args.output / "plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
