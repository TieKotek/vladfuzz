import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vladfuzz_runtime.drivefuzz_seed_export import export_manifest_to_drivefuzz
from vladfuzz_runtime.experiment_manifest import load_manifest_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VLAD-Fuzz manifest seeds to DriveFuzz seed-artifact format.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest produced by build_instruction_manifest.py")
    parser.add_argument("--manifest-id", help="Manifest row id to export")
    parser.add_argument("--output-dir", required=True, help="DriveFuzz seed output directory")
    parser.add_argument("--instruction-source", default="basic", choices=["manual", "basic", "route_prior", "no_prior"])
    parser.add_argument("--instruction", help="Manual instruction used when --instruction-source manual")
    parser.add_argument("--limit", type=int, help="Maximum number of seeds to export")
    args = parser.parse_args()

    rows = None
    if args.manifest_id:
        row = load_manifest_row(args.manifest, args.manifest_id)
        if args.instruction_source == "manual":
            if not args.instruction:
                raise ValueError("--instruction is required when --instruction-source manual is used")
            row = dict(row)
            row["manual_instruction"] = args.instruction
        rows = [row]

    mapping_path = export_manifest_to_drivefuzz(
        args.manifest,
        args.output_dir,
        instruction_source=args.instruction_source,
        limit=args.limit,
        rows=rows,
    )
    print(f"Exported DriveFuzz seeds and mapping to {mapping_path}")


if __name__ == "__main__":
    main()
