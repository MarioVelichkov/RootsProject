from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roots_project.config import PipelineSettings
from roots_project.cv import RootAnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run root segmentation and root-tip extraction on one plate image.")
    parser.add_argument("--image", required=True, type=Path, help="Raw plant plate image.")
    parser.add_argument("--model", default=PipelineSettings.model_path, type=Path, help="Trained U-Net model path.")
    parser.add_argument("--output", default=PipelineSettings.output_dir, type=Path, help="Output directory.")
    parser.add_argument("--patch-size", default=PipelineSettings.patch_size, type=int, help="Patch size used by the model.")
    parser.add_argument("--max-roots", default=PipelineSettings.max_roots, type=int, help="Maximum root tips to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = PipelineSettings(model_path=args.model, output_dir=args.output, patch_size=args.patch_size, max_roots=args.max_roots)
    result = RootAnalysisPipeline(settings).run(args.image, args.output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
