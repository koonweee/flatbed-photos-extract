"""Command-line interface for flatbed photo extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from extractor.core import (
    DEFAULT_DARK_FRACTION,
    DEFAULT_DARK_THRESHOLD,
    DEFAULT_EDGE_RATIO_BAND,
    DEFAULT_GYROSCOPE_MODEL,
    DEFAULT_MAX_SIDE,
    DEFAULT_MAX_TRIM_FRACTION,
    DEFAULT_MAX_TRIM_PX,
    DEFAULT_MIN_AREA,
    DEFAULT_MODEL_PATH,
    DEFAULT_PADDING,
    DEFAULT_REVIEW_MIN_MARGIN,
    DEFAULT_REVIEW_MIN_SCORE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_THRESHOLD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract individual cropped and oriented photos from one or more flatbed scans."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Flatbed scan image(s) to process.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--batch-name",
        help="Name for this extraction batch. Defaults to the current datetime, e.g. 20260521-231500.",
    )
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING)
    parser.add_argument("--dark-threshold", type=int, default=DEFAULT_DARK_THRESHOLD)
    parser.add_argument("--dark-fraction", type=float, default=DEFAULT_DARK_FRACTION)
    parser.add_argument("--max-trim-px", type=int, default=DEFAULT_MAX_TRIM_PX)
    parser.add_argument("--max-trim-fraction", type=float, default=DEFAULT_MAX_TRIM_FRACTION)
    parser.add_argument("--edge-ratio-band", type=int, default=DEFAULT_EDGE_RATIO_BAND)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the YuNet ONNX face detector.",
    )
    parser.add_argument("--gyroscope-model", default=DEFAULT_GYROSCOPE_MODEL)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument(
        "--flag-review",
        action="store_true",
        help="Flag low-confidence orientation results in metadata. By default, always accept the best-scoring rotation.",
    )
    parser.add_argument("--review-min-score", type=float, default=DEFAULT_REVIEW_MIN_SCORE)
    parser.add_argument("--review-min-margin", type=float, default=DEFAULT_REVIEW_MIN_MARGIN)
    parser.add_argument(
        "--debug-panel-width",
        type=int,
        help="Resize each debug panel to this width. By default, debug panels are not scaled down.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not write per-scan debug PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from extractor.batch import run_batch

    run_batch(
        args.inputs,
        args.output_dir,
        args.batch_name,
        args.min_area,
        args.threshold,
        args.padding,
        args.dark_threshold,
        args.dark_fraction,
        args.max_trim_px,
        args.max_trim_fraction,
        args.edge_ratio_band,
        args.model,
        args.gyroscope_model,
        args.max_side,
        args.score_threshold,
        args.flag_review,
        args.review_min_score,
        args.review_min_margin,
        args.debug_panel_width,
        not args.no_debug,
    )
