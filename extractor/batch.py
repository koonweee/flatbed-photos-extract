"""Batch orchestration for flatbed scan extraction."""

from __future__ import annotations

from pathlib import Path

from .core import (
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
    GyroScopeClassifier,
    ScanResult,
    default_batch_name,
    make_detector,
    process_scan,
    write_metadata,
)


def run_batch(
    input_paths: list[Path],
    output_dir: Path,
    batch_name: str | None,
    min_area: int = DEFAULT_MIN_AREA,
    threshold: int = DEFAULT_THRESHOLD,
    padding: int = DEFAULT_PADDING,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    dark_fraction: float = DEFAULT_DARK_FRACTION,
    max_trim_px: int = DEFAULT_MAX_TRIM_PX,
    max_trim_fraction: float = DEFAULT_MAX_TRIM_FRACTION,
    edge_ratio_band: int = DEFAULT_EDGE_RATIO_BAND,
    model_path: Path = DEFAULT_MODEL_PATH,
    gyroscope_model: str = DEFAULT_GYROSCOPE_MODEL,
    max_side: int = DEFAULT_MAX_SIDE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    flag_review: bool = False,
    review_min_score: float = DEFAULT_REVIEW_MIN_SCORE,
    review_min_margin: float = DEFAULT_REVIEW_MIN_MARGIN,
    debug_panel_width: int | None = None,
    write_debug: bool = True,
) -> list[ScanResult]:
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find YuNet model: {model_path}")

    detector = make_detector(model_path, 320, 320, score_threshold)
    gyroscope = GyroScopeClassifier(gyroscope_model)
    summaries = []
    batch_dir = output_dir / (batch_name or default_batch_name())
    photos_dir = batch_dir / "photos"
    debug_dir = batch_dir / "debug"
    metadata_path = batch_dir / "metadata.csv"

    photos_dir.mkdir(parents=True, exist_ok=True)

    for path in photos_dir.glob("*.png"):
        path.unlink()
    if debug_dir.exists():
        for path in debug_dir.glob("*.png"):
            path.unlink()
        if not write_debug:
            try:
                debug_dir.rmdir()
            except OSError:
                pass
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()

    for input_path in input_paths:
        summaries.append(
            process_scan(
                input_path,
                photos_dir,
                debug_dir,
                input_path.stem,
                write_debug=write_debug,
                debug_panel_width=debug_panel_width,
                min_area=min_area,
                threshold=threshold,
                padding=padding,
                dark_threshold=dark_threshold,
                dark_fraction=dark_fraction,
                max_trim_px=max_trim_px,
                max_trim_fraction=max_trim_fraction,
                edge_ratio_band=edge_ratio_band,
                model_path=model_path,
                gyroscope_model=gyroscope_model,
                max_side=max_side,
                score_threshold=score_threshold,
                flag_review=flag_review,
                review_min_score=review_min_score,
                review_min_margin=review_min_margin,
                detector=detector,
                gyroscope=gyroscope,
            )
        )

    all_detections = [detection for summary in summaries for detection in summary.detections]
    write_metadata(metadata_path, all_detections)

    total_elapsed_ms = sum(item.elapsed_ms for item in summaries)
    total_orientation_ms = sum(item.orientation_elapsed_ms for item in summaries)
    total_photos = sum(item.photos for item in summaries)
    total_review = sum(item.needs_review for item in summaries)
    print(
        "summary: "
        f"batch={batch_dir} inputs={len(summaries)} photos={total_photos} needs_review={total_review} "
        f"total_ms={total_elapsed_ms:.1f} orientation_ms={total_orientation_ms:.1f} "
        f"avg_orientation_ms_per_photo={(total_orientation_ms / total_photos if total_photos else 0):.1f}"
    )
    return summaries


run = run_batch
