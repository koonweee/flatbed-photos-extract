#!/usr/bin/env python3
"""Extract individual photos from flatbed scans."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification


EXTRACTOR_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = EXTRACTOR_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_GYROSCOPE_MODEL = "LH-Tech-AI/GyroScope"
DEFAULT_MIN_AREA = 45_000
DEFAULT_THRESHOLD = 90
DEFAULT_PADDING = 45
DEFAULT_DARK_THRESHOLD = 35
DEFAULT_DARK_FRACTION = 0.15
DEFAULT_MAX_TRIM_PX = 10
DEFAULT_MAX_TRIM_FRACTION = 0.008
DEFAULT_EDGE_RATIO_BAND = 4
DEFAULT_MAX_SIDE = 384
DEFAULT_SCORE_THRESHOLD = 0.55
DEFAULT_REVIEW_MIN_SCORE = 0.55
DEFAULT_REVIEW_MIN_MARGIN = 0.08
DEFAULT_LIGHT_BACKGROUND_THRESHOLD = 210
DEFAULT_PHOTO_MAX_ASPECT = 2.0
DEFAULT_DOCUMENT_MAX_ASPECT = 3.8
DEFAULT_PHOTO_MAX_AREA_FRACTION = 0.2
DEFAULT_DOCUMENT_MAX_AREA_FRACTION = 0.65
DEFAULT_DOCUMENT_MIN_SHORT_SIDE = 450
DEFAULT_MIN_SIZE = 250
DEFAULT_EDGE_MARGIN = 5
DEFAULT_BACKGROUND_CONTRAST_DELTA = 30
DEFAULT_BACKGROUND_COLOR_DISTANCE = 42.0
DEFAULT_BACKGROUND_TRIM_FRACTION = 0.55
DEFAULT_BACKGROUND_MAX_TRIM_PX = 28
DEFAULT_BACKGROUND_MAX_TRIM_FRACTION = 0.018
DEFAULT_TRANSITION_LIGHT_BACKGROUND_MIN = 128.0
DEFAULT_TRANSITION_MIN_SCORE = 18.0
DEFAULT_TRANSITION_STEP_PX = 7
DEFAULT_TRANSITION_OUTSIDE_RADIUS = 42
DEFAULT_TRANSITION_INSIDE_RADIUS = 56
DEFAULT_TRANSITION_SAMPLE_GAP_PX = 5
DEFAULT_TRANSITION_MAX_SHIFT_PX = 44.0
DEFAULT_TRANSITION_MIN_POINTS = 24
DEFAULT_TRANSITION_MIN_AREA_RATIO = 0.90
DEFAULT_TRANSITION_MAX_AREA_RATIO = 1.08
ROTATIONS = (0, 90, 180, 270)
SIDES = ("top", "right", "bottom", "left")
ANGLE_BY_CLASS = {0: 0, 1: 90, 2: 180, 3: 270}
METADATA_COLUMNS = [
    "source_file",
    "source_stem",
    "source_photo_index",
    "filename",
    "source_x",
    "source_y",
    "source_width",
    "source_height",
    "corner_tl_x",
    "corner_tl_y",
    "corner_tr_x",
    "corner_tr_y",
    "corner_br_x",
    "corner_br_y",
    "corner_bl_x",
    "corner_bl_y",
    "output_width",
    "output_height",
    "trimmed_output_width",
    "trimmed_output_height",
    "trim_left_px",
    "trim_top_px",
    "trim_right_px",
    "trim_bottom_px",
    "trim_total_px",
    "trim_width_pct",
    "trim_height_pct",
    "dark_edge_ratio_before",
    "dark_edge_ratio_after",
    "source_rotation_deg_clockwise_estimate",
    "orientation_deg",
    "orientation_score",
    "orientation_margin",
    "face_count",
    "orientation_method",
    "needs_review",
    "orientation_scores",
    "yunet_orientation_deg",
    "yunet_orientation_score",
    "yunet_orientation_margin",
    "yunet_face_count",
    "yunet_orientation_scores",
    "gyroscope_orientation_deg",
    "gyroscope_orientation_score",
    "gyroscope_orientation_margin",
    "gyroscope_orientation_scores",
    "refined",
    "refine_reason",
    "contour_area",
]


@dataclass(frozen=True)
class ScanResult:
    input_path: Path
    photos_dir: Path
    debug_dir: Path
    source_stem: str
    detections: list[dict]
    oriented_paths: list[Path]
    photos: int
    needs_review: int
    elapsed_ms: float
    orientation_elapsed_ms: float


@dataclass(frozen=True)
class TransitionSide:
    points: np.ndarray
    line: tuple[float, float, float] | None
    median_score: float
    p25_score: float
    accepted_count: int
    sampled_count: int
    median_shift_px: float


def default_batch_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def find_quadrilateral(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    for factor in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1):
        approx = cv2.approxPolyDP(contour, factor * perimeter, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    return order_points(cv2.boxPoints(cv2.minAreaRect(contour)))


def edge_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    return top, right, bottom, left


def rectified_size(quad: np.ndarray) -> tuple[int, int]:
    top, right, bottom, left = edge_lengths(quad)
    return max(1, int(round((top + bottom) / 2))), max(1, int(round((left + right) / 2)))


def clockwise_angle_degrees(quad: np.ndarray) -> float:
    tl, tr, _br, _bl = quad
    dx, dy = tr - tl
    return math.degrees(math.atan2(dy, dx))


def line_from_points(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 12:
        return None

    points = points.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(4)
    # Normal form: ax + by + c = 0.
    a = float(vy)
    b = float(-vx)
    c = float(vx * y0 - vy * x0)
    norm = math.hypot(a, b)
    if norm == 0:
        return None
    return a / norm, b / norm, c / norm


def intersect_lines(line_a: tuple[float, float, float], line_b: tuple[float, float, float]) -> np.ndarray | None:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    denominator = a1 * b2 - a2 * b1
    if abs(denominator) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / denominator
    y = (c1 * a2 - c2 * a1) / denominator
    return np.array([x, y], dtype=np.float32)


def sample_edge_points(mask: np.ndarray, side: str, margin: int) -> np.ndarray:
    height, width = mask.shape
    points = []

    if side in {"top", "bottom"}:
        xs = range(margin, width - margin)
        for x in xs:
            column = mask[:, x]
            ys = np.flatnonzero(column > 0)
            if len(ys) == 0:
                continue
            y = int(ys[0] if side == "top" else ys[-1])
            if margin <= y < height - margin:
                points.append((x, y))
    else:
        ys = range(margin, height - margin)
        for y in ys:
            row = mask[y, :]
            xs = np.flatnonzero(row > 0)
            if len(xs) == 0:
                continue
            x = int(xs[0] if side == "left" else xs[-1])
            if margin <= x < width - margin:
                points.append((x, y))

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    points_array = np.array(points, dtype=np.float32)
    # Keep the outer quantile only. This avoids interior printed-photo edges and
    # lets aged or stained borders still contribute to the fitted paper edge.
    if side == "top":
        cutoff = np.quantile(points_array[:, 1], 0.25)
        points_array = points_array[points_array[:, 1] <= cutoff + 3]
    elif side == "bottom":
        cutoff = np.quantile(points_array[:, 1], 0.75)
        points_array = points_array[points_array[:, 1] >= cutoff - 3]
    elif side == "left":
        cutoff = np.quantile(points_array[:, 0], 0.25)
        points_array = points_array[points_array[:, 0] <= cutoff + 3]
    elif side == "right":
        cutoff = np.quantile(points_array[:, 0], 0.75)
        points_array = points_array[points_array[:, 0] >= cutoff - 3]

    return points_array


def sample_background_gray(gray: np.ndarray, border_px: int = 80, inset_fraction: float = 0.05) -> float:
    inset_y = int(round(gray.shape[0] * inset_fraction))
    inset_x = int(round(gray.shape[1] * inset_fraction))
    inner = gray[inset_y : gray.shape[0] - inset_y, inset_x : gray.shape[1] - inset_x]
    if inner.size == 0:
        inner = gray
    border_px = max(1, min(border_px, inner.shape[0] // 3, inner.shape[1] // 3))
    border = np.concatenate(
        [
            inner[:border_px, :].reshape(-1),
            inner[-border_px:, :].reshape(-1),
            inner[:, :border_px].reshape(-1),
            inner[:, -border_px:].reshape(-1),
        ]
    )
    return float(np.median(border))


def sample_background_bgr(image_bgr: np.ndarray, border_px: int = 80, inset_fraction: float = 0.05) -> np.ndarray:
    inset_y = int(round(image_bgr.shape[0] * inset_fraction))
    inset_x = int(round(image_bgr.shape[1] * inset_fraction))
    inner = image_bgr[inset_y : image_bgr.shape[0] - inset_y, inset_x : image_bgr.shape[1] - inset_x]
    if inner.size == 0:
        inner = image_bgr
    border_px = max(1, min(border_px, inner.shape[0] // 3, inner.shape[1] // 3))
    border = np.concatenate(
        [
            inner[:border_px, :, :].reshape(-1, 3),
            inner[-border_px:, :, :].reshape(-1, 3),
            inner[:, :border_px, :].reshape(-1, 3),
            inner[:, -border_px:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border, axis=0).astype(np.float32)


def remove_edge_connected_components(mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    if count <= 1:
        return mask

    edge_labels = set(labels[0, :])
    edge_labels.update(labels[-1, :])
    edge_labels.update(labels[:, 0])
    edge_labels.update(labels[:, -1])
    cleaned = mask.copy()
    for label in edge_labels:
        if label:
            cleaned[labels == label] = 0
    return cleaned


def build_foreground_mask(gray: np.ndarray, threshold: int, foreground_polarity: str) -> np.ndarray:
    if foreground_polarity == "dark-foreground":
        threshold_type = cv2.THRESH_BINARY_INV
        threshold = DEFAULT_LIGHT_BACKGROUND_THRESHOLD
    else:
        threshold_type = cv2.THRESH_BINARY
    _ret, mask = cv2.threshold(gray, threshold, 255, threshold_type)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if foreground_polarity == "dark-foreground":
        mask = remove_edge_connected_components(mask)
    return mask


def classify_candidate(width: int, height: int, area: float, image_area: int) -> str | None:
    aspect = max(width / height, height / width)
    short_side = min(width, height)
    area_fraction = area / max(image_area, 1)

    if aspect <= DEFAULT_PHOTO_MAX_ASPECT and area_fraction <= DEFAULT_PHOTO_MAX_AREA_FRACTION:
        return "photo"
    if (
        aspect <= DEFAULT_DOCUMENT_MAX_ASPECT
        and area_fraction <= DEFAULT_DOCUMENT_MAX_AREA_FRACTION
        and short_side >= DEFAULT_DOCUMENT_MIN_SHORT_SIDE
    ):
        return "document"
    return None


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        if any(bbox_iou(candidate["bbox"], existing["bbox"]) > 0.85 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def side_position(quad: np.ndarray, side: str, coordinate: float) -> float:
    tl, tr, br, bl = quad
    if side == "top":
        start, end = tl, tr
        axis = 0
    elif side == "bottom":
        start, end = bl, br
        axis = 0
    elif side == "left":
        start, end = tl, bl
        axis = 1
    else:
        start, end = tr, br
        axis = 1

    delta = float(end[axis] - start[axis])
    if abs(delta) < 1e-6:
        return float((start[1 - axis] + end[1 - axis]) / 2)
    t = (coordinate - float(start[axis])) / delta
    return float(start[1 - axis] + t * (end[1 - axis] - start[1 - axis]))


def side_range(quad: np.ndarray, side: str, margin: int) -> tuple[int, int]:
    tl, tr, br, bl = quad
    if side in {"top", "bottom"}:
        values = (tl[0], tr[0]) if side == "top" else (bl[0], br[0])
    else:
        values = (tl[1], bl[1]) if side == "left" else (tr[1], br[1])
    low = math.ceil(min(values)) + margin
    high = math.floor(max(values)) - margin
    return low, high


def fit_transition_side(
    local_gray: np.ndarray,
    local_quad: np.ndarray,
    side: str,
    *,
    polarity: str,
) -> TransitionSide:
    height, width = local_gray.shape
    margin = max(8, DEFAULT_TRANSITION_OUTSIDE_RADIUS // 2)
    low, high = side_range(local_quad, side, margin)
    axis_limit = width if side in {"top", "bottom"} else height
    low = max(margin, low)
    high = min(axis_limit - margin - 1, high)
    if high <= low:
        return TransitionSide(np.empty((0, 2), dtype=np.float32), None, 0.0, 0.0, 0, 0, 0.0)

    scan_values = list(range(low, high + 1, max(1, DEFAULT_TRANSITION_STEP_PX)))
    points: list[tuple[float, float]] = []
    scores: list[float] = []
    shifts: list[float] = []
    dark_foreground = polarity == "dark-foreground"

    for coordinate in scan_values:
        expected = side_position(local_quad, side, float(coordinate))
        start = int(round(expected - DEFAULT_TRANSITION_OUTSIDE_RADIUS))
        stop = int(round(expected + DEFAULT_TRANSITION_INSIDE_RADIUS))
        if side in {"bottom", "right"}:
            start = int(round(expected - DEFAULT_TRANSITION_INSIDE_RADIUS))
            stop = int(round(expected + DEFAULT_TRANSITION_OUTSIDE_RADIUS))

        if side in {"top", "bottom"}:
            start = max(DEFAULT_TRANSITION_SAMPLE_GAP_PX, start)
            stop = min(height - DEFAULT_TRANSITION_SAMPLE_GAP_PX - 1, stop)
            if stop <= start:
                continue
            profile = local_gray[:, coordinate].astype(np.float32)
        else:
            start = max(DEFAULT_TRANSITION_SAMPLE_GAP_PX, start)
            stop = min(width - DEFAULT_TRANSITION_SAMPLE_GAP_PX - 1, stop)
            if stop <= start:
                continue
            profile = local_gray[coordinate, :].astype(np.float32)

        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).reshape(-1)
        best_score = -1.0
        best_pos: int | None = None
        for pos in range(start, stop + 1):
            before_pos = pos - DEFAULT_TRANSITION_SAMPLE_GAP_PX
            after_pos = pos + DEFAULT_TRANSITION_SAMPLE_GAP_PX
            if side in {"bottom", "right"}:
                outside_value = profile[after_pos]
                inside_value = profile[before_pos]
            else:
                outside_value = profile[before_pos]
                inside_value = profile[after_pos]
            score = (outside_value - inside_value) if dark_foreground else (inside_value - outside_value)
            if score > best_score:
                best_score = float(score)
                best_pos = pos

        if best_pos is None or best_score < DEFAULT_TRANSITION_MIN_SCORE:
            continue
        shift = float(best_pos - expected)
        if abs(shift) > DEFAULT_TRANSITION_MAX_SHIFT_PX:
            continue
        point = (float(coordinate), float(best_pos)) if side in {"top", "bottom"} else (float(best_pos), float(coordinate))
        points.append(point)
        scores.append(best_score)
        shifts.append(shift)

    if not points:
        return TransitionSide(np.empty((0, 2), dtype=np.float32), None, 0.0, 0.0, 0, len(scan_values), 0.0)

    points_array = np.array(points, dtype=np.float32)
    score_array = np.array(scores, dtype=np.float32)
    shift_array = np.array(shifts, dtype=np.float32)

    median_shift = float(np.median(shift_array))
    shift_mad = float(np.median(np.abs(shift_array - median_shift)))
    shift_limit = max(4.0, min(16.0, 2.5 * shift_mad + 3.0))
    keep = np.abs(shift_array - median_shift) <= shift_limit
    points_array = points_array[keep]
    score_array = score_array[keep]
    shift_array = shift_array[keep]

    line = line_from_points(points_array)
    median_score = float(np.median(score_array)) if score_array.size else 0.0
    p25_score = float(np.quantile(score_array, 0.25)) if score_array.size else 0.0
    median_shift = float(np.median(shift_array)) if shift_array.size else 0.0
    return TransitionSide(
        points=points_array,
        line=line,
        median_score=median_score,
        p25_score=p25_score,
        accepted_count=len(points_array),
        sampled_count=len(scan_values),
        median_shift_px=median_shift,
    )


def refine_quad_with_outer_edges(
    gray: np.ndarray,
    rough_quad: np.ndarray,
    threshold: int,
    padding: int,
    *,
    background_gray: float | None = None,
    foreground_polarity: str | None = None,
) -> tuple[np.ndarray, dict]:
    background_gray = sample_background_gray(gray) if background_gray is None else float(background_gray)
    foreground_polarity = foreground_polarity or ("dark-foreground" if background_gray >= 128.0 else "light-foreground")
    if background_gray >= DEFAULT_TRANSITION_LIGHT_BACKGROUND_MIN and foreground_polarity == "dark-foreground":
        quad, debug = refine_quad_with_transition_edges(gray, rough_quad, threshold, padding, background_gray)
        if debug.get("transition_refined"):
            return quad, debug
        fallback_quad, fallback_debug = refine_quad_with_background_edges(
            gray,
            rough_quad,
            threshold,
            padding,
            background_gray=background_gray,
            foreground_polarity=foreground_polarity,
        )
        return fallback_quad, {**fallback_debug, **debug}
    return refine_quad_with_background_edges(
        gray,
        rough_quad,
        threshold,
        padding,
        background_gray=background_gray,
        foreground_polarity=foreground_polarity,
    )


def refine_quad_with_background_edges(
    gray: np.ndarray,
    rough_quad: np.ndarray,
    threshold: int,
    padding: int,
    *,
    background_gray: float,
    foreground_polarity: str,
) -> tuple[np.ndarray, dict]:
    x, y, w, h = cv2.boundingRect(rough_quad.astype(np.int32))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray.shape[1], x + w + padding)
    y1 = min(gray.shape[0], y + h + padding)

    local_gray = gray[y0:y1, x0:x1]
    if foreground_polarity == "dark-foreground":
        local_threshold = int(round(max(0, min(255, background_gray - DEFAULT_BACKGROUND_CONTRAST_DELTA))))
        _ret, local_mask = cv2.threshold(local_gray, local_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        local_threshold = int(round(max(threshold, min(255, background_gray + DEFAULT_BACKGROUND_CONTRAST_DELTA))))
        _ret, local_mask = cv2.threshold(local_gray, local_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    margin = max(5, padding // 3)
    edge_points = {
        side: sample_edge_points(local_mask, side, margin)
        for side in ("top", "right", "bottom", "left")
    }
    lines = {side: line_from_points(points) for side, points in edge_points.items()}

    if any(lines[side] is None for side in ("top", "right", "bottom", "left")):
        return rough_quad, {
            "refined": False,
            "reason": "insufficient background-aware edge points",
            "polarity": foreground_polarity,
            "background_gray": background_gray,
            "local_threshold": local_threshold,
            "edge_point_counts": {side: len(points) for side, points in edge_points.items()},
        }

    local_corners = [
        intersect_lines(lines["top"], lines["left"]),
        intersect_lines(lines["top"], lines["right"]),
        intersect_lines(lines["bottom"], lines["right"]),
        intersect_lines(lines["bottom"], lines["left"]),
    ]
    if any(corner is None for corner in local_corners):
        return rough_quad, {
            "refined": False,
            "reason": "parallel background-aware edge lines",
            "polarity": foreground_polarity,
            "background_gray": background_gray,
            "local_threshold": local_threshold,
        }

    local_quad = order_points(np.array(local_corners, dtype=np.float32))
    global_quad = local_quad + np.array([x0, y0], dtype=np.float32)

    rough_area = cv2.contourArea(rough_quad.astype(np.float32))
    refined_area = cv2.contourArea(global_quad.astype(np.float32))
    if refined_area < rough_area * 0.75 or refined_area > rough_area * 1.25:
        return rough_quad, {
            "refined": False,
            "reason": "background-aware area sanity check failed",
            "polarity": foreground_polarity,
            "background_gray": background_gray,
            "local_threshold": local_threshold,
            "rough_area": rough_area,
            "refined_area": refined_area,
        }

    debug = {
        "refined": True,
        "local_origin": (x0, y0),
        "polarity": foreground_polarity,
        "background_gray": background_gray,
        "local_threshold": local_threshold,
        "edge_point_counts": {side: len(points) for side, points in edge_points.items()},
    }
    return global_quad, debug


def refine_quad_with_transition_edges(
    gray: np.ndarray,
    rough_quad: np.ndarray,
    threshold: int,
    padding: int,
    background_gray: float,
) -> tuple[np.ndarray, dict]:
    x, y, w, h = cv2.boundingRect(rough_quad.astype(np.int32))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray.shape[1], x + w + padding)
    y1 = min(gray.shape[0], y + h + padding)
    local_gray = gray[y0:y1, x0:x1]
    local_quad = rough_quad.astype(np.float32) - np.array([x0, y0], dtype=np.float32)

    sides = {
        side: fit_transition_side(local_gray, local_quad, side, polarity="dark-foreground")
        for side in SIDES
    }
    debug_base = {
        "transition_refined": False,
        "transition_background_gray": background_gray,
        "transition_side_counts": {side: sides[side].accepted_count for side in SIDES},
        "transition_side_scores": {side: round(sides[side].median_score, 3) for side in SIDES},
        "transition_side_shifts": {side: round(sides[side].median_shift_px, 3) for side in SIDES},
    }
    missing = [side for side, info in sides.items() if info.line is None or info.accepted_count < DEFAULT_TRANSITION_MIN_POINTS]
    if missing:
        return rough_quad, {
            **debug_base,
            "refined": False,
            "reason": f"insufficient transition points: {','.join(missing)}",
            "transition_reason": f"insufficient transition points: {','.join(missing)}",
        }

    local_corners = [
        intersect_lines(sides["top"].line, sides["left"].line),
        intersect_lines(sides["top"].line, sides["right"].line),
        intersect_lines(sides["bottom"].line, sides["right"].line),
        intersect_lines(sides["bottom"].line, sides["left"].line),
    ]
    if any(corner is None for corner in local_corners):
        return rough_quad, {
            **debug_base,
            "refined": False,
            "reason": "parallel transition lines",
            "transition_reason": "parallel transition lines",
        }

    local_quad_refined = order_points(np.array(local_corners, dtype=np.float32))
    global_quad = local_quad_refined + np.array([x0, y0], dtype=np.float32)
    rough_area = cv2.contourArea(rough_quad.astype(np.float32))
    refined_area = cv2.contourArea(global_quad.astype(np.float32))
    if refined_area < rough_area * DEFAULT_TRANSITION_MIN_AREA_RATIO or refined_area > rough_area * DEFAULT_TRANSITION_MAX_AREA_RATIO:
        return rough_quad, {
            **debug_base,
            "refined": False,
            "reason": "transition area sanity check failed",
            "transition_reason": "transition area sanity check failed",
            "transition_rough_area": rough_area,
            "transition_refined_area": refined_area,
        }

    return global_quad, {
        **debug_base,
        "refined": True,
        "transition_refined": True,
        "transition_reason": "",
        "transition_rough_area": rough_area,
        "transition_refined_area": refined_area,
        "local_origin": (x0, y0),
    }


def rough_candidates(image_bgr: np.ndarray, threshold: int, min_area: int) -> list[dict]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    background_gray = sample_background_gray(gray)
    background_bgr = sample_background_bgr(image_bgr)
    background_is_light = background_gray >= 128.0
    polarities = ("dark-foreground",) if background_is_light else ("light-foreground",)
    candidates = []
    for foreground_polarity in polarities:
        mask = build_foreground_mask(gray, threshold, foreground_polarity)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            touches_scan_edge = (
                x <= DEFAULT_EDGE_MARGIN
                or y <= DEFAULT_EDGE_MARGIN
                or x + w >= image_bgr.shape[1] - DEFAULT_EDGE_MARGIN
                or y + h >= image_bgr.shape[0] - DEFAULT_EDGE_MARGIN
            )
            if touches_scan_edge or w < DEFAULT_MIN_SIZE or h < DEFAULT_MIN_SIZE:
                continue

            rough_quad = find_quadrilateral(contour)
            width, height = rectified_size(rough_quad)
            candidate_type = classify_candidate(width, height, area, image_area)
            if candidate_type is None:
                continue

            candidates.append(
                {
                    "bbox": (x, y, w, h),
                    "area": area,
                    "rough_quad": rough_quad,
                    "candidate_type": candidate_type,
                    "foreground_polarity": foreground_polarity,
                    "background_gray": background_gray,
                    "background_bgr": background_bgr,
                    "background_is_light": background_is_light,
                }
            )

    candidates = dedupe_candidates(candidates)
    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return candidates


def make_debug_overlay(image_rgb: np.ndarray, detections: list[dict]) -> Image.Image:
    overlay = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        rough = [(float(x), float(y)) for x, y in detection["rough_quad"]]
        refined = [(float(x), float(y)) for x, y in detection["quad"]]
        draw.line(rough + [rough[0]], fill=(255, 180, 0), width=5)
        draw.line(refined + [refined[0]], fill=(0, 255, 80), width=6)
        draw.text((refined[0][0] + 12, refined[0][1] + 12), detection["filename"], fill=(255, 255, 0))
    return overlay


def make_contact_sheet_image(paths: list[Path], thumb_size: int = 360) -> Image.Image | None:
    if not paths:
        return None

    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_size, thumb_size), (32, 32, 32))
        tile.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        thumbs.append(tile)

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    gap = 14
    sheet = Image.new("RGB", (cols * thumb_size + (cols + 1) * gap, rows * thumb_size + (rows + 1) * gap), (24, 24, 24))
    for index, tile in enumerate(thumbs):
        row, col = divmod(index, cols)
        sheet.paste(tile, (gap + col * (thumb_size + gap), gap + row * (thumb_size + gap)))
    return sheet


def make_contact_sheet_from_arrays(images: list[np.ndarray], thumb_size: int = 360) -> Image.Image | None:
    if not images:
        return None

    thumbs = []
    for array in images:
        image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_size, thumb_size), (32, 32, 32))
        tile.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        thumbs.append(tile)

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    gap = 14
    sheet = Image.new("RGB", (cols * thumb_size + (cols + 1) * gap, rows * thumb_size + (rows + 1) * gap), (24, 24, 24))
    for index, tile in enumerate(thumbs):
        row, col = divmod(index, cols)
        sheet.paste(tile, (gap + col * (thumb_size + gap), gap + row * (thumb_size + gap)))
    return sheet


def make_pipeline_debug_image(
    original: Image.Image,
    mask: Image.Image,
    outline: Image.Image,
    before_orientation: Image.Image | None,
    final: Image.Image | None,
    output_path: Path,
    panel_width: int | None = None,
) -> None:
    panels = []
    fallback_width = panel_width or original.width
    fallback = Image.new("RGB", (fallback_width, fallback_width), (24, 24, 24))
    for image in (original, mask, outline, before_orientation or fallback, final or fallback):
        image = image.convert("RGB")
        if panel_width is not None:
            height = max(1, round(image.height * (panel_width / image.width)))
            image = image.resize((panel_width, height), Image.Resampling.LANCZOS)
        panels.append(image)

    max_height = max(panel.height for panel in panels)
    gap = 18
    total_width = sum(panel.width for panel in panels) + gap * (len(panels) + 1)
    summary = Image.new("RGB", (total_width, max_height + gap * 2), (24, 24, 24))
    x = gap
    for panel in panels:
        summary.paste(panel, (x, gap))
        x += panel.width + gap
    summary.save(output_path)


def metadata_row(detection: dict) -> list:
    x, y, w, h = detection["bbox"]
    tl, tr, br, bl = detection["quad"]
    return [
        detection["source_file"],
        detection["source_stem"],
        detection["source_photo_index"],
        detection["filename"],
        x,
        y,
        w,
        h,
        round(float(tl[0]), 3),
        round(float(tl[1]), 3),
        round(float(tr[0]), 3),
        round(float(tr[1]), 3),
        round(float(br[0]), 3),
        round(float(br[1]), 3),
        round(float(bl[0]), 3),
        round(float(bl[1]), 3),
        detection["width"],
        detection["height"],
        detection["trimmed_width"],
        detection["trimmed_height"],
        detection["trim_left"],
        detection["trim_top"],
        detection["trim_right"],
        detection["trim_bottom"],
        detection["trim_left"] + detection["trim_top"] + detection["trim_right"] + detection["trim_bottom"],
        round((detection["trim_left"] + detection["trim_right"]) / detection["width"], 6),
        round((detection["trim_top"] + detection["trim_bottom"]) / detection["height"], 6),
        round(detection["dark_edge_ratio_before"], 6),
        round(detection["dark_edge_ratio_after"], 6),
        round(detection["angle"], 6),
        detection["orientation_deg"],
        round(detection["orientation_score"], 6),
        round(detection["orientation_margin"], 6),
        detection["face_count"],
        detection["orientation_method"],
        detection["needs_review"],
        " ".join(
            f"{item['rotation']}:{item['score']:.6f}/{item['face_count']}"
            for item in detection["orientation_scores"]
        ),
        detection["yunet_orientation_deg"],
        round(detection["yunet_orientation_score"], 6),
        round(detection["yunet_orientation_margin"], 6),
        detection["yunet_face_count"],
        " ".join(
            f"{item['rotation']}:{item['score']:.6f}/{item['face_count']}"
            for item in detection["yunet_orientation_scores"]
        ),
        detection["gyroscope_orientation_deg"],
        round(detection["gyroscope_orientation_score"], 6),
        round(detection["gyroscope_orientation_margin"], 6),
        " ".join(
            f"{item['rotation']}:{item['score']:.6f}"
            for item in detection["gyroscope_orientation_scores"]
        ),
        detection["refined"],
        detection["refine_reason"],
        round(detection["area"], 1),
    ]


def write_metadata(metadata_path: Path, detections: list[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_COLUMNS)
        writer.writerows(metadata_row(detection) for detection in detections)


def append_metadata(metadata_path: Path, detections: list[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    new_source_stems = {detection["source_stem"] for detection in detections}
    existing_rows = []
    if metadata_path.exists() and metadata_path.stat().st_size:
        with metadata_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header == METADATA_COLUMNS:
                existing_rows = [
                    row
                    for row in reader
                    if len(row) > 1 and row[1] not in new_source_stems
                ]

    temp_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_COLUMNS)
        writer.writerows(existing_rows)
        writer.writerows(metadata_row(detection) for detection in detections)
    temp_path.replace(metadata_path)


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {degrees}")


def resize_for_detection(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def make_detector(model_path: Path, width: int, height: int, score_threshold: float) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (width, height),
        score_threshold,
        0.3,
        5000,
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )


def score_faces_best_plus_small_bonus(faces: np.ndarray | None) -> tuple[float, int]:
    if faces is None or len(faces) == 0:
        return 0.0, 0

    # YuNet columns: x, y, w, h, landmarks..., score. Prioritize the best face;
    # extra faces only help a little so one false extra face cannot dominate.
    face_scores = []
    for face in faces:
        width = float(face[2])
        height = float(face[3])
        confidence = float(face[-1])
        area_bonus = min(0.08, (width * height) / 1_200_000)
        face_scores.append(confidence + area_bonus)
    face_scores.sort(reverse=True)
    score = face_scores[0] + 0.03 * sum(face_scores[1:])
    return score, int(len(face_scores))


def classify_orientation_with_detector(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN,
    max_side: int,
) -> dict:
    scores = []
    for degrees in ROTATIONS:
        rotated = rotate_image(image, degrees)
        preview = resize_for_detection(rotated, max_side)
        height, width = preview.shape[:2]
        detector.setInputSize((width, height))
        _retval, faces = detector.detect(preview)
        score, face_count = score_faces_best_plus_small_bonus(faces)
        scores.append({"rotation": degrees, "score": score, "face_count": face_count})

    best = max(scores, key=lambda item: item["score"])
    sorted_scores = sorted(scores, key=lambda item: item["score"], reverse=True)
    margin = sorted_scores[0]["score"] - sorted_scores[1]["score"]
    return {
        "rotation": best["rotation"],
        "score": best["score"],
        "margin": margin,
        "face_count": best["face_count"],
        "scores": scores,
    }


class GyroScopeClassifier:
    def __init__(self, model_name: str) -> None:
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def classify(self, image_bgr: np.ndarray) -> dict:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(image_rgb)).unsqueeze(0)
        logits = self.model(pixel_values=tensor).logits[0]
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        best_index = int(np.argmax(probabilities))
        ranked = np.sort(probabilities)[::-1]
        scores = [
            {"rotation": ANGLE_BY_CLASS[index], "score": float(probability), "face_count": ""}
            for index, probability in enumerate(probabilities)
        ]
        return {
            "rotation": ANGLE_BY_CLASS[best_index],
            "score": float(probabilities[best_index]),
            "margin": float(ranked[0] - ranked[1]),
            "face_count": "",
            "scores": scores,
        }


def classify_orientation_hybrid(
    image: np.ndarray,
    detector: cv2.FaceDetectorYN,
    gyroscope: GyroScopeClassifier,
    max_side: int,
) -> dict:
    yunet = classify_orientation_with_detector(image, detector, max_side)
    gyro = gyroscope.classify(image)
    chosen_method = "yunet" if yunet["face_count"] > 0 and yunet["score"] > 0 else "gyroscope"
    chosen = yunet if chosen_method == "yunet" else gyro
    return {
        "rotation": chosen["rotation"],
        "score": chosen["score"],
        "margin": chosen["margin"],
        "face_count": yunet["face_count"],
        "method": chosen_method,
        "scores": chosen["scores"],
        "yunet": yunet,
        "gyroscope": gyro,
    }


def dark_edge_ratio(image_bgr: np.ndarray, dark_threshold: int, band: int) -> float:
    height, width = image_bgr.shape[:2]
    band = max(1, min(band, height // 2, width // 2))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[:band, :] = True
    edge_mask[-band:, :] = True
    edge_mask[:, :band] = True
    edge_mask[:, -band:] = True
    edge_pixels = gray[edge_mask]
    if edge_pixels.size == 0:
        return 0.0
    return float(np.mean(edge_pixels <= dark_threshold))


def edge_dark_fractions(gray: np.ndarray, side: str, limit: int, dark_threshold: int) -> list[float]:
    fractions = []
    for offset in range(limit):
        if side == "top":
            values = gray[offset, :]
        elif side == "bottom":
            values = gray[gray.shape[0] - 1 - offset, :]
        elif side == "left":
            values = gray[:, offset]
        else:
            values = gray[:, gray.shape[1] - 1 - offset]
        fractions.append(float(np.mean(values <= dark_threshold)))
    return fractions


def contiguous_dark_trim(fractions: list[float], dark_fraction: float) -> int:
    trim = 0
    for fraction in fractions:
        if fraction < dark_fraction:
            break
        trim += 1
    return trim


def edge_background_fractions(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray,
    side: str,
    limit: int,
    color_distance: float,
) -> list[float]:
    fractions = []
    background = background_bgr.reshape(1, 1, 3).astype(np.float32)
    for offset in range(limit):
        if side == "top":
            values = image_bgr[offset : offset + 1, :, :]
        elif side == "bottom":
            values = image_bgr[image_bgr.shape[0] - 1 - offset : image_bgr.shape[0] - offset, :, :]
        elif side == "left":
            values = image_bgr[:, offset : offset + 1, :]
        else:
            values = image_bgr[:, image_bgr.shape[1] - 1 - offset : image_bgr.shape[1] - offset, :]
        distances = np.linalg.norm(values.astype(np.float32) - background, axis=2)
        fractions.append(float(np.mean(distances <= color_distance)))
    return fractions


def contiguous_background_trim(fractions: list[float], background_fraction: float) -> int:
    trim = 0
    for fraction in fractions:
        if fraction < background_fraction:
            break
        trim += 1
    return trim


def trim_dark_edges(
    image_bgr: np.ndarray,
    dark_threshold: int,
    dark_fraction: float,
    max_trim_px: int,
    max_trim_fraction: float,
    edge_ratio_band: int,
    *,
    background_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    before_ratio = dark_edge_ratio(image_bgr, dark_threshold, edge_ratio_band)
    if background_bgr is not None:
        max_trim = max(
            0,
            min(
                DEFAULT_BACKGROUND_MAX_TRIM_PX,
                int(round(min(width, height) * DEFAULT_BACKGROUND_MAX_TRIM_FRACTION)),
            ),
        )
        if max_trim == 0 or width <= 2 or height <= 2:
            return image_bgr, {
                "trim_left": 0,
                "trim_top": 0,
                "trim_right": 0,
                "trim_bottom": 0,
                "dark_edge_ratio_before": before_ratio,
                "dark_edge_ratio_after": before_ratio,
            }

        background_bgr = np.asarray(background_bgr, dtype=np.float32)
        top = contiguous_background_trim(
            edge_background_fractions(image_bgr, background_bgr, "top", max_trim, DEFAULT_BACKGROUND_COLOR_DISTANCE),
            DEFAULT_BACKGROUND_TRIM_FRACTION,
        )
        bottom = contiguous_background_trim(
            edge_background_fractions(image_bgr, background_bgr, "bottom", max_trim, DEFAULT_BACKGROUND_COLOR_DISTANCE),
            DEFAULT_BACKGROUND_TRIM_FRACTION,
        )
        left = contiguous_background_trim(
            edge_background_fractions(image_bgr, background_bgr, "left", max_trim, DEFAULT_BACKGROUND_COLOR_DISTANCE),
            DEFAULT_BACKGROUND_TRIM_FRACTION,
        )
        right = contiguous_background_trim(
            edge_background_fractions(image_bgr, background_bgr, "right", max_trim, DEFAULT_BACKGROUND_COLOR_DISTANCE),
            DEFAULT_BACKGROUND_TRIM_FRACTION,
        )

        if top + bottom >= height:
            top = bottom = 0
        if left + right >= width:
            left = right = 0

        trimmed = image_bgr[top : height - bottom, left : width - right]
        after_ratio = dark_edge_ratio(trimmed, dark_threshold, edge_ratio_band)
        return trimmed, {
            "trim_left": left,
            "trim_top": top,
            "trim_right": right,
            "trim_bottom": bottom,
            "dark_edge_ratio_before": before_ratio,
            "dark_edge_ratio_after": after_ratio,
        }

    max_trim = max(0, min(max_trim_px, int(round(min(width, height) * max_trim_fraction))))
    if max_trim == 0 or width <= 2 or height <= 2:
        return image_bgr, {
            "trim_left": 0,
            "trim_top": 0,
            "trim_right": 0,
            "trim_bottom": 0,
            "dark_edge_ratio_before": before_ratio,
            "dark_edge_ratio_after": before_ratio,
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    top = contiguous_dark_trim(edge_dark_fractions(gray, "top", max_trim, dark_threshold), dark_fraction)
    bottom = contiguous_dark_trim(edge_dark_fractions(gray, "bottom", max_trim, dark_threshold), dark_fraction)
    left = contiguous_dark_trim(edge_dark_fractions(gray, "left", max_trim, dark_threshold), dark_fraction)
    right = contiguous_dark_trim(edge_dark_fractions(gray, "right", max_trim, dark_threshold), dark_fraction)

    # Leave at least one pixel in each direction even if a pathological crop has
    # dark content on both opposing edges.
    if top + bottom >= height:
        top = bottom = 0
    if left + right >= width:
        left = right = 0

    trimmed = image_bgr[top : height - bottom, left : width - right]
    after_ratio = dark_edge_ratio(trimmed, dark_threshold, edge_ratio_band)
    return trimmed, {
        "trim_left": left,
        "trim_top": top,
        "trim_right": right,
        "trim_bottom": bottom,
        "dark_edge_ratio_before": before_ratio,
        "dark_edge_ratio_after": after_ratio,
    }


def process_scan(
    input_path: Path,
    photos_dir: Path,
    debug_dir: Path,
    source_stem: str,
    write_debug: bool = True,
    debug_panel_width: int | None = None,
    *,
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
    detector: cv2.FaceDetectorYN | None = None,
    gyroscope: GyroScopeClassifier | None = None,
) -> ScanResult:
    started = time.perf_counter()
    input_path = Path(input_path)
    photos_dir = Path(photos_dir)
    debug_dir = Path(debug_dir)

    photos_dir.mkdir(parents=True, exist_ok=True)
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if detector is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find YuNet model: {model_path}")
        detector = make_detector(model_path, 320, 320, score_threshold)
    if gyroscope is None:
        gyroscope = GyroScopeClassifier(gyroscope_model)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    original_debug = Image.fromarray(image_rgb) if write_debug else None
    candidates = rough_candidates(image_bgr, threshold, min_area)

    detections = []
    before_orientation_images = []
    oriented_paths = []
    orientation_elapsed_ms = 0.0
    for index, candidate in enumerate(candidates, start=1):
        quad, debug = refine_quad_with_outer_edges(
            gray,
            candidate["rough_quad"],
            threshold,
            padding,
            background_gray=candidate.get("background_gray"),
            foreground_polarity=candidate.get("foreground_polarity"),
        )
        width, height = rectified_size(quad)
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(
            image_bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        trimmed, trim_debug = trim_dark_edges(
            warped,
            dark_threshold,
            dark_fraction,
            max_trim_px,
            max_trim_fraction,
            edge_ratio_band,
            background_bgr=candidate.get("background_bgr"),
        )

        filename = f"{source_stem}_{index:02d}.png"
        if write_debug:
            before_orientation_images.append(trimmed)

        orientation_started = time.perf_counter()
        orientation = classify_orientation_hybrid(trimmed, detector, gyroscope, max_side)
        orientation_elapsed_ms += (time.perf_counter() - orientation_started) * 1000
        oriented = rotate_image(trimmed, orientation["rotation"])
        output_path = photos_dir / filename
        cv2.imwrite(str(output_path), oriented)
        oriented_paths.append(output_path)
        needs_review = bool(
            flag_review
            and (
                orientation["face_count"] == 0
                or orientation["score"] < review_min_score
                or orientation["margin"] < review_min_margin
            )
        )

        detections.append(
            {
                "filename": filename,
                "source_file": str(input_path),
                "source_stem": source_stem,
                "source_photo_index": index,
                "bbox": candidate["bbox"],
                "area": candidate["area"],
                "rough_quad": candidate["rough_quad"],
                "quad": quad,
                "width": width,
                "height": height,
                "trimmed_width": trimmed.shape[1],
                "trimmed_height": trimmed.shape[0],
                "angle": clockwise_angle_degrees(quad),
                "refined": debug.get("refined", False),
                "refine_reason": debug.get("reason", ""),
                "orientation_deg": orientation["rotation"],
                "orientation_score": orientation["score"],
                "orientation_margin": orientation["margin"],
                "face_count": orientation["face_count"],
                "orientation_scores": orientation["scores"],
                "orientation_method": orientation["method"],
                "yunet_orientation_deg": orientation["yunet"]["rotation"],
                "yunet_orientation_score": orientation["yunet"]["score"],
                "yunet_orientation_margin": orientation["yunet"]["margin"],
                "yunet_face_count": orientation["yunet"]["face_count"],
                "yunet_orientation_scores": orientation["yunet"]["scores"],
                "gyroscope_orientation_deg": orientation["gyroscope"]["rotation"],
                "gyroscope_orientation_score": orientation["gyroscope"]["score"],
                "gyroscope_orientation_margin": orientation["gyroscope"]["margin"],
                "gyroscope_orientation_scores": orientation["gyroscope"]["scores"],
                "needs_review": needs_review,
                **trim_debug,
            }
        )

    if write_debug and original_debug is not None:
        _ret, debug_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask_debug = Image.fromarray(debug_mask)
        outline_debug = make_debug_overlay(image_rgb, detections)
        before_orientation_debug = make_contact_sheet_from_arrays(before_orientation_images)
        final_debug = make_contact_sheet_image(oriented_paths)
        make_pipeline_debug_image(
            original_debug,
            mask_debug,
            outline_debug,
            before_orientation_debug,
            final_debug,
            debug_dir / f"{source_stem}_debug.png",
            debug_panel_width,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    review_count = sum(1 for detection in detections if detection["needs_review"])
    print(
        f"{input_path}: detected {len(detections)} photos -> {photos_dir} "
        f"({elapsed_ms:.1f} ms total, {orientation_elapsed_ms:.1f} ms orientation)"
    )
    for detection in detections:
        refine_state = "refined" if detection["refined"] else "rough"
        trims = (
            detection["trim_left"],
            detection["trim_top"],
            detection["trim_right"],
            detection["trim_bottom"],
        )
        print(
            f"{detection['filename']} {detection['trimmed_width']}x{detection['trimmed_height']} "
            f"angle={detection['angle']:.3f} {refine_state} trim_ltrb={trims} "
            f"dark_edge={detection['dark_edge_ratio_before']:.4f}->{detection['dark_edge_ratio_after']:.4f} "
            f"orient={detection['orientation_deg']} method={detection['orientation_method']} score={detection['orientation_score']:.4f} "
            f"margin={detection['orientation_margin']:.4f} faces={detection['face_count']} "
            f"review={detection['needs_review']}"
        )
    return ScanResult(
        input_path=input_path,
        photos_dir=photos_dir,
        debug_dir=debug_dir,
        source_stem=source_stem,
        detections=detections,
        oriented_paths=oriented_paths,
        photos=len(detections),
        needs_review=review_count,
        elapsed_ms=elapsed_ms,
        orientation_elapsed_ms=orientation_elapsed_ms,
    )
