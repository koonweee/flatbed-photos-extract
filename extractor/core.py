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
ROTATIONS = (0, 90, 180, 270)
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


def refine_quad_with_outer_edges(
    gray: np.ndarray,
    rough_quad: np.ndarray,
    threshold: int,
    padding: int,
) -> tuple[np.ndarray, dict]:
    x, y, w, h = cv2.boundingRect(rough_quad.astype(np.int32))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(gray.shape[1], x + w + padding)
    y1 = min(gray.shape[0], y + h + padding)

    local_gray = gray[y0:y1, x0:x1]
    _ret, local_mask = cv2.threshold(local_gray, threshold, 255, cv2.THRESH_BINARY)
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
        return rough_quad, {"refined": False, "reason": "insufficient edge points"}

    local_corners = [
        intersect_lines(lines["top"], lines["left"]),
        intersect_lines(lines["top"], lines["right"]),
        intersect_lines(lines["bottom"], lines["right"]),
        intersect_lines(lines["bottom"], lines["left"]),
    ]
    if any(corner is None for corner in local_corners):
        return rough_quad, {"refined": False, "reason": "parallel edge lines"}

    local_quad = order_points(np.array(local_corners, dtype=np.float32))
    global_quad = local_quad + np.array([x0, y0], dtype=np.float32)

    rough_area = cv2.contourArea(rough_quad.astype(np.float32))
    refined_area = cv2.contourArea(global_quad.astype(np.float32))
    if refined_area < rough_area * 0.85 or refined_area > rough_area * 1.2:
        return rough_quad, {"refined": False, "reason": "area sanity check failed"}

    debug = {
        "refined": True,
        "local_origin": (x0, y0),
        "edge_point_counts": {side: len(points) for side, points in edge_points.items()},
    }
    return global_quad, debug


def rough_candidates(image_bgr: np.ndarray, threshold: int, min_area: int) -> list[dict]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _ret, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > image_area * 0.2:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_scan_edge = x <= 5 or y <= 5 or x + w >= image_bgr.shape[1] - 5 or y + h >= image_bgr.shape[0] - 5
        if touches_scan_edge or w < 250 or h < 250:
            continue

        rough_quad = find_quadrilateral(contour)
        width, height = rectified_size(rough_quad)
        aspect = max(width / height, height / width)
        if aspect > 2.0:
            continue

        candidates.append({"bbox": (x, y, w, h), "area": area, "rough_quad": rough_quad})

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


def trim_dark_edges(
    image_bgr: np.ndarray,
    dark_threshold: int,
    dark_fraction: float,
    max_trim_px: int,
    max_trim_fraction: float,
    edge_ratio_band: int,
) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    max_trim = max(0, min(max_trim_px, int(round(min(width, height) * max_trim_fraction))))
    before_ratio = dark_edge_ratio(image_bgr, dark_threshold, edge_ratio_band)
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
        quad, debug = refine_quad_with_outer_edges(gray, candidate["rough_quad"], threshold, padding)
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
