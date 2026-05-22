from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.IMREAD_COLOR = 1
    cv2_stub.COLOR_BGR2RGB = 2
    cv2_stub.COLOR_BGR2GRAY = 3
    cv2_stub.FaceDetectorYN = object
    cv2_stub.imread = Mock(return_value=object())
    cv2_stub.cvtColor = Mock(side_effect=lambda image, _code: image)
    sys.modules["cv2"] = cv2_stub

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

try:
    import PIL  # noqa: F401
except ModuleNotFoundError:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    image_draw_stub = types.ModuleType("PIL.ImageDraw")
    pil_stub.Image = image_stub
    pil_stub.ImageDraw = image_draw_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub
    sys.modules["PIL.ImageDraw"] = image_draw_stub

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")

    def inference_mode():
        return lambda function: function

    torch_stub.inference_mode = inference_mode
    sys.modules["torch"] = torch_stub

try:
    import torchvision  # noqa: F401
except ModuleNotFoundError:
    torchvision_stub = types.ModuleType("torchvision")
    transforms_stub = types.ModuleType("torchvision.transforms")
    transforms_stub.Compose = Mock()
    transforms_stub.Resize = Mock()
    transforms_stub.CenterCrop = Mock()
    transforms_stub.ToTensor = Mock()
    transforms_stub.Normalize = Mock()
    torchvision_stub.transforms = transforms_stub
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = transforms_stub

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForImageClassification = Mock()
    sys.modules["transformers"] = transformers_stub

from extractor.batch import run_batch
from extractor import ScanResult, process_scan, write_metadata


def metadata_detection(filename: str) -> dict:
    return {
        "source_file": "scan.png",
        "source_stem": "scan",
        "source_photo_index": 1,
        "filename": filename,
        "bbox": (1, 2, 300, 400),
        "quad": [(1, 2), (301, 2), (301, 402), (1, 402)],
        "width": 300,
        "height": 400,
        "trimmed_width": 298,
        "trimmed_height": 398,
        "trim_left": 1,
        "trim_top": 1,
        "trim_right": 1,
        "trim_bottom": 1,
        "dark_edge_ratio_before": 0.2,
        "dark_edge_ratio_after": 0.0,
        "angle": 0.0,
        "orientation_deg": 0,
        "orientation_score": 1.0,
        "orientation_margin": 1.0,
        "face_count": 1,
        "orientation_method": "yunet",
        "needs_review": False,
        "orientation_scores": [{"rotation": 0, "score": 1.0, "face_count": 1}],
        "yunet_orientation_deg": 0,
        "yunet_orientation_score": 1.0,
        "yunet_orientation_margin": 1.0,
        "yunet_face_count": 1,
        "yunet_orientation_scores": [{"rotation": 0, "score": 1.0, "face_count": 1}],
        "gyroscope_orientation_deg": 0,
        "gyroscope_orientation_score": 1.0,
        "gyroscope_orientation_margin": 1.0,
        "gyroscope_orientation_scores": [{"rotation": 0, "score": 1.0}],
        "refined": True,
        "refine_reason": "",
        "area": 120_000,
    }


class ExtractorRefactorTests(unittest.TestCase):
    def test_run_batch_passes_each_input_stem_to_process_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = [root / "drybox-b-1.png", root / "drybox-b-2.png"]
            for input_path in inputs:
                input_path.touch()

            def fake_process_scan(input_path, photos_dir, debug_dir, source_stem, **_kwargs):
                return ScanResult(
                    input_path=Path(input_path),
                    photos_dir=Path(photos_dir),
                    debug_dir=Path(debug_dir),
                    source_stem=source_stem,
                    detections=[metadata_detection(f"{source_stem}_01.png")],
                    oriented_paths=[Path(photos_dir) / f"{source_stem}_01.png"],
                    photos=1,
                    needs_review=0,
                    elapsed_ms=1.0,
                    orientation_elapsed_ms=0.5,
                )

            with (
                patch("extractor.batch.make_detector", return_value=object()),
                patch("extractor.batch.GyroScopeClassifier", return_value=object()),
                patch("extractor.batch.process_scan", side_effect=fake_process_scan) as process_mock,
            ):
                results = run_batch(inputs, root / "output", "batch")

            self.assertEqual([result.source_stem for result in results], ["drybox-b-1", "drybox-b-2"])
            self.assertEqual(process_mock.call_count, 2)

    def test_process_scan_without_debug_does_not_create_debug_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "blank.png"
            photos_dir = root / "photos"
            debug_dir = root / "debug"
            input_path.touch()

            with (
                patch("extractor.core.rough_candidates", return_value=[]),
                patch("extractor.core.cv2.imread", return_value=object()),
                patch("extractor.core.cv2.cvtColor", side_effect=lambda image, _code: image),
            ):
                result = process_scan(
                    input_path,
                    photos_dir,
                    debug_dir,
                    "blank",
                    write_debug=False,
                    detector=Mock(),
                    gyroscope=Mock(),
                )

            self.assertEqual(result.photos, 0)
            self.assertTrue(photos_dir.exists())
            self.assertFalse(debug_dir.exists())
            self.assertEqual(list(root.glob("debug/*.png")), [])

    def test_metadata_rows_equal_detections_plus_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.csv"
            detections = [metadata_detection("scan_01.png"), metadata_detection("scan_02.png")]

            write_metadata(metadata_path, detections)

            with metadata_path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), len(detections) + 1)


if __name__ == "__main__":
    unittest.main()
