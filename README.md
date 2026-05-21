# flatbed-photos-extract

Extract individual photos from one or more flatbed scan images. The script detects photo paper on a dark scanner background, crops each photo to its border, straightens it, applies a conservative dark-edge trim, and rotates it using a hybrid local orientation classifier.

## Install

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The YuNet face detector is included in `models/`. The GyroScope orientation model is loaded through Hugging Face Transformers and may need to be downloaded the first time it is used.

## Usage

```bash
python extract.py /path/to/scan-1.png /path/to/scan-2.png --output-dir output
```

Each command creates one batch folder under `output/`. By default the batch name is the current datetime. You can name it explicitly:

```bash
python extract.py /path/to/scan-1.png /path/to/scan-2.png --output-dir output --batch-name family-box-b
```

All photos from all input scans are written into the same batch:

```text
output/
  family-box-b/
    photos/
      scan-1_01.png
      scan-1_02.png
      scan-2_01.png
    metadata.csv
    debug/
      batch-contact-sheet.png
      batch-contact-sheet-before-orientation.png
      scan-1_original.png
      scan-1_mask.png
      scan-1_overlay.png
      scan-1_contact-sheet.png
      scan-1_contact-sheet-before-orientation.png
      scan-1_contact-sheet-after-orientation.png
      scan-1_original-mask-contact.png
      pre_orientation/
        scan-1_01.png
        scan-1_02.png
        scan-2_01.png
```

## Pipeline

1. Threshold the scan to separate light photo paper from the dark scanner background.
2. Find connected paper-like components and reject implausible regions.
3. Estimate each photo's outer border from contour geometry, then refine it by fitting lines to the detected paper edges.
4. Perspective-warp the detected border so the photo edges align with the output image edges.
5. Apply a conservative trim only where dark scanner background remains on the warped edges.
6. Classify orientation locally. YuNet is used when faces are detected; GyroScope is used as a general fallback. The best rotation is applied automatically.
7. Write final photos, metadata, and debug artifacts.

The final pipeline intentionally does not dewarp curled or non-flat photos. It only straightens the detected outer border and trims dark edge residue.

## Reading Outputs

`photos/` contains the processed images intended for import into a gallery or archive.

`metadata.csv` records source file names, per-scan photo indexes, source bounding boxes, fitted corners, output sizes, trim amounts, estimated scan rotation, orientation decisions, and classifier scores. `source_rotation_deg_clockwise_estimate` is the rotation of the photo on the scanner bed before rectification. `orientation_deg` is the rotation applied after extraction.

Per-scan debug files are prefixed with the scan stem. For example, `debug/scan-1_original.png` is the source scan copy, `debug/scan-1_mask.png` is the threshold mask used for candidate detection, and `debug/scan-1_overlay.png` draws rough and refined borders over the original scan. `debug/scan-1_original-mask-contact.png` puts that scan's original, mask, and final contact sheet side by side for quick review.

`debug/pre_orientation/` contains extracted photos after crop/straighten/trim but before orientation correction. `debug/batch-contact-sheet.png` shows all final photos in the batch, while each `debug/<scan>_contact-sheet.png` shows final photos from one source scan.
