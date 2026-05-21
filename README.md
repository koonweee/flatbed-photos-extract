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

Each input scan gets its own folder under `output/`:

```text
output/
  scan-1/
    photos/
      01.png
      02.png
    metadata.csv
    debug/
      original.png
      mask.png
      overlay.png
      contact-sheet.png
      contact-sheet-before-orientation.png
      contact-sheet-after-orientation.png
      original-mask-contact.png
      pre_orientation/
        01.png
        02.png
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

`metadata.csv` records source bounding boxes, fitted corners, output sizes, trim amounts, estimated scan rotation, orientation decisions, and classifier scores. `source_rotation_deg_clockwise_estimate` is the rotation of the photo on the scanner bed before rectification. `orientation_deg` is the rotation applied after extraction.

`debug/original.png` is the source scan copy, `debug/mask.png` is the threshold mask used for candidate detection, and `debug/overlay.png` draws rough and refined borders over the original scan. `debug/original-mask-contact.png` puts the original, mask, and final contact sheet side by side for quick review.

`debug/pre_orientation/` and `debug/contact-sheet-before-orientation.png` show extracted photos after crop/straighten/trim but before orientation correction.
