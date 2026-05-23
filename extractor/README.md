# Extractor

The extractor crops each detected photo or document-like item to its border,
straightens it, trims scanner-background residue, and applies automatic
orientation correction.

The bundled YuNet face detector lives in `extractor/models/`. The GyroScope
orientation model is loaded through Hugging Face Transformers on first use.

## Library API

```python
from extractor import process_scan

result = process_scan(
    input_path,
    photos_dir,
    debug_dir,
    source_stem,
    write_debug=True,
    debug_panel_width=None,
)
```

## CLI Output

```text
output/
  family-box-b/
    photos/
      scan-1_01.png
      scan-1_02.png
      scan-2_01.png
    metadata.csv
    debug/
      scan-1_debug.png
      scan-2_debug.png
```

`photos/` contains the extracted images. They are not intentionally scaled
down; each output is warped at the detected source-pixel border size, then
trimmed and rotated.

`metadata.csv` records the source scan, per-scan photo index, fitted corners,
trim amounts, output dimensions, and orientation scores.

Each debug PNG shows the pipeline for one scan from left to right:

```text
original -> mask -> outline -> before orientation -> final
```

## Pipeline

1. Sample the scan border to infer the high-contrast background polarity.
2. Threshold the scan to separate foreground items from the background.
3. Find photo-like and document-like connected components.
4. Fit and refine each item's outer border, using shadow-tolerant edge
   transitions for dark items on light backgrounds.
5. Perspective-warp each item to a rectangle.
6. Trim remaining scanner-background-colored edges.
7. Rotate using YuNet face detection or GyroScope fallback.
8. Write photos, metadata, and optional debug PNGs.
