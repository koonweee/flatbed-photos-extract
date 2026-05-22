# Extractor

The extractor crops each detected photo to its paper border, straightens it,
trims dark edge residue, and applies automatic orientation correction.

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

1. Threshold the scan to separate photo paper from the dark background.
2. Find paper-like connected components.
3. Fit and refine each photo's outer border.
4. Perspective-warp each photo to a rectangle.
5. Trim remaining dark scanner edges.
6. Rotate using YuNet face detection or GyroScope fallback.
7. Write photos, metadata, and optional debug PNGs.
