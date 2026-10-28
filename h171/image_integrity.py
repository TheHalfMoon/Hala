"""Original-browser exact visible pixel replay independent of PNG encoding.

Chromium may encode an otherwise identical RGBA raster with different PNG
compression; only the permanently frozen source file's PNG SHA is authoritative.
"""
import io
from PIL import Image

def same_exact_png_pixels(frozen_bytes,replayed_bytes):
    try:
        with Image.open(io.BytesIO(frozen_bytes)) as a:
            with Image.open(io.BytesIO(replayed_bytes)) as b:
                if (a.format!="PNG" or b.format!="PNG" or
                    a.size!=(960,720) or b.size!=(960,720)):
                    return False
                a.load()
                b.load()
                return a.convert("RGBA").tobytes()==b.convert("RGBA").tobytes()
    except (OSError,ValueError):
        return False
