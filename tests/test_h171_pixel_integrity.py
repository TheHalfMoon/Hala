"""Exact H171 pixels, not PNG compressed encoding, are the replay identity."""
import io
from PIL import Image
from h171.image_integrity import same_exact_png_pixels

def png(color,compression):
    im=Image.new("RGBA",(960,720),color)
    buf=io.BytesIO()
    im.save(buf,format="PNG",compress_level=compression)
    return buf.getvalue()

def test_different_png_compression_same_pixels():
    a=png((240,70,10,255),0)
    b=png((240,70,10,255),9)
    assert a!=b
    assert same_exact_png_pixels(a,b)

def test_different_rgb_content_rejected():
    a=png((240,70,10,255),0)
    b=png((240,71,10,255),9)
    assert not same_exact_png_pixels(a,b)

def test_different_alpha_content_rejected():
    assert not same_exact_png_pixels(
        png((240,70,10,255),0),png((240,70,10,254),9))

def test_non_png_and_corrupt_rejected():
    assert not same_exact_png_pixels(b"not a png",png((1,1,1,255),9))
