"""H108 pure-pixel (not model weights) position-to-coordinate visual ruler.

Research analog of explicit position markers, NOT the published RULER-token
architecture or interleaved MRoPE. Source screenshot dimensions and H61
strict task boxes remain unchanged. No task answers appear in overlay.
"""
from __future__ import annotations
from PIL import Image,ImageDraw,ImageFont

GRID_STEP=64
LABEL_STEP=128
SCHEMA="hala.h108.visual-pixel-ruler-source-coordinate-v1"

def position_ruler(original:Image.Image)->Image.Image:
    """Create deterministic semi-transparent grid/ticks over existing image."""
    if original is None or not hasattr(original,"size"):
        raise ValueError("original screenshot required")
    width,height=original.size
    # H61 scales uniformly into a MAXIMUM 1024x640 canvas, preserving
    # each source's aspect ratio. Never assume every frame is exactly
    # 1024x640, since it would reject legitimate original source scenes.
    if not (64<=width<=1024 and 64<=height<=640):
        raise ValueError("H108 requires original H61 aspect-preserved bounded frame")
    base=original.convert("RGB").copy().convert("RGBA")
    layer=Image.new("RGBA",(width,height),(0,0,0,0))
    draw=ImageDraw.Draw(layer,"RGBA")
    for x in range(GRID_STEP,width,GRID_STEP):
        strong=(x%LABEL_STEP==0)
        draw.line([(x,0),(x,height-1)],
                  fill=(255,255,255,75 if strong else 36),width=1)
        if strong:
            draw.line([(x+1,0),(x+1,height-1)],fill=(0,0,0,45),width=1)
    for y in range(GRID_STEP,height,GRID_STEP):
        strong=(y%LABEL_STEP==0)
        draw.line([(0,y),(width-1,y)],fill=(255,255,255,75 if strong else 36),width=1)
        if strong:
            draw.line([(0,y+1),(width-1,y+1)],fill=(0,0,0,45),width=1)
    # Coordinate axes are fixed a priori, independent of action targets.
    font=ImageFont.load_default()
    for x in range(0,width,LABEL_STEP):
        label=f"x{x}"
        px=min(x+2,width-32)
        draw.rectangle((px,2,px+28,15),fill=(244,247,250,225))
        draw.text((px+2,3),label,font=font,fill=(0,0,0,255))
    for y in range(0,height,LABEL_STEP):
        label=f"y{y}"
        py=min(y+18,height-15)
        draw.rectangle((2,py,32,py+13),fill=(244,247,250,225))
        draw.text((3,py+1),label,font=font,fill=(0,0,0,255))
    return Image.alpha_composite(base,layer).convert("RGB")


def annotated_prompt(prompt:str,width:int,height:int)->str:
    """The ruler provides positions only; do not leak bbox or oracle action."""
    if not isinstance(prompt,str) or not prompt or len(prompt)>12000:
        raise ValueError("actual original H61 text prompt required")
    if (type(width) is not int or type(height) is not int
            or not 64<=width<=1024 or not 64<=height<=640):
        raise ValueError("actual H61 source dimensions required")
    return (prompt+"\nThe screenshot has faint coordinate gridlines every 64 pixels; "
            "labels x0,x128,... and y0,y128,... show pixel positions. "
            "Read the target objects in the original screenshot. "
            f"Use original image pixel coordinates ({width}x{height}), not grid-cell indexes. "
            "Output the same exact JSON schema.")
