"""H109 owner-only coarse-to-fine source pixel crop geometry.

No model inference or hidden TEST. Never edits original H61 labels/scorer.
This does NOT guarantee a coarse model point actually lies near its target.
"""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

PATCH=320
ZOOM=2


@dataclass(frozen=True)
class CropFrame:
    source_w:int
    source_h:int
    left:int
    top:int
    width:int
    height:int
    scale:int=ZOOM

    def __post_init__(self):
        fields=(self.source_w,self.source_h,self.left,self.top,
                self.width,self.height,self.scale)
        if (any(type(v) is not int for v in fields)
                or self.source_w<64 or self.source_h<64
                or self.source_w>1024 or self.source_h>640
                or self.width<1 or self.height<1
                or self.left<0 or self.top<0
                or self.left+self.width>self.source_w
                or self.top+self.height>self.source_h
                or self.scale!=ZOOM):
            raise ValueError("original source-bound image crop not qualified")

    def source_to_zoomed(self,point:list[int])->list[int]:
        if (type(point) is not list or len(point)!=2
                or any(type(n) is not int for n in point)
                or not self.left<=point[0]<self.left+self.width
                or not self.top<=point[1]<self.top+self.height):
            raise ValueError("original point outside candidate region")
        return [(point[0]-self.left)*self.scale,
                (point[1]-self.top)*self.scale]

    def zoomed_to_source(self,point:list[int])->list[int]:
        if (type(point) is not list or len(point)!=2
                or any(type(n) is not int for n in point)
                or not 0<=point[0]<self.width*self.scale
                or not 0<=point[1]<self.height*self.scale):
            raise ValueError("predicted crop point outside zoomed frame")
        return [self.left+point[0]//self.scale,
                self.top+point[1]//self.scale]

    def source_region_covers(self,bbox:list[int])->bool:
        if (type(bbox) is not list or len(bbox)!=4
                or any(type(x) is not int for x in bbox)):
            raise ValueError("bbox source coordinates required")
        return (self.left<=bbox[0]<=bbox[2]<self.left+self.width
                and self.top<=bbox[1]<=bbox[3]<self.top+self.height)


def crop_around_point(image:Image.Image,coarse:list[int],size:int=PATCH):
    if not hasattr(image,"size") or image.mode not in ("RGB","RGBA"):
        raise ValueError("bounded original image required")
    sw,sh=image.size
    if (type(size) is not int or size!=PATCH
            or type(coarse) is not list or len(coarse)!=2
            or any(type(v) is not int for v in coarse)
            or not (0<=coarse[0]<sw and 0<=coarse[1]<sh)):
        raise ValueError("real source coarse point needed; do not use reference GT")
    w,h=min(sw,size),min(sh,size)
    left=min(max(coarse[0]-w//2,0),sw-w)
    top=min(max(coarse[1]-h//2,0),sh-h)
    frame=CropFrame(sw,sh,left,top,w,h)
    clipped=image.crop((left,top,left+w,top+h))
    image_zoom=clipped.resize((w*ZOOM,h*ZOOM),resample=Image.Resampling.BICUBIC)
    return image_zoom,frame
