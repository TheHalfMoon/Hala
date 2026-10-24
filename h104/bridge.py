"""H104 explicit optional source-pixel <-> processor-pixel coordinate bridge.

No mutation of H71/old H61 prompts/scorer; source and processor frames
are kept separately. Pure CPU, suitable for owner-private oracle-only audit.
"""
from __future__ import annotations
from dataclasses import dataclass
import json

from h61.official_eval import score


def _pixel_map(coord:int,source:int,destination:int)->int:
    if (type(coord) is not int or type(source) is not int or type(destination) is not int
            or source < 1 or destination < 1 or coord < 0 or coord >= source):
        raise ValueError("pixel index outside source/target frame contract")
    # Map pixel CENTERS, not pixel edges, when processor changes grid width.
    x=(coord+0.5)*destination/source-0.5
    return min(destination-1,max(0,int(round(x))))


@dataclass(frozen=True)
class FrameBridge:
    source_width:int
    source_height:int
    processor_width:int
    processor_height:int

    def __post_init__(self):
        if any(type(v) is not int or v<1 or v>32768
               for v in (self.source_width,self.source_height,
                         self.processor_width,self.processor_height)):
            raise ValueError("original and processor frame dimensions required")

    def to_processor(self,point:list[int])->list[int]:
        if type(point) is not list or len(point)!=2:
            raise ValueError("source point must be two integer pixel indexes")
        return [_pixel_map(point[0],self.source_width,self.processor_width),
                _pixel_map(point[1],self.source_height,self.processor_height)]

    def to_source(self,point:list[int])->list[int]:
        if type(point) is not list or len(point)!=2:
            raise ValueError("processed point must be two integer pixel indexes")
        return [_pixel_map(point[0],self.processor_width,self.source_width),
                _pixel_map(point[1],self.processor_height,self.source_height)]

    def source_bbox_to_processor(self,box:list[int])->list[int]:
        from h102.oracle import in_bounds
        if not in_bounds(box,self.source_width,self.source_height):
            raise ValueError("source bbox must use strict source pixel indices")
        start=self.to_processor(box[:2])
        end=self.to_processor(box[2:])
        return start+end


def roundtrip_oracle(scene:dict,bridge:FrameBridge)->dict:
    """Score source->processed->source GT, unchanged original H61 evaluator."""
    from h102.oracle import in_bounds,center
    if (scene["image_width"]!=bridge.source_width
            or scene["image_height"]!=bridge.source_height):
        raise ValueError("source scene/frame mismatch")
    for kind in ("start_bbox","end_bbox"):
        if not in_bounds(scene["task"][kind],bridge.source_width,bridge.source_height):
            raise ValueError("original source bbox outside actual image frame")
    original_start=center(scene["task"]["start_bbox"])
    original_end=center(scene["task"]["end_bbox"])
    p_start=bridge.to_processor(original_start)
    p_end=bridge.to_processor(original_end)
    recovered_start=bridge.to_source(p_start)
    recovered_end=bridge.to_source(p_end)
    if p_start==p_end or recovered_start==recovered_end:
        raise ValueError("processor mapping collapses non-degenerate source drag")
    action={"type":"drag","start_point":recovered_start,
            "end_point":recovered_end,"coordinate_space":"source_image_pixels",
            "ordered":scene["task"]["ordered"]}
    outcome=score([scene],{scene["id"]:json.dumps(action,separators=(",",":"))})
    if outcome["action_valid"]!=1 or outcome["task_success"]!=1:
        raise ValueError("mapping cannot preserve original strict source geometry")
    delta=max(abs(a-b) for orig,new in
              ((original_start,recovered_start),(original_end,recovered_end))
              for a,b in zip(orig,new))
    return {"valid_source_geometry":True,"max_pixel_roundtrip_error":delta,
            "processor_frame_matches_source":(
                bridge.source_width==bridge.processor_width and
                bridge.source_height==bridge.processor_height)}
