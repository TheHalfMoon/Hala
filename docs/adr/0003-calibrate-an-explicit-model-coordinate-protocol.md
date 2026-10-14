# ADR 0003: calibrate an explicit coordinate protocol before adaptation

Status: accepted for experiments only, 2026-10-14.

The first unchanged Qwen3-VL-2B FP16 run produced eight fenced JSON responses with coordinates inconsistent with the requested pixel space. The strict contract rejected all eight. This real 0/8 result remains immutable.

Before attributing the deficit to visual capability or training an adapter, run a separately labeled calibration with an explicit normalized 0..1000 protocol. Accept only a bare exact point object or one exact JSON fence, reject duplicate/nonfinite/extra fields, and deterministically convert both axes to the fixed 640x360 fixture. No automatic protocol guessing or execution authority is added. Pixel decoding stays strict by default.

Reusing these eight observed fixtures is calibration, not held-out qualification. This experiment cannot establish complete-task performance or justify release. Production contracts and the fail-closed real backend remain unchanged. A later held-out evaluation must validate any adopted decoder and prompt before model-quality or adapter-quality conclusions.
