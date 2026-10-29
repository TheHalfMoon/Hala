# H217 — Exact Qwen-VL Processor Identity Gate

Base: H216 Draft #237 at 15bb4ced1e1398f83415377cda36808fd75f9169.

## Safety gap

H216 required an offline snapshot directory named after a revision and checked processor methods, but another processor class with those methods could be accepted as Qwen3-VL. Directory naming is not cryptographic evidence of file contents.

## Repair

The real offline AutoProcessor result must have the exact installed transformers.Qwen3VLProcessor type. Different classes and subclasses fail before image/token audit. Synthetic tests do not load transformers or weights.

This class-identity safeguard is not a complete immutable snapshot-byte hash receipt. Cache file-byte provenance remains unverified.

## Scope

No Mac, GPU, paid compute, weights, generation, training, merge, release, or publication. H212 96 research images remain unadmitted. H173 consumed OOD permanently sealed. Historical original source/fork and decoded-RGBA registry completeness remains unproven. CI tests are synthetic and do not replace prior owner Windows processor evidence.
