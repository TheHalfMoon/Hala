"""Recover original H71 TRAIN screenshot/box geometry from verified shards.

The H61 *SFT export* intentionally contains prompt, completion and image
hash, but does NOT carry original END bbox or image width/height. These
sensitive TRAIN target fields must be read from original source shards,
never guessed from an SFT row or copied from official EVAL/TEST.

No new dataset, model, paid compute, benchmark or modification to H71.
"""
from pathlib import Path
from h61.curriculum import read_shard,shard_roots
from h105.curriculum import source_geometry


def admit_original_source_map(training_root:Path):
    root=Path(training_root).resolve()
    if not root.is_dir():
        raise ValueError("H122 original H71 TRAIN export root must exist")
    source={}
    for shard in shard_roots(root):
        prefix=shard.relative_to(root).as_posix() if shard!=root else ""
        original_rows,_=read_shard(shard,prefix)
        for row in original_rows:
            ident=row.get("id")
            if (type(ident) is not str or not ident
                    or ident in source or row.get("split")!="train"
                    or row.get("action_type")!="drag"):
                raise ValueError("H122 unique authentic original TRAIN source required")
            source[ident]=row
    if len(source)<10000:
        raise ValueError("H122 original 10k DragOn TRAIN-source admission incomplete")
    return source


def original_for_sft(sft_row,original_source_map):
    """Return true original source geometry only after SFT/label identity."""
    if (type(sft_row) is not dict or sft_row.get("split")!="train"
            or not isinstance(original_source_map,dict)
            or sft_row.get("id") not in original_source_map):
        raise ValueError("H122 original H71 TRAIN SFT source absent")
    original=original_source_map[sft_row["id"]]
    if (sft_row.get("language")!=original.get("language")
            or sft_row.get("family")!=original.get("family")
            or sft_row.get("image_sha256")!=original.get("image_sha256")
            or sft_row.get("image_path")!=original.get("image_path")):
        raise ValueError("H122 SFT image/lang/family does not match original source")
    # H105 original geometry validator checks exact original action,
    # original source END bbox, ordered field and TRAIN split.
    if source_geometry(sft_row,original) is None:
        raise ValueError("H122 original source geometry was excluded by H105")
    return original
