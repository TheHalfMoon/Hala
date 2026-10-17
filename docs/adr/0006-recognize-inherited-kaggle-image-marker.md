# 0006. Recognize the inherited Kaggle image marker

Status: accepted (2026-10-16)

Supersedes [0005](0005-qualify-kaggle-evaluation-separately.md) only for its operational mixed-marker assumption; all other constraints remain in force.

## Context

The actual Kaggle Interactive runtime has `/kaggle/working` and `KAGGLE_KERNEL_RUN_TYPE=Interactive`, together with `COLAB_RELEASE_TAG=release-colab-external-images_20260917-060051_RC00`. A T4 is now live, with reported Python 3.13.15 and Torch 2.11/CUDA 12.8. The inherited Colab image tag caused the initial evaluation admission check to reject this genuine Kaggle runtime. Full dependency, font, model and matched evaluation qualification remain pending.

## Decision

Explicit Kaggle evaluation may retain a Colab image tag only when all existing Kaggle checks pass: explicit provider selection, remote-development opt-in, Linux, an allowlisted Interactive/Batch kernel marker and `/kaggle/working` directory. Record the image tag verbatim as `inherited_colab_release_tag` in provider provenance when present; never delete or spoof environment markers. Colab evaluation continues to reject Kaggle context. The training guard, frozen inference/data/settings and exact package gates remain unchanged.

## Consequences

+ Genuine Kaggle image inheritance no longer prevents bounded evaluation admission, and its provenance remains inspectable.
- Environment markers alone still do not qualify hardware, packages, model quality or execution authority. Unknown or incomplete Kaggle context remains rejected before ML imports; independent source gates and actual runtime qualification remain required.
