"""H61 coordinate-focused supervision for the native drag curriculum.

The measured cause of the H60 collapse
--------------------------------------
The H60 adapter, trained for 60 real optimizer steps over 1,586 rows, learned
the JSON output contract (strict bare-JSON validity 0 -> 17 of 48) and then
collapsed onto a constant degenerate point: geometrically correct stayed at 0
of 24 on both public DEV sets and real GUI drag task success stayed at 0 of 48.

The completion is a JSON object whose tokens are dominated by the deterministic
structural keys ("type", "coordinate_space", "ordered"), while the tokens that
carry the actual grounding are the coordinate digits. A plain token-mean cross
entropy therefore spends most of its gradient on tokens that are already
predictable and only a small part on the coordinates, and the cheapest way to
reduce the mean loss is to predict a plausible constant coordinate instead of
the correct one. That is precisely the observed failure, and it is a
supervision-density problem rather than a capacity problem.

This module makes the density explicit. Every supervised completion token
keeps weight 1, and every token that carries a digit - the coordinate tokens -
receives a configurable weight. The weighted loss is accumulated in bounded
chunks, so the extra fp32 logit copy never has to exist for the whole
sequence at once.
"""
from __future__ import annotations

DIGIT_WEIGHT = 8.0
STRUCTURAL_WEIGHT = 1.0
MAX_COORDINATE_WEIGHT = 64.0
MIN_COORDINATE_WEIGHT = 1.0
CHUNK = 8192


def coordinate_token_flags(token_ids, decode) -> list[bool]:
    """Mark the completion tokens that carry a coordinate digit.

    `decode` maps one token id list to its text. A token is a coordinate token
    when its own text contains a decimal digit. The original native drag
    completion is a JSON object whose only digits are the original coordinate
    values, so this rule selects exactly the grounding tokens without needing
    to know how the tokenizer splits them.
    """
    flags: list[bool] = []
    for token in token_ids:
        if type(token) is not int:
            raise ValueError("Token ids must be integers")
        text = decode([token])
        if not isinstance(text, str):
            raise ValueError("Token decoder must return text")
        flags.append(any(character.isdigit() for character in text))
    return flags


def completion_weights(token_ids, decode, coordinate_weight: float = DIGIT_WEIGHT) -> list[float]:
    """Per-token supervision weights for one completion span.

    The weights are returned for the completion tokens only. The prompt prefix
    is masked by the label boundary and never receives a weight.
    """
    if (type(coordinate_weight) not in (int, float) or
            not MIN_COORDINATE_WEIGHT <= coordinate_weight <= MAX_COORDINATE_WEIGHT):
        raise ValueError("Bounded coordinate supervision weight required")
    weights: list[float] = []
    for flag in coordinate_token_flags(token_ids, decode):
        weights.append(float(coordinate_weight) if flag else STRUCTURAL_WEIGHT)
    if not any(flag for flag in coordinate_token_flags(token_ids, decode)):
        raise ValueError("Completion carries no coordinate token to supervise")
    return weights


def weighted_lm_loss(logits, labels, weights, chunk_size: int = CHUNK):
    """Weighted next-token cross entropy, accumulated in bounded chunks.

    `logits` is [batch, sequence, vocabulary], `labels` is [batch, sequence]
    with -100 on every unsupervised position and `weights` is [batch,
    sequence] with 0.0 on every unsupervised position.

    The shifted targets are the label positions 1..n-1 and the shifted logits
    are the positions 0..n-2, exactly like the model's own loss, so the only
    difference from the plain token mean is the per-token weight.
    """
    import torch

    if logits.dim() != 3 or labels.dim() != 2 or weights.dim() != 2:
        raise ValueError("Expected [batch, sequence, vocabulary] logits")
    if labels.shape != weights.shape or labels.shape[0] != logits.shape[0]:
        raise ValueError("Label and weight shapes must match the logit batch")
    if labels.shape[1] < 2:
        raise ValueError("A supervised span needs at least two positions")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_weights = weights[:, 1:]
    supervised = shifted_labels != -100
    if not bool(supervised.any()):
        raise ValueError("No supervised token in the completion span")
    if not torch.all(shifted_weights[supervised] > 0):
        raise ValueError("Every supervised token needs a positive weight")
    if bool((shifted_weights[~supervised] != 0).any()):
        raise ValueError("Every unsupervised token needs a zero weight")
    denominator = shifted_weights[supervised].sum()
    if float(denominator) <= 0:
        raise ValueError("Positive supervision weight is required")
    flat_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1])
    flat_labels = shifted_labels.reshape(-1)
    flat_weights = shifted_weights.reshape(-1)
    flat_supervised = supervised.reshape(-1)
    total = flat_logits.new_zeros((), dtype=torch.float32)
    selected_labels = flat_labels[flat_supervised]
    selected_weights = flat_weights[flat_supervised]
    selected_logits = flat_logits[flat_supervised]
    for start in range(0, selected_logits.shape[0], chunk_size):
        stop = start + chunk_size
        chunk_logits = selected_logits[start:stop].float()
        chunk_labels = selected_labels[start:stop]
        chunk_weights = selected_weights[start:stop]
        losses = torch.nn.functional.cross_entropy(
            chunk_logits, chunk_labels, reduction="none"
        )
        total = total + (losses * chunk_weights).sum()
    return total / denominator
