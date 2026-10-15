"""Processor-independent supervision boundaries for one example at a time."""


def completion_labels(full_ids, prefix_ids, token_budget):
    if (type(token_budget) is not int or token_budget <= 0 or
            not prefix_ids or len(prefix_ids) >= len(full_ids) or
            len(full_ids) > token_budget or full_ids[:len(prefix_ids)] != prefix_ids or
            any(type(token) is not int or token < 0 for token in full_ids)):
        raise ValueError('Processor completion boundary/token budget mismatch')
    return [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids):])
