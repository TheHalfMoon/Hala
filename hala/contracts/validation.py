"""Small strict parsing helpers; no coercion of untrusted values."""
from dataclasses import fields, is_dataclass
import json
import math


class ContractError(ValueError):
    """The untrusted payload violates a structural or contextual constraint."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def object_fields(value, required, optional=()):
    require(type(value) is dict, "Expected an object")
    require(set(value) <= set(required) | set(optional), "Unknown object fields")
    require(set(required) <= set(value), "Missing required fields")
    return value


def string(value, limit=128, empty=False):
    require(type(value) is str, "Expected a string")
    require((empty or bool(value.strip())) and len(value) <= limit, "String outside bounds")
    require("\x00" not in value, "NUL is not allowed")
    return value


def number(value, minimum, maximum, integer=False):
    require(type(value) in ((int,) if integer else (int, float)), "Expected a number, not boolean")
    require(minimum <= value <= maximum and math.isfinite(value), "Number outside finite bounds")
    return value


def boolean(value):
    require(type(value) is bool, "Expected a boolean")
    return value


def sequence(value, maximum, length=None):
    require(type(value) in (list, tuple), "Expected an array")
    require(len(value) <= maximum and (length is None or len(value) == length), "Array outside bounds")
    return tuple(value)


def frozen_sequence(value, maximum):
    require(type(value) is tuple, "Records require immutable tuples")
    return sequence(value, maximum)


def rect(value):
    result = sequence(value, 4, 4)
    number(result[0], 0, 8192)
    number(result[1], 0, 8192)
    number(result[2], 0.001, 8192)
    number(result[3], 0.001, 8192)
    return result


def in_frame(bounds, width, height):
    return bounds[0] + bounds[2] <= width and bounds[1] + bounds[3] <= height


def contains(bounds, point):
    return bounds[0] <= point[0] < bounds[0] + bounds[2] and bounds[1] <= point[1] < bounds[1] + bounds[3]


def overlaps(first, second):
    return first[0] < second[0] + second[2] and second[0] < first[0] + first[2] and first[1] < second[1] + second[3] and second[1] < first[1] + first[3]


def wire(value):
    if is_dataclass(value):
        return {field.name: wire(getattr(value, field.name)) for field in fields(value) if getattr(value, field.name) is not None}
    if type(value) is tuple:
        return [wire(item) for item in value]
    return value


class Record:
    """JSON representation of a validated immutable record."""

    __slots__ = ()

    def to_dict(self):
        return wire(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def parse_json(value):
    string(value, 262144)

    def unique_pairs(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, "Duplicate JSON field")
            result[key] = item
        return result

    def reject_constant(value):
        raise ContractError("Nonfinite JSON number")

    try:
        return json.loads(value, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, RecursionError, OverflowError) as error:
        raise ContractError("Invalid JSON payload") from error
