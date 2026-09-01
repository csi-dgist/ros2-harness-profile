"""Profile-selected legacy trajectory fallback modules."""

from .direct_point import build as build_direct_point
from .two_stage import build as build_two_stage


BUILDERS = {
    "direct_point": build_direct_point,
    "two_stage": build_two_stage,
}


def build_fallback(name, node, experiment, latest_positions):
    try:
        builder = BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"unknown fallback module: {name}") from error
    return builder(node, experiment, latest_positions)


__all__ = ["BUILDERS", "build_fallback"]
