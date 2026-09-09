"""Method identities implemented by the production experiment lifecycle."""

from typing import Final, Literal, TypeAlias

MethodId: TypeAlias = Literal[
    "erm", "grit", "groupdro", "rex", "irm", "fish", "lisa", "swad", "matchdg"
]

# This order is part of deterministic candidate planning and presentation.
IMPLEMENTED_METHODS: Final[tuple[MethodId, ...]] = (
    "erm",
    "grit",
    "groupdro",
    "rex",
    "irm",
    "fish",
    "lisa",
    "swad",
    "matchdg",
)

METHOD_LABELS: Final[dict[MethodId, str]] = {
    "erm": "ERM",
    "grit": "GRIT",
    "groupdro": "GroupDRO",
    "rex": "REx",
    "irm": "IRM",
    "fish": "Fish",
    "lisa": "LISA",
    "swad": "SWAD",
    "matchdg": "MatchDG",
}
