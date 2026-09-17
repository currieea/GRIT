"""Method identities implemented by the production experiment lifecycle."""

from typing import Final, Literal, TypeAlias

MethodId: TypeAlias = Literal[
    "erm",
    "grit",
    "groupdro",
    "rex",
    "irm",
    "fish",
    "lisa",
    "swad",
    "matchdg",
    "sd",
    "fishr",
    "rdm",
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
    "sd",
    "fishr",
    "rdm",
)

# SD, Fishr, and RDM are bound to the trainer for CMNIST only. The other datasets'
# search configurations reject them rather than planning candidates no runner binds.
CMNIST_ONLY_METHODS: Final[tuple[MethodId, ...]] = ("sd", "fishr", "rdm")

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
    "sd": "SD",
    "fishr": "Fishr",
    "rdm": "RDM",
}

# Every method the Waterbirds runner binds; the CMNIST-only additions are deferred.
WATERBIRDS_METHODS: Final[tuple[MethodId, ...]] = tuple(
    method for method in IMPLEMENTED_METHODS if method not in CMNIST_ONLY_METHODS
)
