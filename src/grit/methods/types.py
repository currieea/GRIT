"""Method identities implemented by the production experiment lifecycle."""

from typing import Final, Literal, TypeAlias, cast

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
    "erm_consistency",
    "rex_grit",
    "rex_consistency",
    "irm_grit",
    "irm_consistency",
    "fishr_grit",
    "fishr_consistency",
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
    "erm_consistency",
    "rex_grit",
    "rex_consistency",
    "irm_grit",
    "irm_consistency",
    "fishr_grit",
    "fishr_consistency",
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
    "matchdg": "MatchDG-style representation consistency",
    "sd": "SD",
    "fishr": "Fishr",
    "rdm": "RDM",
    "erm_consistency": "ERM + prediction consistency",
    "rex_grit": "V-REx + GRIT",
    "rex_consistency": "V-REx + prediction consistency",
    "irm_grit": "IRMv1 + GRIT",
    "irm_consistency": "IRMv1 + prediction consistency",
    "fishr_grit": "Fishr + GRIT",
    "fishr_consistency": "Fishr + prediction consistency",
}

# Every method the Waterbirds runner binds to the trainer. RotatedMNIST stays limited
# to ERM and GRIT, which its own search configuration enforces.
WATERBIRDS_METHODS: Final[tuple[MethodId, ...]] = IMPLEMENTED_METHODS


def base_objective(method: MethodId) -> MethodId:
    """Legacy GRIT is ERM with projection; combination IDs are presentation keys."""
    if method == "grit":
        return "erm"
    return cast(MethodId, method.split("_")[0])


def pair_intervention(method: MethodId) -> Literal["vanilla", "grit", "consistency"]:
    if method == "grit" or method.endswith("_grit"):
        return "grit"
    if method.endswith("_consistency"):
        return "consistency"
    return "vanilla"


def consumes_pairs(method: MethodId) -> bool:
    return method == "matchdg" or pair_intervention(method) != "vanilla"
