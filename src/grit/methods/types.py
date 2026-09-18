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
    "erm_representation_consistency",
    "erm_two_layer",
    "rex_representation_consistency",
    "rex_two_layer",
    "irm_representation_consistency",
    "irm_two_layer",
    "fishr_representation_consistency",
    "fishr_two_layer",
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
    "erm_representation_consistency",
    "erm_two_layer",
    "rex_representation_consistency",
    "rex_two_layer",
    "irm_representation_consistency",
    "irm_two_layer",
    "fishr_representation_consistency",
    "fishr_two_layer",
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
    "erm_representation_consistency": "ERM + representation consistency",
    "erm_two_layer": "ERM + two-layer control",
    "rex_representation_consistency": "V-REx + representation consistency",
    "rex_two_layer": "V-REx + two-layer control",
    "irm_representation_consistency": "IRMv1 + representation consistency",
    "irm_two_layer": "IRMv1 + two-layer control",
    "fishr_representation_consistency": "Fishr + representation consistency",
    "fishr_two_layer": "Fishr + two-layer control",
}

# Every method the Waterbirds runner binds to the trainer. RotatedMNIST stays limited
# to ERM and GRIT, which its own search configuration enforces.
WATERBIRDS_METHODS: Final[tuple[MethodId, ...]] = IMPLEMENTED_METHODS


def base_objective(method: MethodId) -> MethodId:
    """Legacy GRIT is ERM with projection; combination IDs are presentation keys."""
    if method == "grit":
        return "erm"
    return cast(MethodId, method.split("_")[0])


PairIntervention: TypeAlias = Literal[
    "vanilla", "grit", "consistency", "representation_consistency", "two_layer"
]


def pair_intervention(method: MethodId) -> PairIntervention:
    if method == "grit" or method.endswith("_grit"):
        return "grit"
    if method.endswith("_representation_consistency"):
        return "representation_consistency"
    if method.endswith("_two_layer"):
        return "two_layer"
    if method.endswith("_consistency"):
        return "consistency"
    return "vanilla"


def consumes_pairs(method: MethodId) -> bool:
    return method == "matchdg" or pair_intervention(method) in (
        "grit",
        "consistency",
        "representation_consistency",
    )
