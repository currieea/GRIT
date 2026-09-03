"""Method identities implemented by the production experiment lifecycle."""

from typing import Final, Literal, TypeAlias

MethodId: TypeAlias = Literal["erm", "grit"]

# This order is part of deterministic candidate planning and presentation.
IMPLEMENTED_METHODS: Final[tuple[MethodId, ...]] = ("erm", "grit")
