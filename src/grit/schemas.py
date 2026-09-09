"""Strict, canonical boundary-schema primitives for the rewrite."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, ConfigDict


class CmnistSelector(str, Enum):
    """The two prespecified ordinary selectors plus the labeled test-oracle track."""

    PRIMARY_ROBUST = "primary_robust"
    SECONDARY_SOURCE = "secondary_source"
    TEST_ORACLE = "test_oracle"

    @property
    def is_ordinary(self) -> bool:
        return self is not CmnistSelector.TEST_ORACLE


ORDINARY_CMNIST_SELECTORS = (
    CmnistSelector.PRIMARY_ROBUST,
    CmnistSelector.SECONDARY_SOURCE,
)


class SeedStage(str, Enum):
    """The model-selection stage associated with a training seed."""

    TUNING = "tuning"
    CONFIRMATION = "confirmation"
    FINAL = "final"


class StrictBoundaryModel(BaseModel):
    """Base for immutable, strict, canonical JSON boundary models."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        """Return the authoritative compact JSON representation."""

        return canonical_json_value(self.model_dump(mode="json"))

    def canonical_digest(self) -> str:
        """Return a version-explicit digest of the canonical JSON bytes."""

        payload = self.canonical_json().encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_json_value(value: object) -> str:
    """Serialize an already JSON-compatible value using canonical settings."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_digest_value(value: object) -> str:
    """Digest an already JSON-compatible value using the canonical representation."""

    payload = canonical_json_value(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
