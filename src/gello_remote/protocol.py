import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class Observation(BaseModel):
    sequence: int = Field(ge=0)
    timestamp_ns: int = Field(gt=0)
    joint_positions: list[float] = Field(min_length=1, max_length=64)


class Action(BaseModel):
    sequence: int = Field(ge=0)
    joint_targets: list[float] = Field(min_length=1, max_length=64)
    policy: str

    @model_validator(mode="after")
    def finite_targets(self) -> "Action":
        if any(value != value or abs(value) == float("inf") for value in self.joint_targets):
            raise ValueError("joint targets must be finite")
        return self


class GelloCommandV1(BaseModel):
    """Versioned, calibrated 7-DoF GELLO command owned by gello-remote."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gello"] = "gello"
    arm: Literal["left", "right"] | None = None
    version: Literal[1] = 1
    session_id: str = Field(default="legacy", min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    timestamp_ns: int = Field(gt=0)
    joint_targets: list[float] = Field(min_length=7, max_length=7)
    gripper_closed: StrictBool

    @model_validator(mode="after")
    def validate_payload(self) -> "GelloCommandV1":
        if not self.session_id.strip():
            raise ValueError("session_id must not be blank")
        if not all(math.isfinite(value) for value in self.joint_targets):
            raise ValueError("joint_targets must contain exactly 7 finite values")
        return self


# Compatibility name used by the existing GELLO server/client code.
GelloCommand = GelloCommandV1
