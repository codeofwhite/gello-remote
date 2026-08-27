from pydantic import BaseModel, Field, model_validator


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
