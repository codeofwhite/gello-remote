import os
import socket

from fastapi import FastAPI, HTTPException

from .protocol import Action, Observation

app = FastAPI(title="GELLO Remote Inference", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "all")
    return {"status": "ok", "host": socket.gethostname(), "cuda_visible_devices": visible}


@app.post("/v1/action", response_model=Action)
def action(observation: Observation) -> Action:
    if not observation.joint_positions:
        raise HTTPException(status_code=422, detail="joint_positions cannot be empty")
    # Transport-only milestone: echo positions, never produce autonomous motion.
    return Action(
        sequence=observation.sequence,
        joint_targets=observation.joint_positions,
        policy="mock-hold-position",
    )
