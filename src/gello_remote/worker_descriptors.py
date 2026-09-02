"""Import-safe worker capabilities consumed by orchestration layers."""

from robot_data_contracts import WorkerDescriptorV1


GENESIS_WORKER = WorkerDescriptorV1(
    worker_id="gello_remote_genesis",
    engine_id="genesis",
    entrypoint="gello_remote.genesis_worker",
    control_modes=("keyboard", "gello"),
    capabilities={
        "recording": True,
        "automatic_outcome": True,
        "scene_reconfigure": True,
        "webrtc": False,
    },
)

ISAAC_SIM_WORKER = WorkerDescriptorV1(
    worker_id="gello_remote_isaac_sim",
    engine_id="isaac_sim",
    entrypoint="gello_remote.isaaclab_gello",
    control_modes=("gello",),
    capabilities={
        "recording": True,
        "automatic_outcome": False,
        "scene_reconfigure": False,
        "webrtc": True,
    },
)


WORKERS = {descriptor.engine_id: descriptor for descriptor in (GENESIS_WORKER, ISAAC_SIM_WORKER)}
