# GELLO Remote

Local GELLO control client with a GPU inference service running on the remote
8x RTX 5090 host. The first milestone deliberately uses a mock policy: validate
transport and safety boundaries before connecting real robot actuation.

## Remote server

```bash
cd /data/zhanghejun/gello-remote
python3 -m venv .venv
.venv/bin/pip install -e '.[server]'
.venv/bin/uvicorn gello_remote.server:app --host 127.0.0.1 --port 8000
```

## Local tunnel and smoke test

```bash
ssh -N -L 18000:127.0.0.1:8000 zhanghejun@219.223.200.145
curl http://127.0.0.1:18000/health
python -m gello_remote.client --url http://127.0.0.1:18000
```

The remote service returns suggestions only. Joint limits, velocity limits,
watchdogs, and emergency stop must remain on the robot-side machine.

## Simulation-first Franka backends

`MockFrankaBackend` is the default backend for tests and transport development.
Every backend call accepts exactly seven finite arm-joint targets; invalid input
is rejected before any backend state is changed.

```python
from gello_remote.backends import MockFrankaBackend

backend = MockFrankaBackend()
backend.set_joint_targets([0.0] * 7)
```

An optional bounded Isaac Lab smoke test uses the installed Franka asset and
the Isaac Lab 6.1.11 APIs. Isaac Lab imports are lazy, so ordinary imports and
unit tests do not require the simulator environment. Run it headlessly with:

```bash
env CONDA_PREFIX=/home/zhanghejun/miniconda3/envs/isaaclab \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/zhanghejun/gello-remote/src \
  /data/zhanghejun/IsaacLab/isaaclab.sh -p -m gello_remote.isaaclab_smoke \
  --steps 10 --device cuda:0 --viz none
```

The smoke process creates one simulated Franka, holds its default seven arm
joints for a bounded number of physics steps, and exits. It does not expose a
real-robot adapter, connect to robot hardware, or change the existing HTTP API.

## Direct GELLO control of Isaac Sim

The IsaacLab 3.0 / Isaac Sim 6.0.1 entry point consumes the same calibrated
`GelloCommand` used by the Genesis collector. It maps seven absolute joint
targets directly to `panda_joint1` through `panda_joint7`; it does not use IK.
The eighth GELLO motor selects the open (`0.04 m`) or closed (`0.00 m`) Panda
finger target.

The simulated arm uses Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`: shoulder and
forearm joint stiffness are `400`, damping is `80`, and articulation gravity is
disabled to represent a gravity-compensated position servo. This changes only
the simulated actuator dynamics; GELLO commands remain direct joint targets and
no IK is introduced.

Before opening the live server, run the bounded numeric smoke on an idle GPU:

The local IsaacLab checkout currently names the unpublished Isaac 6.0 asset
root. Keep that checkout untouched and explicitly select NVIDIA's published
5.1 Franka asset:

```bash
env CONDA_PREFIX=/home/zhanghejun/miniconda3/envs/isaaclab \
  PYTHONPATH=/data/zhanghejun/gello-remote/src \
  /data/zhanghejun/IsaacLab/isaaclab.sh -p -m gello_remote.isaaclab_gello \
  --gpu-id 0 --smoke_steps 240 \
  --franka-usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd
```

The smoke commands a `+0.03 rad` J1 move and closes the gripper, then stops
commands and checks that the 300 ms watchdog holds exactly once. It fails unless
motion direction, arm tracking, finger tracking, and target-hold gates pass.

Launch live control with:

```bash
env CONDA_PREFIX=/home/zhanghejun/miniconda3/envs/isaaclab \
  PYTHONPATH=/data/zhanghejun/gello-remote/src PUBLIC_IP=219.223.200.145 \
  /data/zhanghejun/IsaacLab/isaaclab.sh -p -m gello_remote.isaaclab_gello \
  --gpu-id 6 --enable_cameras --livestream 1 --rendering_mode balanced \
  --host 127.0.0.1 --port 18081 \
  --franka-usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd \
  --kit_args "--/UJITSO/geometry=false --/persistent/UJITSO/geometry=false --/app/window/width=1280 --/app/window/height=720"
```

Forward the control page to the Windows machine and open it in desktop Chrome
or Edge:

```bash
ssh -N -L 8080:127.0.0.1:18081 zhanghejun@219.223.200.145
```

Open `http://localhost:8080`. The single page contains the Isaac Sim WebRTC
video, stream health, GELLO controls, and simulator telemetry. Explicitly choose
COM3, capture the arm home and both gripper endpoints, set all seven joint
directions, and only then enable control. Set `--gpu-id N` to the physical GPU
index shown by `nvidia-smi`; the entry point keeps Isaac Lab, PhysX, the active
RTX GPU, and the single-GPU renderer subset aligned automatically. Do not also
pass `--device` or GPU-routing settings through `--kit_args`, and do not set
`CUDA_VISIBLE_DEVICES`: Isaac Sim 6.0.1 warns that CUDA and Vulkan enumerate
devices differently.

On the tested eight-RTX-5090 host, `--gpu-id` selects the requested GPU as the
only active RTX/PhysX workload device, but Isaac Sim 6.0.1 still creates an
approximately 512 MiB startup CUDA context on every enumerated GPU. Therefore
this option is coordinated device selection, not hard process isolation. If
other GPUs must show no context from this process, run Isaac Sim in an external
device-isolated environment; `CUDA_VISIBLE_DEVICES` is not used here because it
prevented the Vulkan renderer from creating a compatible device in this setup.

The live entry point validates `/World/Franka`, the ground plane, the light,
the active viewport, and its camera before reporting `sim_ready`. It also keeps
Kit's app loop pumping so WebRTC receives the populated scene instead of a
static default viewport. Geometry streaming is disabled because Isaac Sim 6.0.1
warns that combining it with Fabric transform reads can prevent dynamic
articulations from streaming correctly.

The server uses Isaac's articulation limits, a default `1.0 rad/s` target slew,
monotonic sequence rejection, and a 300 ms receive-time watchdog. Disabling the
browser control or disconnecting GELLO stops new messages; the simulator holds
the last safe arm and gripper targets.

With `--data-dir`, the live entry point also exposes episode controls in the
same browser console. `Start` creates one temporary episode, `Success` or
`Failure` atomically commits it, and `Discard` removes the temporary data. The
episode stores synchronized `base` and `wrist` RGB streams, simulator time, Franka joint
positions and velocities, commanded joint targets, policy state/action arrays,
and GELLO sequence/source timestamps using RoboData Studio's common episode
contract. Automatic task success is intentionally unavailable for the empty
Franka scene; only the operator outcome is written.

Pass `--camera-profile-file /absolute/path/to/profile.json` to select the same
sensor profile used by RoboData Studio. The lab profile models the active main
D435i (`261722075807`) as `base` and the active wrist D435i (`261622075579`) as
`wrist`; the historical third camera remains inventory-only. The wrist mount
uses the Franka visuomotor reference pose pinned to the clean local IsaacLab
checkout at `28a37cecdd433c22d9eabd6a5954add9f13a8951`. Until per-device 640x480 RGB
intrinsics are populated in the profile, both render cameras are explicitly
recorded as `nominal_pending_intrinsics` and use a nominal 42-degree vertical
field of view rather than claiming physical calibration.

This milestone validates the code structure and one simulated Franka in an
empty scene. It does not yet control cloth, validate real COM3 calibration, or
send any command to a physical robot. The recording path has unit/import
coverage, but no GPU recording episode has been collected in this change set.
