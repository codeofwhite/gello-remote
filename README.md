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

## Genesis worker

The shared Genesis browser collector is launched from this repository and
loads ClothTwin's scene, recording, and dataset implementation from its plugin
checkout. Robot Data Studio launches `gello_remote.genesis_worker` for both
keyboard and GELLO control; the historical `franka_only.*_collect` modules
remain compatibility entry points.

`gello_remote.genesis_policy_rollout` is the non-interactive policy-evaluation
worker. It connects to an OpenPI-compatible websocket server without importing
Robot Data Studio, converts joint7 plus gripper-closed-fraction outputs to the
physical Franka joint8 contract, and retains the scene's joint-limit and target
slew filters. Each bounded run records the canonical requested/safe/applied
actions, inference timing, four RGB streams, native evaluator result, and four
MP4 videos. A rollout succeeds only after close, release, native task success,
and the requested number of stable target frames.

## Direct GELLO control of Isaac Sim

The IsaacLab 3.0 / Isaac Sim 6.0.1 entry point consumes the same calibrated
`GelloCommand` used by the Genesis collector. It maps seven absolute joint
targets directly to `panda_joint1` through `panda_joint7`; it does not use IK.
The eighth GELLO motor selects the open (`0.04 m`) or closed (`0.00 m`) Panda
finger target.

The same entry point also supports a bounded, non-interactive OpenPI evaluation
mode. `--control-source openpi` connects to the websocket policy server, runs
all requested episodes in one Isaac process, and writes canonical episodes,
four MP4 files per rollout, `rollouts.csv`, and `batch_report.json` under the
selected data directory. The fixed `rigid_cube_lift` evaluator reports grasp,
lift, hold, and release separately; grasp requires the closed gripper to move
the cube at least 2 cm above its initial height.

```bash
env CONDA_PREFIX=/home/zhanghejun/miniconda3/envs/isaaclab \
  PYTHONPATH=/data/zhanghejun/gello-remote/src:/data/zhanghejun/openpi/packages/openpi-client/src:/data/zhanghejun/robot-data-studio/packages/robot-data-contracts/src \
  /data/zhanghejun/IsaacLab/isaaclab.sh -p -m gello_remote.isaaclab_gello \
  --control-source openpi --scene rigid_cube_lift --gpu-id 6 \
  --enable_cameras --rollout-count 20 --max-episode-frames 400 \
  --policy-checkpoint /absolute/path/to/checkpoint \
  --data-dir /absolute/path/to/rollouts
```

For a paired gripper-discretization diagnostic, keep the checkpoint, seed,
scene, cameras, horizon, and policy RNG restart fixed. The legacy arm uses
`--gripper-filter single_threshold --gripper-close-threshold 0.5`. The
hysteresis arm uses `--gripper-filter hysteresis
--gripper-close-threshold 0.65 --gripper-open-threshold 0.35
--gripper-min-closed-seconds 0.75`. Add `--full-horizon` to both arms so every
rollout records the same number of policy calls. Reports retain both the raw
model threshold crossings and the filtered binary gripper commands.

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
  --scene rigid_pick_place --work-surface-height-m 0.25 \
  --host 127.0.0.1 --port 18081 \
  --franka-usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd \
  --kit_args "--/UJITSO/geometry=false --/persistent/UJITSO/geometry=false --/app/window/width=960 --/app/window/height=720"
```

Forward the control page to the Windows machine and open it in desktop Chrome
or Edge:

```bash
ssh -N -L 8080:127.0.0.1:18081 zhanghejun@219.223.200.145
```

Open `http://localhost:8080`. The single page contains the Isaac Sim WebRTC
video, four synchronized camera previews, stream health, GELLO controls, and
simulator telemetry. Click `base`, `top`, `side`, or `wrist` (or press `1`–`4`)
to promote that camera to the high-frame-rate WebRTC viewport; press `0` to
restore the free operator view. The WebRTC viewport defaults to 20 FPS via
`--stream-fps`, leaving render capacity for the four 10 Hz dataset cameras.
Explicitly choose COM3, place GELLO near the middle of its mechanical travel,
capture the arm home and both gripper endpoints, set all seven joint
directions, and only then enable control. Set `--gpu-id N` to the physical GPU
index shown by `nvidia-smi`. The entry point exposes only that physical device
through `CUDA_VISIBLE_DEVICES`, routes Isaac Lab and PhysX to the resulting
logical `cuda:0`, and keeps RTX/Vulkan on physical GPU `N`. It also disables
multi-GPU rendering. `--gpu-id` takes precedence over an inherited
`CUDA_VISIBLE_DEVICES`; do not also pass `--device` or GPU-routing settings
through `--kit_args`.

This split mapping is required because CUDA renumbers a one-device visibility
mask to logical device 0 while Kit's RTX `activeGpu` follows the physical index
shown by `nvidia-smi`. On a shared host, verify after startup that the Isaac Sim
PID appears only on the selected GPU.

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
episode stores synchronized `base`, `top`, `side`, and `wrist` RGB streams,
simulator time, Franka joint
positions and velocities, commanded joint targets, policy state/action arrays,
and GELLO sequence/source timestamps using RoboData Studio's common episode
contract. Camera previews default to 10 Hz and use a bounded latest-frame
encoder, while Episode frames are never dropped. Episode cadence defaults to
the camera profile's `dataset_fps_default`; `--preview-fps` and `--dataset-fps`
may override these values when both divide `--control_hz` exactly.
`franka_empty` remains available for control diagnostics. The
`rigid_pick_place` scene adds a resettable 5 cm cube, target marker, and a work
surface whose top is `0.25 m` above the Franka base by default. Tune it within
`0.18--0.32 m` using `--work-surface-height-m`. Automatic task success is not
available for this placement task, so only the operator outcome is written.

For the first Isaac-to-OpenPI pilot, select `--scene rigid_cube_lift`. This scene
keeps the same fixed table and green 5 cm cube but removes the placement target.
Each demonstration must close the gripper, lift the cube at least 8 cm, hold it
above that threshold for one second, and then release. The worker records end-
effector pose, object pose, and object velocity, and stores an independent
`isaac_cube_lift_v1` evaluator result alongside the operator outcome.

Pass `--camera-profile-file /absolute/path/to/profile.json` to select the same
sensor profile used by RoboData Studio. The lab profile models the active main
D435i (`261722075807`) as `base` and the active wrist D435i (`261622075579`) as
`wrist`; the historical third camera remains inventory-only. The wrist mount
uses the Franka visuomotor reference pose pinned to the clean local IsaacLab
checkout at `28a37cecdd433c22d9eabd6a5954add9f13a8951`. Until per-device 640x480 RGB
intrinsics are populated in the profile, both render cameras are explicitly
recorded as `nominal_pending_intrinsics` and use a nominal 42-degree vertical
field of view rather than claiming physical calibration.

The `top` and `side` streams are explicitly marked virtual auxiliary views in
the Isaac Sim profile. They improve XY and contact/height judgment without
claiming that the lab owns two additional matching physical cameras.

This milestone provides empty-scene diagnostics and a rigid pick-place
collection scene. It does not yet control cloth, validate real COM3
calibration, or send any command to a physical robot. GPU validation uses short
simulated hold episodes; it is not evidence of successful task manipulation.
