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

## Simulation-first Franka backend

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
