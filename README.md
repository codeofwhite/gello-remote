# GELLO Remote

Robot teleoperation and simulation workers, with a separate HTTP policy transport
for inference integration. Control limits and stale-input handling stay with the
worker or robot-side adapter; the policy service returns action suggestions.

## Scope and branch status

This README describes `remote/simulation`. Branches contain different capabilities;
check out the branch required by your deployment before following its examples.

| Component | Entry point | Scope |
| --- | --- | --- |
| HTTP policy transport | `server.py`, `client.py`, `protocol.py` | Mock policy for transport and safety-boundary development |
| Franka backend | `backends.py` | Mock backend; no physical robot adapter |
| Isaac Lab diagnostic | `isaaclab_smoke.py` | Bounded simulated joint hold; requires a separate simulator runtime |

## Requirements and installation

Python 3.10 or newer is required by the package. The HTTP service uses the
`server` extra. Simulation additionally needs a compatible Isaac Lab / Isaac Sim
or Genesis environment, external scene assets, and any shared-contract package
used by the selected worker. Installing this package does not install a simulator.
For browser GELLO input, use a desktop browser with Web Serial support and USB
access on the computer running that browser.

## Local configuration

Set these example values for your own workspace and SSH host before using the
commands below. The IP address is reserved for documentation; replace it with
your reachable streaming address when livestreaming is needed. Keep credentials
and machine-specific configuration outside Git.

```bash
export WORKSPACE="$HOME/robot-workspace"
export ISAACLAB_ENV="$WORKSPACE/.venvs/isaaclab"
export SSH_TARGET="user@host.example"
export PUBLIC_IP="203.0.113.10"
```

## HTTP transport quick start

```bash
cd "${WORKSPACE}/gello-remote"
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[server]'
.venv/bin/uvicorn gello_remote.server:app --host 127.0.0.1 --port 8000
```

## Connect through an SSH tunnel

```bash
ssh -N -L 18000:127.0.0.1:8000 "${SSH_TARGET}"
curl http://127.0.0.1:18000/health
"${WORKSPACE}/gello-remote/.venv/bin/python" -m gello_remote.client --url http://127.0.0.1:18000
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
the compatible Isaac Lab APIs. Isaac Lab imports are lazy, so ordinary imports and
unit tests do not require the simulator environment. Run it headlessly with:

```bash
env CONDA_PREFIX=${ISAACLAB_ENV} \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKSPACE}/gello-remote/src \
  ${WORKSPACE}/IsaacLab/isaaclab.sh -p -m gello_remote.isaaclab_smoke \
  --steps 10 --device cuda:0 --viz none
```

The smoke process creates one simulated Franka, holds its default seven arm
joints for a bounded number of physics steps, and exits. It does not expose a
real-robot adapter, connect to robot hardware, or change the existing HTTP API.


## Verification and operating boundaries

Run CPU-only transport and safety tests in an environment with the test extra:

```bash
cd "${WORKSPACE}/gello-remote"
.venv/bin/python -m pip install -e '.[test,server]'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_protocol.py
```

Simulator tests may require additional external packages, scene assets, and a
sibling Robot Data Studio checkout. Use the deployment's pinned versions rather
than assuming that all branch combinations are compatible. Do not bypass joint
limits, velocity limits, watchdogs, or emergency-stop enforcement. These software
checks do not certify real-robot safety.

Keep API credentials, robot calibration, private keys, and local machine settings
outside version control. Configuration examples do not contain working credentials.
Historical commits are retained; current-file cleanup does not remove old values
from Git history.
