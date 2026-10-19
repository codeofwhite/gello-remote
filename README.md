# GELLO Remote

HTTP policy transport and robot-side client scaffolding for inference integration. Control limits and stale-input handling stay with the
worker or robot-side adapter; the policy service returns action suggestions.

## Scope and branch status

This README describes `main`. Branches contain different capabilities;
check out the branch required by your deployment before following its examples.

| Component | Entry point | Scope |
| --- | --- | --- |
| HTTP policy transport | `server.py`, `client.py`, `protocol.py` | Mock policy for transport and safety-boundary development |

The default transport scaffold does not include the simulation collection
workers. The dual-FR3 development branch is
[`remote/newton-workstation-fix-20261013`](https://github.com/codeofwhite/gello-remote/tree/remote/newton-workstation-fix-20261013).

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
