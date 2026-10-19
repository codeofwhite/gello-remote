# GELLO Remote

Local GELLO control client with a GPU inference service running on the remote
8x RTX 5090 host. The first milestone deliberately uses a mock policy: validate
transport and safety boundaries before connecting real robot actuation.

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

## Remote server

```bash
cd ${WORKSPACE}/gello-remote
python3 -m venv .venv
.venv/bin/pip install -e '.[server]'
.venv/bin/uvicorn gello_remote.server:app --host 127.0.0.1 --port 8000
```

## Local tunnel and smoke test

```bash
ssh -N -L 18000:127.0.0.1:8000 "${SSH_TARGET}"
curl http://127.0.0.1:18000/health
python -m gello_remote.client --url http://127.0.0.1:18000
```

The remote service returns suggestions only. Joint limits, velocity limits,
watchdogs, and emergency stop must remain on the robot-side machine.
