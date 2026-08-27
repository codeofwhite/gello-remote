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
