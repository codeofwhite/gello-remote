# Agent ownership

- Local Codex owns `src/gello_remote/client.py`, robot adapters, and local safety.
- Remote Codex owns `src/gello_remote/server.py`, model loading, and GPU runtime.
- Protocol changes in `src/gello_remote/protocol.py` require tests and review from both sides.
- Local work uses branches prefixed `local/`; remote work uses `remote/`.
- Never commit credentials, model weights, robot calibration, or private IP keys.
- Never bypass local joint, velocity, watchdog, or emergency-stop enforcement.
