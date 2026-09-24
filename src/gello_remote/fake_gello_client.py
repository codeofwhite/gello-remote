"""Send bounded fake Franka GELLO commands to a simulator WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import json
import math
import time

from .protocol import GelloCommand


FRANKA_HOME = (0.0, -0.65, 0.0, -2.25, 0.0, 1.75, 0.78)


def make_command(
    sequence: int, elapsed_s: float, *, amplitude: float, session_id: str
) -> GelloCommand:
    targets = list(FRANKA_HOME)
    targets[0] += amplitude * math.sin(0.5 * elapsed_s)
    targets[5] += amplitude * 0.5 * math.sin(0.35 * elapsed_s)
    return GelloCommand(
        session_id=session_id,
        sequence=sequence,
        timestamp_ns=time.time_ns(),
        joint_targets=targets,
        gripper_closed=int(elapsed_s // 3.0) % 2 == 1,
    )


async def run(url: str, fps: float, duration: float, amplitude: float, session_id: str) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("fake GELLO streaming requires the server extras: pip install -e '.[server]'") from exc

    started = time.monotonic()
    period = 1.0 / fps
    async with websockets.connect(url, max_size=2**22) as websocket:
        async def drain_server_messages() -> None:
            async for _ in websocket:
                pass

        drain_task = asyncio.create_task(drain_server_messages())
        try:
            sequence = 0
            while duration <= 0 or time.monotonic() - started < duration:
                tick_started = time.monotonic()
                command = make_command(
                    sequence, tick_started - started, amplitude=amplitude, session_id=session_id
                )
                await websocket.send(json.dumps({"type": "gello", **command.model_dump()}))
                sequence += 1
                remaining = period - (time.monotonic() - tick_started)
                if remaining > 0:
                    await asyncio.sleep(remaining)
        finally:
            drain_task.cancel()
            with suppress(asyncio.CancelledError):
                await drain_task
    print(f"sent {sequence} bounded GELLO commands to {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8080")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds; 0 runs until interrupted")
    parser.add_argument("--amplitude", type=float, default=0.15, help="maximum test motion in radians")
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        default=None,
        help=(
            "stream as one arm of a dual-arm scene by suffixing the session id "
            "('fake:left' / 'fake:right'); omit for a single-arm scene"
        ),
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.duration < 0 or not 0 < args.amplitude <= 0.25:
        parser.error("fps must be positive, duration non-negative, and amplitude in (0, 0.25]")
    session_id = f"fake:{args.arm}" if args.arm else "legacy"
    asyncio.run(run(args.url, args.fps, args.duration, args.amplitude, session_id))


if __name__ == "__main__":
    main()
