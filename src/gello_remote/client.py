import argparse
import time

import httpx

from .protocol import Action, Observation


def request_action(base_url: str, observation: Observation, timeout: float = 0.5) -> Action:
    response = httpx.post(
        f"{base_url.rstrip('/')}/v1/action",
        json=observation.model_dump(),
        timeout=timeout,
    )
    response.raise_for_status()
    return Action.model_validate(response.json())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    args = parser.parse_args()
    observation = Observation(
        sequence=0,
        timestamp_ns=time.time_ns(),
        joint_positions=[0.0] * 7,
    )
    print(request_action(args.url, observation).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
