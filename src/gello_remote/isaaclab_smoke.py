"""Bounded, headless Isaac Lab Franka adapter smoke test."""

from __future__ import annotations

import argparse


def main() -> None:
    # Isaac Lab requires AppLauncher to start before importing simulation APIs.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10, help="bounded physics steps (default: 10)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab_assets import FRANKA_PANDA_CFG

        from .backends import IsaacLabFrankaBackend

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

        robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Franka")
        robot = Articulation(cfg=robot_cfg)
        sim.reset()
        robot.reset()

        backend = IsaacLabFrankaBackend(robot, device=sim.device)
        targets = robot.data.default_joint_pos.torch[0, :7].tolist()
        for _ in range(args.steps):
            backend.set_joint_targets(targets)
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
        print(f"Isaac Lab Franka smoke passed: {args.steps} headless steps on {sim.device}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
