"""Load ClothForge garments into an existing Isaac Lab workstation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def newton_garment_config(directory, preset):
    """Use the existing ClothForge contact defaults and contact refresh profile."""
    from clothforge.repair import resolve_config
    source = json.loads((Path(directory) / f"config-{preset}.json").read_text())
    return resolve_config(source, {"id": "studio-workstation", "overrides": {
        "topological_contact_filter_threshold": 1, "collision_detection_interval": 1,
    }})


def newton_hand_jacobian(robot):
    """Compute a world-frame hand Jacobian from native FK.

    The installed bridge's gathered Jacobian does not match native FK for this
    dual articulation asset. Central differences keep the arm/body mapping
    explicit without modifying the shared Isaac Lab backend.
    """
    import newton
    import warp as wp
    import torch
    from isaaclab_newton.physics import NewtonManager
    from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
    model, state = NewtonManager.get_model(), NewtonManager.get_state_0()
    prefix = robot.cfg.prim_path + '/'
    body = next(i for i, label in enumerate(model.body_label) if label.startswith(prefix) and label.endswith('/panda_hand'))
    q_starts = model.joint_q_start.numpy()
    q = state.joint_q.numpy()
    scratch = model.state()
    columns = []
    eps = .001
    for joint in range(1, 8):
        index = next(i for i, label in enumerate(model.joint_label)
                     if label.startswith(prefix) and label.endswith(f'/panda_joint{joint}'))
        poses = []
        for sign in (1, -1):
            perturbed = q.copy(); perturbed[q_starts[index]] += sign * eps
            newton.eval_fk(model, wp.array(perturbed, dtype=wp.float32, device=model.device), state.joint_qd, scratch)
            poses.append(torch.tensor(scratch.body_q.numpy()[body], device=robot.device))
        linear = (poses[0][:3] - poses[1][:3]) / (2 * eps)
        angular = axis_angle_from_quat(quat_mul(poses[0][3:].unsqueeze(0),
            quat_conjugate(poses[1][3:].unsqueeze(0))))[0] / (2 * eps)
        columns.append(torch.cat([linear, angular]))
    return torch.stack(columns, dim=1).unsqueeze(0)


def load_garment(directory: Path, parameters: dict, start_m: tuple, backend="physx") -> dict:
    """Resolve a material preset and rotate the sewn garment onto the worktop."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    preset = parameters.get("material_preset", "medium")
    if preset not in {"light", "medium", "heavy"}:
        raise ValueError("material_preset= must be light, medium or heavy")
    config = json.loads((directory / f"config-{preset}.json").read_text())
    if backend == "newton":
        config = newton_garment_config(directory, preset)
        config["simulation"]["iterations"] = int(parameters.get("cloth_iterations", 30))
    material = config["material"]
    material = {**material, **{key: parameters[key] for key in
                ("mass_per_area", "friction", "tri_ke", "tri_ka", "edge_ke") if key in parameters}}
    from clothforge.physx_worker import material_parameters

    mapped = material_parameters(material)
    mesh_path = directory / "mesh.npz"
    mesh_sha = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if mesh_sha != manifest["mesh_sha256"]:
        raise ValueError("ClothForge mesh differs from its manifest")
    with np.load(mesh_path, allow_pickle=False) as source:
        vertices = source["vertices"].copy()
        faces = source["faces"].astype(np.int32)
        rest = source["rest_triangles"].copy() if backend == "newton" else None
        seams = source["seam_edges"].copy() if backend == "newton" else None
    scale = float(parameters.get("garment_scale", 1.0))
    angle = math.radians(float(parameters.get("garment_rotation_deg", 0.0)))
    # The source is sewn in the X/Z plane; rotate rigidly onto X/Y, retaining
    # the front/back panel separation. The native PhysX reference convention
    # uses a tiny separation instead of exactly coincident opposite panels.
    reference = vertices.copy()
    reference[:, 1] *= 1e-4
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    def horizontal(points):
        points = (points - center) * scale
        return np.column_stack((points[:, 0], points[:, 2], -points[:, 1])) @ rotation.T
    vertices, reference = horizontal(vertices), horizontal(reference)
    translation = np.array(start_m, dtype=float)
    translation[:2] += [float(parameters.get("garment_offset_x_m", 0)),
                        float(parameters.get("garment_offset_y_m", 0))]
    translation[2] -= vertices[:, 2].min()
    vertices += translation
    reference += translation
    return dict(vertices=vertices.astype(np.float32), reference=reference.astype(np.float32), faces=faces,
                rest_triangles=rest * scale if rest is not None else None, seam_edges=seams,
                simulation=config.get("simulation", {}),
                metadata={"garment_id": manifest["id"], "category": manifest["category"],
                          "mesh_sha256": mesh_sha, "material_preset": preset,
                          "material": material, "physx_material": mapped, "parameters": parameters,
                          "physics_backend": backend,
                          "simulation": config.get("simulation", {}),
                          "unmapped_material_parameters": ["tri_kd", "edge_kd"] if backend == "physx" else []})


def restore_newton_pattern(builder, garment, particle_offsets):
    """Restore planar elasticity, area-weighted masses and free seam hinges."""
    p = garment["rest_triangles"]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    al = np.linalg.norm(a, axis=1)
    bx = (b * (a / al[:, None])).sum(axis=1)
    by = np.sqrt(np.maximum((b * b).sum(axis=1) - bx * bx, 0))
    if np.any(al <= 1e-12) or np.any(by <= 1e-12):
        raise ValueError("degenerate ClothForge rest triangles")
    matrices = np.zeros((len(p), 2, 2))
    matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1] = al, bx, by
    inverse, areas = np.linalg.inv(matrices), al * by * .5
    faces = garment["faces"]
    masses = np.bincount(faces.reshape(-1),
        weights=np.repeat(areas * garment["metadata"]["material"]["mass_per_area"] / 3, 3),
        minlength=len(garment["vertices"]))
    seams = {tuple(sorted(edge)) for edge in garment["seam_edges"].tolist()}
    for offset in particle_offsets:
        end = offset + len(masses)
        tri_ids = [i for i, tri in enumerate(builder.tri_indices) if offset <= tri[0] < end]
        if len(tri_ids) != len(faces) or not np.array_equal(
                np.asarray(builder.tri_indices)[tri_ids] - offset, faces):
            raise ValueError("Newton rejected or reordered ClothForge triangles")
        for source, target in enumerate(tri_ids):
            builder.tri_poses[target] = inverse[source].tolist()
            builder.tri_areas[target] = float(areas[source])
        builder.particle_mass[offset:end] = masses.tolist()
        for i, edge in enumerate(builder.edge_indices):
            if offset <= edge[2] < end and offset <= edge[3] < end:
                builder.edge_rest_angle[i] = 0.0
                if tuple(sorted([edge[2] - offset, edge[3] - offset])) in seams:
                    builder.edge_bending_properties[i] = (0., 0.)
    builder.color(include_bending=True)
    print("CLOTHFORGE_NEWTON_PATTERN " + json.dumps({
        "particle_offsets": particle_offsets, "material_area_m2": float(areas.sum()),
        "mass_kg": float(masses.sum()), "rest_reference": "planar pattern metric",
        "seams": "shared vertices with free hinges",
    }), flush=True)


def clothforge_newton_manager():
    """Run registered pattern hooks before finalizing the flat scene builder."""
    from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
    from isaaclab_newton.physics import NewtonManager

    class NewtonClothForgeManager(NewtonCoupledMJWarpVBDManager):
        @classmethod
        def instantiate_builder_from_stage(cls):
            super().instantiate_builder_from_stage()
            for hook in NewtonManager._post_replicate_hooks:
                hook(NewtonManager._builder)

    return NewtonClothForgeManager


def spawn_garment(prim_path: str, garment: dict, backend="physx"):
    """Create the surface body on the current stage, then let Isaac Lab own it."""
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObject, DeformableObjectCfg
    from isaaclab_physx.sim import PhysxDeformableBodyPropertiesCfg, PhysxSurfaceDeformableBodyMaterialCfg
    from pxr import UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, prim_path + "/mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(garment["vertices"]))
    mesh.GetFaceVertexCountsAttr().Set([3] * len(garment["faces"]))
    mesh.GetFaceVertexIndicesAttr().Set(garment["faces"].reshape(-1).tolist())
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)
    properties = PhysxDeformableBodyPropertiesCfg(
        self_collision=True, solver_position_iteration_count=int(garment["metadata"]["parameters"].get("cloth_iterations", 30)),
        contact_offset=0.002, rest_offset=0.0003, self_collision_filter_distance=0.0006,
    )
    if backend == "newton":
        from isaaclab_newton.assets import DeformableObject
        from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
        properties = NewtonDeformableBodyPropertiesCfg()
    sim_utils.schemas.define_deformable_body_properties(prim_path, properties, stage=stage, deformable_type="surface")
    sim_mesh = stage.GetPrimAtPath(prim_path + "/sim_mesh")
    sim_mesh.GetAttribute("omniphysics:restShapePoints").Set(Vt.Vec3fArray.FromNumpy(garment["reference"]))
    mapped, material = garment["metadata"]["physx_material"], garment["metadata"]["material"]
    physics = PhysxSurfaceDeformableBodyMaterialCfg(
        density=mapped["density_kg_m3"], youngs_modulus=mapped["young_pa"], poissons_ratio=mapped["poisson"],
        static_friction=material["friction"], dynamic_friction=material["friction"],
        surface_thickness=mapped["thickness_m"], surface_bend_stiffness=mapped["bend_pa"],
    )
    if backend == "newton":
        from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
        physics = NewtonSurfaceDeformableBodyMaterialCfg(
            density=material["mass_per_area"],
            particle_radius=garment["simulation"].get("particle_radius_m", .002),
            **{key: material[key] for key in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")},
        )
    physics.func(prim_path + "/physicsMaterial", physics)
    sim_utils.bind_physics_material(prim_path, prim_path + "/physicsMaterial", stage=stage)
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.42, 0.63), roughness=0.9)
    visual.func(prim_path + "/visualMaterial", visual)
    sim_utils.bind_visual_material(mesh.GetPath().pathString, prim_path + "/visualMaterial", stage=stage)
    asset = DeformableObject(cfg=DeformableObjectCfg(prim_path=prim_path, spawn=None))
    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager
        def restore_pattern(builder):
            restore_newton_pattern(builder, garment, asset._registry_entry.particle_offsets)
        NewtonManager._post_replicate_hooks.append(restore_pattern)
    return asset
