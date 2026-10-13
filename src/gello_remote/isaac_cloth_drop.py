"""ClothForge observation in Isaac RTX with native Newton VBD, without robots."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import queue
import threading
import time
from http import HTTPStatus


def run_cloth_drop(args, app):
    from .isaaclab_gello import _enable_newton_extensions
    _enable_newton_extensions()
    import carb
    import cv2
    import numpy as np
    import torch
    import warp as wp
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import NewtonCfg, NewtonManager
    from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg, NewtonModelCfg
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed
    from .isaac_clothforge import newton_garment_config, clothforge_newton_manager, load_garment, spawn_garment
    from .isaaclab_gello import (
        _enable_newton_extensions, _apply_renderer_kit_settings, _bimanual_tabletop_geometry,
        _scene_camera_overrides, _camera_intrinsic_matrix, CameraPreviewStore,
        TelemetryStore, RuntimePerformance, GelloWebServer, WEB_ROOT,
    )

    _apply_renderer_kit_settings(carb.settings.get_settings(), args.renderer_profile)
    params = args.clothforge_params
    height = float(params.get('drop_height_m', .25))
    if not np.isfinite(height) or not .02 <= height <= 1:
        raise ValueError('drop_height_m must be in [.02, 1]')
    config = newton_garment_config(args.clothforge_dir, params.get('material_preset', 'medium'), params)
    settings = config['simulation']
    settings['gpu_physical_index'] = args.gpu_id
    friction = float(params.get('friction', config['material']['friction']))

    @configclass
    class ClothPhysicsCfg(NewtonCfg):
        model_cfg: NewtonModelCfg | None = None

    physics = ClothPhysicsCfg(
        solver_cfg=VBDSolverCfg(
            class_type=clothforge_newton_manager(coupled=False),
            iterations=settings['iterations'], particle_enable_self_contact=True,
            particle_self_contact_radius=settings['self_contact_radius_m'],
            particle_self_contact_margin=settings['self_contact_margin_m'],
            particle_collision_detection_interval=settings['collision_detection_interval'],
            particle_topological_contact_filter_threshold=settings['topological_contact_filter_threshold'],
            particle_vertex_contact_buffer_size=settings['vertex_contact_buffer_size'],
            particle_edge_contact_buffer_size=settings['edge_contact_buffer_size'],
            integrate_with_external_rigid_solver=False,
        ),
        model_cfg=NewtonModelCfg(soft_contact_ke=settings['soft_contact_ke'],
                                soft_contact_kd=settings['soft_contact_kd'], soft_contact_mu=friction),
        num_substeps=settings['substeps'], use_cuda_graph=args.newton_cuda_graph,
    )
    dt = 1 / args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics=physics))
    stage = omni.usd.get_context().get_stage()
    ground = sim_utils.CuboidCfg(size=(8., 8., .05), collision_props=sim_utils.CollisionPropertiesCfg(),
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.14, .16, .19), roughness=.72))
    ground.func('/World/defaultGroundPlane', ground, translation=(0., 0., -.025))
    light = sim_utils.DomeLightCfg(intensity=350., color=(1., 1., 1.), texture_file=str(args.hdri_file),
                                  texture_format='latlong', visible_in_primary_ray=True)
    light.func('/World/Light', light)
    bench = sim_utils.UsdFileCfg(usd_path=str(args.workstation_usd))
    bench.func('/World/Workstation', bench)
    from robot_data_contracts.usd_surface import apply_environment, workstation_targets
    apply_environment(stage, args.environment_appearance, workstation_targets(args.visual_asset['composition']))
    geometry = _bimanual_tabletop_geometry(args.visual_asset['composition']['table_top_z_m'], args.visual_asset['composition'])
    table = sim_utils.CuboidCfg(size=geometry['work_surface_size_m'], collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(opacity=0.))
    table.func('/World/WorkSurface', table, translation=geometry['work_surface_center_m'])
    from pxr import UsdGeom
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/WorkSurface')).MakeInvisible()
    center = args.visual_asset['composition']['task_center_world_m']
    garment = load_garment(args.clothforge_dir, params, (center[0], center[1], geometry['work_surface_top_m'] + height), 'newton')
    cloth = spawn_garment('/World/ClothForgeGarment', garment, 'newton')
    overrides = _scene_camera_overrides(args.camera_profile, work_surface_top_m=geometry['work_surface_top_m'],
                                       task_center_xy=tuple(center[:2]))
    overrides['side'], overrides['wrist'] = dict(overrides['left']), dict(overrides['right'])
    cameras = {}
    values = args.camera_lighting['values'] if args.camera_lighting else {}
    for name in ('base', 'top', 'side', 'wrist'):
        pose = overrides[name]
        matrix, _ = _camera_intrinsic_matrix(args.camera_profile['roles']['agentview'], width=args.width,
            height=args.height, nominal_vertical_fov_deg=pose.get('nominal_vertical_fov_deg', 42.) + values.get('fov_offset_deg', 0))
        cfg = CameraCfg(prim_path='/World/RecordCamera_' + name, update_period=1 / args.preview_fps,
                        height=args.height, width=args.width, data_types=['rgb'],
                        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(matrix, width=args.width, height=args.height, clipping_range=(.01, 100.)))
        cameras[name] = Camera(cfg=cfg)
    sim.reset()
    from isaacsim.core.rendering_manager import ViewportManager
    for name, camera in cameras.items():
        ViewportManager.set_camera_view(stage.GetPrimAtPath(camera.cfg.prim_path),
            eye=tuple(overrides[name]['eye_world_m']), target=tuple(overrides[name]['target_world_m']))
    # Initialize RTX sensors and tone-map settings before applying imaging overrides.
    sim.render()
    for camera in cameras.values():
        camera.update(dt=0.)
        _ = camera.data.output['rgb']
    if args.camera_lighting:
        from robot_data_contracts.usd_camera_lighting import apply_camera, apply_lighting, apply_imaging
        for camera in cameras.values():apply_camera(stage, camera.cfg.prim_path, {**values, 'fov_offset_deg': 0}, args.width, args.height)
        apply_lighting(stage, values); apply_imaging(carb.settings.get_settings(), values)
    for camera in cameras.values():camera.reset()
    model = NewtonManager.get_model()
    if model.body_count != 0:
        raise RuntimeError(f'Cloth observation unexpectedly contains {model.body_count} dynamic bodies')
    initial = model.state(); initial.assign(NewtonManager.get_state_0())
    output = Path(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = dict(scene_id=args.scene, solver='newton_vbd', physical_gpu_id=args.gpu_id, logical_device=args.device,
                    robots=0, dynamic_bodies=model.body_count, fixed_vertices=int((model.particle_inv_mass.numpy() == 0).sum()),
                    particles=model.particle_count, shapes=model.shape_count,
                    color_groups=len(model.particle_color_groups),
                    parameters=params, simulation=settings, physics_hz=args.physics_hz,
                    camera_profile=args.camera_profile, renderer_profile=args.renderer_profile,
                    garment=garment['metadata'], drop_height_m=height)
    (output/'cloth_observation_config.json').write_text(json.dumps(snapshot, indent=2))
    telemetry = TelemetryStore()
    telemetry.update(backend='isaaclab', control_source='scripted', observation_only=True,
        scene_id=args.scene, physical_gpu_id=args.gpu_id, sim_ready=True, scene_prims_ready=True, viewport_ready=True,
        solver='newton_vbd', robots=0, simulation_numerics=settings, cloth_node_count=model.particle_count,
        shape_count=model.shape_count, color_group_sizes=[len(group) for group in model.particle_color_groups],
        soft_contact_capacity=NewtonManager._contacts.soft_contact_max, table_top_m=geometry['work_surface_top_m'])

    class FullPreview(CameraPreviewStore):
        @staticmethod
        def _encode_jpeg(image):
            ok, buf = cv2.imencode('.jpg', cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:raise RuntimeError('JPEG encode failed')
            return buf.tobytes()

    previews = FullPreview(target_fps=args.preview_fps)
    commands = queue.SimpleQueue()
    stop = threading.Event(); ready = threading.Event(); server_box = {}; failures = []
    def request(connection, req):
        if 'Upgrade' in req.headers:return None
        if req.path.split('?')[0] != '/':return GelloWebServer._respond(connection, HTTPStatus.NOT_FOUND, 'Not found')
        response = GelloWebServer._respond(connection, HTTPStatus.OK, (WEB_ROOT/'cloth_drop.html').read_text())
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response
    def client(ws):
        sent = {}
        try:
            while not stop.is_set():
                ws.send(json.dumps({'type': 'status', **telemetry.snapshot(), 'previews': previews.stats()}))
                for name, version, packet in previews.packets_since(sent):
                    ws.send(packet); sent[name] = version
                try:message = ws.recv(timeout=.1)
                except TimeoutError:continue
                data = json.loads(message)
                if data.get('type') != 'observation' or data.get('command') not in {'pause', 'resume', 'reset'}:
                    ws.send(json.dumps({'type':'error','message':'Supported commands: pause, resume, reset'}));continue
                commands.put(data['command'])
        except ConnectionClosed:pass
    def serve_thread():
        try:
            with serve(client, args.host, args.port, process_request=request) as server:
                server_box['server'] = server; ready.set(); server.serve_forever()
        except BaseException as exc:failures.append(exc);ready.set()
    previews.start()
    thread = threading.Thread(target=serve_thread, daemon=True); thread.start(); ready.wait(10)
    if failures:raise RuntimeError('Observation server failed') from failures[0]
    if not ready.is_set():raise RuntimeError('Observation server readiness timeout')
    print('ISAAC_SCENE_READY '+json.dumps(dict(scene_prims_ready=True, viewport_ready=True)), flush=True)
    print('ISAAC_CONTROL_READY '+json.dumps(dict(control_source='scripted', host=args.host, port=args.port)), flush=True)
    perf = RuntimePerformance(args.runtime_performance_report, lambda:torch.cuda.synchronize(sim.device), args.runtime_performance_mode)
    step = frame = 0; epoch = reset_count = 0; paused = False; started = time.perf_counter(); last_capture = 0
    positions = deque(maxlen=120); sim_times = deque(maxlen=120); wall_times = deque(maxlen=120); captures = deque(maxlen=120); sample_epochs = deque(maxlen=120)
    images = None
    decimation = args.preview_control_decimation * int(round(args.physics_hz / args.control_hz))
    stream_decimation = args.stream_physics_decimation
    try:
        while app.is_running():
            changed = False
            while not commands.empty():
                command = commands.get()
                if command == 'pause':paused = True
                elif command == 'resume':paused = False
                else:
                    NewtonManager.get_state_0().assign(initial); NewtonManager.get_state_1().assign(initial)
                    NewtonManager._mark_state_dirty(); cloth.reset(); epoch += 1; reset_count += 1
                    last_capture = step
                    for camera in cameras.values():camera.reset()
                    sim.render_context.reset_transform_cadence()
                    changed = True
            if failures:raise RuntimeError('Observation server stopped') from failures[0]
            capture = (step % decimation == 0 and not paused) or changed
            if capture:
                with perf.measure('capture_scene_sync'):sim.render(skip_app_pumping=True)
                with perf.measure('camera_capture'):
                    for camera in cameras.values():camera.update(dt=(step-last_capture)*dt)
                    images = {name:camera.data.output['rgb'].torch[0].detach().cpu().numpy() for name,camera in cameras.items()}
                    last_capture = step
                stamp = time.monotonic_ns();previews.submit(images, frame_index=frame, captured_at_ns=stamp)
                if args.smoke_steps or frame == 0:
                    for name,image in images.items():
                        directory=output/(name+'_rgb');directory.mkdir(exist_ok=True)
                        cv2.imwrite(str(directory/f'{frame:06d}.jpg'),cv2.cvtColor(image[...,:3],cv2.COLOR_RGB2BGR))
                positions.append(NewtonManager.get_state_0().particle_q.numpy().copy());sim_times.append(step*dt)
                wall_times.append(time.perf_counter()-started);captures.append(stamp);sample_epochs.append(epoch);frame += 1
            with perf.measure('telemetry'):
                if capture or paused:
                    q = wp.to_torch(NewtonManager.get_state_0().particle_q)
                    telemetry.update(paused=paused, reset_count=reset_count, observation_epoch=epoch,
                        physics_step=step, simulation_time_s=step*dt, wall_seconds=time.perf_counter()-started,
                        physics_actual_hz=step/max(time.perf_counter()-started,1e-9),
                        real_time_factor=step*dt/max(time.perf_counter()-started,1e-9),
                        cloth_center_m=q.mean(0).cpu().tolist(), cloth_z_range_m=[float(q[:,2].min().item()),float(q[:,2].max().item())],
                        soft_contact_count=int(NewtonManager._contacts.soft_contact_count.numpy()[0]), camera_frame=frame)
            if paused:
                sim.render();time.sleep(.02);continue
            with perf.measure('physics'):sim.step(render=False);cloth.update(dt)
            step += 1
            if step % stream_decimation == 0:
                with perf.measure('viewport'):sim.render()
            if capture:perf.publish(step,dt)
            if args.smoke_steps and step >= args.smoke_steps:break
    finally:
        stop.set();server_box['server'].shutdown();thread.join(timeout=5);previews.stop()
        perf.publish(step,dt)
        if images is not None:
            for name,image in images.items():cv2.imwrite(str(output/(name+'_final.jpg')),cv2.cvtColor(image[...,:3],cv2.COLOR_RGB2BGR))
        if positions:
            np.savez_compressed(output/'cloth_observation.npz',positions=np.asarray(positions),simulation_time_s=list(sim_times),
                                wall_time_s=list(wall_times),capture_timestamp_ns=list(captures),epoch=list(sample_epochs))
        result={**telemetry.snapshot(),'physics_steps':step,'frames':frame,'wall_seconds':time.perf_counter()-started,
                'robots':0,'solver':'newton_vbd','diagnostic_only':True,'episode_created':False}
        (output/'cloth_observation_result.json').write_text(json.dumps(result,indent=2))
    return True
