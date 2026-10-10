import hashlib
import json

import numpy as np
import pytest

from gello_remote.isaac_clothforge import load_garment


@pytest.fixture
def garment(tmp_path):
    # Two separated panels, as in the sewn source meshes.
    vertices = np.array([[-.2, -.005, 0], [.2, -.005, 0], [0, -.005, .6],
                         [-.2, .005, 0], [.2, .005, 0], [0, .005, .6]])
    faces = np.array([[0, 1, 2], [3, 5, 4]])
    np.savez(tmp_path / 'mesh.npz', vertices=vertices, faces=faces,
             rest_triangles=vertices[faces][:, :, [0, 2]], seam_edges=[[0, 1]])
    sha = hashlib.sha256((tmp_path / 'mesh.npz').read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(id='tshirt-test', category='tshirt', mesh_sha256=sha)))
    for preset, density in [('light', .1), ('medium', .22), ('heavy', .4)]:
        (tmp_path / f'config-{preset}.json').write_text(json.dumps({'material': dict(
            mass_per_area=density, tri_ke=3000, tri_ka=3000, edge_ke=.005, friction=.5,
            tri_kd=.2, edge_kd=.001), 'simulation': {}}))
    return tmp_path, vertices


def test_horizontal_placement_preserves_sewn_geometry_and_scaled_rest(garment):
    path, vertices = garment
    before = (path / 'mesh.npz').read_bytes()
    result = load_garment(path, dict(garment_scale=.5, garment_rotation_deg=90,
                                   garment_offset_x_m=.1), (.55, -.06, .8))
    v = result['vertices']
    assert v[:, 2].min() == pytest.approx(.8)
    assert (v.max(0) + v.min(0))[:2] / 2 == pytest.approx([.65, -.06])
    assert np.ptp(v, axis=0) == pytest.approx([.3, .2, .005])
    assert np.linalg.norm(v[0]-v[3]) == pytest.approx(.005)
    assert np.linalg.norm(result['reference'][0]-result['reference'][3]) > 0
    assert (path / 'mesh.npz').read_bytes() == before


def test_preset_and_override_reach_physx_units_without_changing_source(garment):
    path, _ = garment
    baseline = (path / 'config-heavy.json').read_bytes()
    result = load_garment(path, dict(material_preset='heavy', friction=.8, mass_per_area=.33), (0, 0, .8))
    assert result['metadata']['material']['friction'] == .8
    mapping = result['metadata']['physx_material']
    assert mapping['density_kg_m3'] * mapping['thickness_m'] == pytest.approx(.33)
    assert result['metadata']['material_preset'] == 'heavy'
    assert (path / 'config-heavy.json').read_bytes() == baseline


def test_changed_mesh_is_not_loaded_as_the_original_garment(garment):
    path, _ = garment
    with (path / 'mesh.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='differs'):
        load_garment(path, {}, (0, 0, .8))


def test_newton_rest_metric_mass_and_seams_preserve_other_assets(garment):
    from types import SimpleNamespace
    from gello_remote.isaac_clothforge import restore_newton_pattern
    path, _ = garment
    result = load_garment(path, dict(garment_scale=.5, mass_per_area=.4), (0, 0, .8), 'newton')
    builder = SimpleNamespace(
        tri_indices=[[0, 1, 2], [3, 4, 5], [6, 8, 7]],
        tri_poses=[None] * 3, tri_areas=[99.] * 3,
        particle_mass=[99.] * 9, edge_indices=[[-1, -1, 0, 1], [-1, -1, 3, 4], [-1, -1, 6, 7]],
        edge_rest_angle=[1.] * 3, edge_bending_properties=[(.005, .001)] * 3,
        color=lambda **kwargs: None,
    )
    restore_newton_pattern(builder, result, [3])
    assert builder.tri_areas == pytest.approx([99., .03, .03])
    assert sum(builder.particle_mass[3:]) == pytest.approx(.024)
    assert builder.particle_mass[:3] == [99.] * 3
    assert builder.edge_rest_angle == [1., 0., 0.]
    assert builder.edge_bending_properties == [(.005, .001), (0., 0.), (.005, .001)]
    assert np.linalg.det(builder.tri_poses[1]) == pytest.approx(1 / .06)
    assert result['metadata']['unmapped_material_parameters'] == []
    assert result['simulation']['soft_contact_ke'] == 1e4
    assert result['simulation']['soft_contact_kd'] == 10.
    assert result['simulation']['collision_detection_interval'] == 1
    assert result['simulation']['iterations'] == 30


def test_newton_flat_scene_runs_pattern_hook_before_finalization(monkeypatch):
    import sys
    from types import SimpleNamespace
    from gello_remote.isaac_clothforge import clothforge_newton_manager
    events = []
    base = SimpleNamespace(_builder=None, _post_replicate_hooks=[lambda builder: events.append(('pattern', builder))])
    class CoupledManager:
        @classmethod
        def instantiate_builder_from_stage(cls):
            events.append(('load', None))
            base._builder = 'loaded-builder'
    monkeypatch.setitem(sys.modules, 'isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager',
                        SimpleNamespace(NewtonCoupledMJWarpVBDManager=CoupledManager))
    monkeypatch.setitem(sys.modules, 'isaaclab_newton.physics', SimpleNamespace(NewtonManager=base))
    clothforge_newton_manager().instantiate_builder_from_stage()
    assert events == [('load', None), ('pattern', 'loaded-builder')]
