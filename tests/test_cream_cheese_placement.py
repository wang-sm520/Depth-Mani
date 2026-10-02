import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from bddl.parsing import scan_tokens

from depth_policy.common import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG = ROOT / "configs/cream_cheese_on.json"
EXPANDED_CONFIG = ROOT / "configs/cream_cheese_3x_on_eval.json"
LIBERO_ROOT = Path(json.loads(OFFICIAL_CONFIG.read_text())["libero_root"]) / "libero/libero"
ORIGINAL_BDDL = LIBERO_ROOT / "bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl"
CHEESE_XML = LIBERO_ROOT / "assets/stable_hope_objects/cream_cheese/cream_cheese.xml"


def region_bounds(tokens, name):
    regions = next(group for group in tokens if group[0] == ":regions")
    region = next(entry for entry in regions[1:] if entry[0] == name)
    ranges = next(attr for attr in region[1:] if attr[0] == ":ranges")
    return ranges[1][0]


def test_only_cheese_region_differs_from_official():
    config = json.loads(EXPANDED_CONFIG.read_text())
    assert config["evaluation_only"] is True
    assert config["evaluation_bddl"] == "configs/evaluation/cream_cheese_3x.bddl"
    altered = ROOT / config["evaluation_bddl"]
    assert sha256_file(altered) == config["evaluation_bddl_sha256"]
    original = scan_tokens(filename=str(ORIGINAL_BDDL))
    expanded = scan_tokens(filename=str(altered))
    np.testing.assert_allclose(
        [float(value) for value in region_bounds(original, "cream_cheese_region")],
        [-0.06, 0.12, -0.04, 0.14],
    )
    np.testing.assert_allclose(
        [float(value) for value in region_bounds(expanded, "cream_cheese_region")],
        [-0.14, 0.04, 0.04, 0.22],
    )
    normalized = copy.deepcopy(expanded)
    region_bounds(normalized, "cream_cheese_region")[:] = region_bounds(
        original, "cream_cheese_region"
    )
    assert normalized == original


def test_effective_center_bounds_are_exactly_triple():
    original = json.loads(OFFICIAL_CONFIG.read_text())
    expanded = json.loads(EXPANDED_CONFIG.read_text())
    assert original["task"] == expanded["task"] == "put_the_cream_cheese_in_the_bowl"
    assert original["official_instruction"] == expanded["official_instruction"] == \
        "Put the cream cheese on the bowl"
    assert not original.get("evaluation_only")
    assert original["official_center_bounds"] == expanded["official_center_bounds"]
    for key in (
        "openpi_root", "libero_root", "teacher_checkpoint", "suite", "resolution",
        "control_freq", "settle_steps", "max_steps", "teacher_replan_steps",
        "seed", "manifest", "teacher_port",
    ):
        assert original[key] == expanded[key]

    site = ET.parse(CHEESE_XML).getroot().find(".//site[@name='horizontal_radius_site']")
    radius = float(site.attrib["pos"].split()[0])
    assert radius == 0.03

    original_tokens = scan_tokens(filename=str(ORIGINAL_BDDL))
    expanded_tokens = scan_tokens(filename=str(ROOT / expanded["evaluation_bddl"]))
    original_rect = np.array(region_bounds(original_tokens, "cream_cheese_region"), dtype=float)
    expanded_rect = np.array(region_bounds(expanded_tokens, "cream_cheese_region"), dtype=float)
    old_start = original_rect[:2] + radius
    old_end = original_rect[2:] - radius
    new_start = expanded_rect[:2] + radius
    new_end = expanded_rect[2:] - radius
    assert np.all(old_start > old_end)  # Original NumPy uniform arguments are reversed.
    assert np.all(new_start < new_end)
    old_bounds = np.stack([np.minimum(old_start, old_end), np.maximum(old_start, old_end)])
    new_bounds = np.stack([np.minimum(new_start, new_end), np.maximum(new_start, new_end)])
    np.testing.assert_allclose(old_bounds, [[-0.07, 0.11], [-0.03, 0.15]])
    np.testing.assert_allclose(new_bounds, [[-0.11, 0.07], [0.01, 0.19]])
    np.testing.assert_allclose(new_bounds.mean(axis=0), old_bounds.mean(axis=0))
    np.testing.assert_allclose(np.diff(new_bounds, axis=0), 3 * np.diff(old_bounds, axis=0))
    for config, bounds, field in (
        (original, old_bounds, "official_center_bounds"),
        (expanded, old_bounds, "official_center_bounds"),
        (expanded, new_bounds, "evaluation_center_bounds"),
    ):
        recorded = config[field]["cream_cheese_1"]
        np.testing.assert_allclose(recorded["minimum_xy_m"], bounds[0])
        np.testing.assert_allclose(recorded["maximum_xy_m"], bounds[1])


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") != "osmesa", reason="Run with scripts/sim_cpu.sh")
def test_native_expanded_sampler_and_reset_are_legal():
    from depth_policy.simulation import make_env

    config = json.loads(EXPANDED_CONFIG.read_text())
    environment, _ = make_env(config, 2026092400)
    try:
        samplers = environment.env.placement_initializer.samplers.values()
        sampler = next(
            part for part in samplers if any(obj.name == "cream_cheese_1" for obj in part.mujoco_objects)
        )
        assert sampler.ensure_object_boundary_in_range
        assert sampler.ensure_valid_placement
        assert sampler.rotation == (0.0, 0.0)
        assert sampler.rotation_axis == "x"
        assert sampler.mujoco_objects[0].horizontal_radius == 0.03
        np.testing.assert_allclose(environment.env.workspace_offset[:2], [0, 0])
        np.testing.assert_allclose(sampler.x_ranges, [[-0.14, 0.04]])
        np.testing.assert_allclose(sampler.y_ranges, [[0.04, 0.22]])

        environment.env.reset()
        positions = {
            name: environment.sim.data.get_joint_qpos(obj.joints[-1])[:2]
            for name, obj in environment.env.objects_dict.items()
        }
        cheese = positions["cream_cheese_1"]
        np.testing.assert_array_less([-0.1100001, 0.0699999], cheese)
        np.testing.assert_array_less(cheese, [0.0100001, 0.1900001])
        original_tokens = scan_tokens(filename=str(ORIGINAL_BDDL))
        for name, region in (
            ("wine_bottle_1", "wine_bottle_region"),
            ("akita_black_bowl_1", "akita_black_bowl_region"),
            ("plate_1", "plate_region"),
        ):
            obj = environment.env.objects_dict[name]
            rectangle = np.array(region_bounds(original_tokens, region), dtype=float)
            start = rectangle[:2] + obj.horizontal_radius
            end = rectangle[2:] - obj.horizontal_radius
            assert np.all(positions[name] >= np.minimum(start, end) - 1e-7)
            assert np.all(positions[name] <= np.maximum(start, end) + 1e-7)
        assert not environment.check_success()
    finally:
        environment.close()
