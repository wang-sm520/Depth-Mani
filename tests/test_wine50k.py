import json

import numpy as np

from depth_policy.common import ROOT, atomic_json


def test_libero_config_readers_never_see_truncated_yaml(tmp_path, monkeypatch):
    from pathlib import Path
    import sys
    import yaml
    import depth_policy.simulation as simulation

    monkeypatch.setattr(simulation, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    config = {"libero_root": str(tmp_path / "upstream")}
    simulation.configure_libero(config)
    destination = tmp_path / "runtime/libero/config.yaml"
    expected = dict(yaml.safe_load(destination.read_text()))
    original_open = Path.open

    def open_with_concurrent_reader(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if ("w" in mode or "x" in mode) and path.parent == destination.parent:
            try:
                observed = dict(yaml.safe_load(destination.read_text()))
                assert observed == expected
            except BaseException:
                stream.close()
                raise
        return stream

    monkeypatch.setattr(Path, "open", open_with_concurrent_reader)
    simulation.configure_libero(config)
    assert dict(yaml.safe_load(destination.read_text())) == expected


def test_wine50k_protocol_uses_4000_paired_rollouts():
    config = json.loads((ROOT / "configs/wine50k_experiment.json").read_text())
    assert config["checkpoint_steps"] == [10000, 20000, 30000, 40000, 50000]
    assert config["training_steps"] == 50000 and config["training_limit"] == 300
    assert config["modalities"] == ["depth", "rgb"]
    assert len(config["checkpoint_steps"]) * len(config["modalities"]) * len(config["groups"]) * config["episodes_per_group"] == 4000
    seeds = [set(range(group["seed"], group["seed"] + config["episodes_per_group"]))
             for group in config["groups"].values()]
    assert not seeds[0] & seeds[1]


def test_wine50k_figures_cover_both_distributions(tmp_path, monkeypatch):
    import scripts.wine50k_report as module

    config = json.loads((ROOT / "configs/wine50k_experiment.json").read_text())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rows, details = [], {}
    for group in config["groups"]:
        path = tmp_path / "data/wine/evaluation" / config["label"] / f"{group}.json"
        atomic_json(path, {"states": [{"initial_positions": {"wine_bottle_1": [-0.2, -0.05, 0.97]}}
                                     for index in range(200)]})
        for step in config["checkpoint_steps"]:
            for modality in config["modalities"]:
                rows.append({"distribution": group, "training_step": step, "modality": modality,
                             "successes": 100, "success_rate": 0.5})
                details[f"{modality}-{step}-{group}"] = {"success_by_initial_state": (np.arange(200) % 2 == 0).tolist()}
    destination = module.make_figures(config, rows, details)
    assert {path.name for path in destination.glob("*.png")} == {
        "success-curves.png", "success-table.png", "official-range200-position-maps.png",
        "expanded10cm200-position-maps.png"}
