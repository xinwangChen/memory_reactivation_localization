from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pb2023_eeg.data.spatial import (
    DEFAULT_ELECTRODES,
    GraphLaplacian,
    atomic_json,
    build_graph,
    graph_options,
)
from pb2023_eeg.data.normalization import load_normalization_stats
from pb2023_eeg.utils.run import snapshot_data_artifacts, validate_snapshot_data_artifacts


@pytest.fixture
def electrodes(tmp_path):
    indices = np.arange(64)
    z = 1.0 - 2.0 * (indices + .5) / 64
    angle = indices * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(1.0 - z ** 2)
    positions = .09 * np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    names = [f"E{i}" for i in indices]
    names[0], names[17] = "Fp1", "Cz"
    path = tmp_path / "electrodes.npz"
    np.savez(path, channel_names=np.asarray(names), positions_m=positions,
             sphere_origin_m=np.zeros(3), sphere_radius_m=np.asarray(.09),
             coord_frame=np.asarray("head"), channel_index=indices,
             template=np.asarray("synthetic"))
    return path


def test_graph_is_reproducible_and_preserves_channel_order(tmp_path: Path, electrodes) -> None:
    payload = build_graph(electrodes, 6)
    assert len(payload["channel_names"]) == 64
    assert payload["channel_names"][0] == "Fp1"
    assert payload["channel_names"][17] == "Cz"
    assert min(payload["neighbor_counts"]) >= 6
    path = tmp_path / "graph.json"
    atomic_json(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    graph = GraphLaplacian.load(path, digest)
    values = np.ones((2, 64, 9), dtype=np.float32)
    transformed = graph.apply(values)
    assert transformed.shape == values.shape
    assert float(np.max(np.abs(transformed))) < 1e-5
    with pytest.raises(ValueError, match="checksum"):
        GraphLaplacian.load(path, "0" * 64)


def test_graph_options_are_explicit_and_validate_unknown_fields() -> None:
    assert graph_options(None) == {
        "enabled": False,
        "n_neighbors": 6,
        "electrodes_npz": str(DEFAULT_ELECTRODES.resolve()),
        "mode": "replace",
    }
    assert graph_options({"graph_laplacian": {"enabled": True, "n_neighbors": 8}})["enabled"]
    with pytest.raises(ValueError, match="positive integer"):
        graph_options({"graph_laplacian": {"n_neighbors": 0}})
    with pytest.raises(ValueError, match="Unknown"):
        graph_options({"graph_laplacian": {"lambda2": 1e-5}})


def test_graph_artifact_is_frozen_with_other_data_protocol(tmp_path: Path, electrodes) -> None:
    files = {}
    for key, suffix in (("manifest_csv", ".csv"), ("split_csv", ".csv"), ("normalization_json", ".json")):
        path = tmp_path / f"source_{key}{suffix}"
        path.write_text("x\n")
        files[key] = str(path)
    # Snapshotting now binds the actual external files referenced by a manifest.
    eeg, meta = tmp_path / 'eeg.npz', tmp_path / 'meta.csv'
    np.savez(eeg, X=np.ones((1, 64, 2)), trial0=np.array([0]))
    meta.write_text('trial0,choice\n0,1\n')
    Path(files['manifest_csv']).write_text(
        f'subject_uid,npz_path,meta_path\nExp1_Sub01,{eeg},{meta}\n')
    graph_path = tmp_path / "graph.json"
    payload = build_graph(electrodes, 6)
    atomic_json(graph_path, payload)
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    config = {"data": {**files, "graph_laplacian_json": str(graph_path), "graph_laplacian_sha256": graph_sha}}
    run = tmp_path / "run"
    snapshot_data_artifacts(config, run)
    assert Path(config["data"]["graph_laplacian_json"]).is_file()
    validate_snapshot_data_artifacts(config["data"], run, required=True)
    Path(config["data"]["graph_laplacian_json"]).write_text("changed")
    with pytest.raises(ValueError, match="Snapshot artifact changed"):
        validate_snapshot_data_artifacts(config["data"], run, required=True)


def test_graph_normalization_requires_the_same_operator(tmp_path: Path, electrodes) -> None:
    payload = build_graph(electrodes, 6)
    graph_path = tmp_path / "graph.json"
    atomic_json(graph_path, payload)
    graph = GraphLaplacian.load(graph_path, hashlib.sha256(graph_path.read_bytes()).hexdigest())
    stats_path = tmp_path / "normalization.json"
    stats_path.write_text(json.dumps({
        "schema_version": 2,
        "graph_laplacian_sha256": graph.sha256,
        "stats": {"Exp1_Sub01": {"mean": [0.0] * 64, "std": [1.0] * 64}},
    }))
    assert load_normalization_stats(stats_path, graph_sha256=graph.sha256)["Exp1_Sub01"]["std"].shape == (64,)
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        load_normalization_stats(stats_path, graph_sha256="different")


def test_fixed_sigma_preserves_common_edge_weights_and_supports_full_graph(tmp_path, electrodes):
    reference = build_graph(electrodes, 6)
    sigma = reference["sigma_m"]
    reference_weights = np.asarray(reference["weights"])
    common = reference_weights > 0
    for k in (3, 6, 12, 24, 63):
        payload = build_graph(electrodes, k, sigma_m=sigma)
        weights = np.asarray(payload["weights"])
        assert payload["sigma_m"] == sigma
        shared = common & (weights > 0)
        np.testing.assert_array_equal(weights[shared], reference_weights[shared])
        if k == 6:
            np.testing.assert_array_equal(payload["matrix"], reference["matrix"])
        if k == 63:
            assert payload["connectivity"] == "full"
            assert payload["neighbor_counts"] == [63] * 64
            assert np.all(weights[~np.eye(64, dtype=bool)] > 0)
        path = tmp_path / f"graph{k}.json"
        atomic_json(path, payload)
        graph = GraphLaplacian.load(path)
        np.testing.assert_allclose(graph.apply(np.ones((1, 64, 3), dtype=np.float32)), 0, atol=1e-6)
    assert build_graph(electrodes, 63)["sigma_m"] != sigma


@pytest.mark.parametrize("sigma", [0, -1, True, "0.04", float("nan"), float("inf")])
def test_fixed_sigma_rejects_invalid_values(sigma, electrodes):
    with pytest.raises(ValueError, match="sigma_m"):
        graph_options({"graph_laplacian": {"sigma_m": sigma}})
    with pytest.raises(ValueError, match="sigma_m"):
        build_graph(electrodes, 6, sigma_m=sigma)
