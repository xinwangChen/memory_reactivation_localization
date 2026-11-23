import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pb2023_eeg.data.dataset import SubjectArrayStore
from pb2023_eeg.data.manifest import SubjectRecord
from pb2023_eeg.data.offline_cache import load_offline_cache, offline_cache_path


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "eeg.npz"
    np.savez(source, X=np.ones((2, 1, 4), dtype=np.float32), trial0=np.arange(2),
             time_s=np.arange(4, dtype=np.float32) / 100)
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({"Exp1_Sub01": {"mean": [0.0], "std": [1.0]}}))
    record = SubjectRecord(1, "Sub01", "Exp1_Sub01", str(source), str(tmp_path / "unused.csv"), 2)
    return record, norm, tmp_path / "cache"


def replace_eeg(record, value):
    path = Path(record.npz_path)
    before = path.stat()
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["X"].fill(value)
    np.savez(path, **arrays)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size


def test_same_size_mtime_eeg_replacement_gets_new_cache(inputs):
    record, norm, cache = inputs
    first = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    np.testing.assert_array_equal(first.get_normalized(0, 0), 1.0)
    old_path = first._offline_cache_path(record)
    old_bytes = old_path.read_bytes()
    replace_eeg(record, 9.0)
    second = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    assert second._offline_cache_path(record) != old_path
    np.testing.assert_array_equal(second.get_normalized(0, 0), 9.0)
    assert old_path.read_bytes() == old_bytes


def test_finite_shape_preserving_cache_corruption_is_rejected(inputs):
    record, norm, cache = inputs
    first = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    first.get_normalized(0, 0)
    path = first._offline_cache_path(record)
    stat = path.stat()
    np.save(path, np.full((2, 1, 4), 8.0, dtype=np.float32))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    for readonly in (False, True):
        with pytest.raises(ValueError, match="cache content/identity mismatch"):
            load_offline_cache(record, cache, None, "replace", readonly=readonly)


def test_legacy_cache_is_not_trusted_or_overwritten(inputs):
    record, norm, cache = inputs
    cache.mkdir()
    stat = Path(record.npz_path).stat()
    key = hashlib.sha256(f"{record.npz_path}:{stat.st_mtime_ns}_{stat.st_size}".encode()).hexdigest()[:16]
    legacy = cache / f"{record.subject_uid}_{key}.npy"
    np.save(legacy, np.zeros((2, 1, 4), dtype=np.float32))
    before = legacy.read_bytes()
    np.testing.assert_array_equal(load_offline_cache(record, cache, None, "replace", readonly=True), 1.0)
    assert list(cache.iterdir()) == [legacy]
    store = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    np.testing.assert_array_equal(store.get_normalized(0, 0), 1.0)
    assert legacy.read_bytes() == before


def test_partial_publication_is_rebuilt_and_complete_cache_is_reused(inputs, monkeypatch):
    import pb2023_eeg.data.offline_cache as module
    record, _, cache = inputs
    cache.mkdir()
    path = offline_cache_path(record, cache, None, "replace")
    np.save(path, np.full((2, 1, 4), 8.0, dtype=np.float32))
    np.testing.assert_array_equal(load_offline_cache(record, cache, None, "replace"), 1.0)
    assert path.with_suffix(".json").is_file()
    original = module.np.load
    def no_source_decompression(candidate, *args, **kwargs):
        if Path(candidate) == Path(record.npz_path):
            raise AssertionError("verified cache should not decompress X again")
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(module.np, "load", no_source_decompression)
    np.testing.assert_array_equal(load_offline_cache(record, cache, None, "replace"), 1.0)


def test_graph_contents_and_mode_have_separate_correct_caches(inputs):
    record, _, cache = inputs
    double = SimpleNamespace(sha256="graph-2", apply=lambda value: value * 2)
    triple = SimpleNamespace(sha256="graph-3", apply=lambda value: value * 3)
    raw = load_offline_cache(record, cache, None, "replace")
    two = load_offline_cache(record, cache, double, "replace")
    three = load_offline_cache(record, cache, triple, "replace")
    concat = load_offline_cache(record, cache, double, "concat")
    np.testing.assert_array_equal(two, raw * 2)
    np.testing.assert_array_equal(three, raw * 3)
    np.testing.assert_array_equal(concat, np.concatenate([raw, raw * 2], axis=1))
    assert len(list(cache.glob("*.npy"))) == 4


def test_source_changed_during_decompression_is_not_published(inputs, monkeypatch):
    import pb2023_eeg.data.offline_cache as module
    record, _, cache = inputs
    def changed(values, graph, mode):
        replace_eeg(record, 9.0)
        return values
    monkeypatch.setattr(module, "apply_graph_input", changed)
    with pytest.raises(ValueError, match="Source EEG changed while loading cache"):
        load_offline_cache(record, cache, None, "replace")
    assert not list(cache.glob("*.npy"))
    assert not list(cache.glob("*.json"))


def test_verified_mmaps_survive_lru_eviction_without_rehashing(inputs, monkeypatch):
    import pb2023_eeg.data.dataset as module
    record, norm, cache = inputs
    store = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    expected = store.get_normalized(0, 0).clone()
    store.cache.clear()  # Simulate another subject evicting the first one.
    def no_rehash(*args, **kwargs):
        raise AssertionError("a store must not hash entire EEG files per trial")
    monkeypatch.setattr(module, "load_offline_cache", no_rehash)
    np.testing.assert_array_equal(store.get_normalized(0, 0), expected)


@pytest.mark.parametrize("corrupt", [False, True])
def test_serialized_store_reopens_and_verifies_readonly_files(inputs, corrupt):
    record, norm, cache = inputs
    store = SubjectArrayStore([record], norm, offline_cache_dir=cache)
    expected = store.get_normalized(0, 0).clone()
    restored = pickle.loads(pickle.dumps(store))
    path = store._offline_cache_path(record)
    if corrupt:
        np.save(path, np.full((2, 1, 4), 8.0, dtype=np.float32))
        with pytest.raises(ValueError, match="cache content/identity mismatch"):
            restored.get_normalized(0, 0)
    else:
        np.testing.assert_array_equal(restored.get_normalized(0, 0), expected)
        reopened = restored._verified_offline_cache[0]
        assert reopened.filename == path and not reopened.flags.writeable


def test_file_identity_does_not_trust_even_unchanged_ctime(tmp_path, monkeypatch):
    from pb2023_eeg.utils import source_files
    path = tmp_path / "same-stat.bin"
    path.write_bytes(b"1111")
    stamp = source_files._stamp(path)
    monkeypatch.setattr(source_files, "_stamp", lambda _: stamp)
    before = source_files.content_sha256(path)
    path.write_bytes(b"9999")
    assert source_files.content_sha256(path) != before


def _cache_worker(arguments):
    record, directory = arguments
    values = load_offline_cache(record, directory, None, "replace")
    return values.shape, float(values.sum())


def test_concurrent_builders_publish_one_valid_cache(inputs):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    record, _, cache = inputs
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_cache_worker, [(record, cache)] * 4))
    assert results == [((2, 1, 4), 8.0)] * 4
    assert len(list(cache.glob("*.npy"))) == len(list(cache.glob("*.json"))) == 1
    np.testing.assert_array_equal(load_offline_cache(record, cache, None, "replace"), 1.0)


