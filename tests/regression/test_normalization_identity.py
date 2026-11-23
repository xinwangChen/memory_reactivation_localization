import json

import pytest

from pb2023_eeg.data.normalization import compute_normalization_stats, load_normalization_stats, validate_normalization_identity
from .helpers_subject_cv import _bundle
from pb2023_eeg.data.manifest import write_block_cv_splits


def test_per_subject_statistics_reject_another_fold_and_allow_record_subsets(tmp_path):
    _, records = _bundle(tmp_path)
    splits = write_block_cv_splits(records, tmp_path / 'blocks', n_folds=2, seed=42)
    path = tmp_path / 'normalization.json'
    compute_normalization_stats(records, splits[0], path)
    validate_normalization_identity(records[:1], splits[0], path, mode='per_subject')
    with pytest.raises(ValueError, match='identity differs'):
        validate_normalization_identity(records, splits[1], path, mode='per_subject')
    assert set(load_normalization_stats(path)) == {r.subject_uid for r in records}
    raw = json.loads(path.read_text())
    raw[records[0].subject_uid].pop('fit_split_sha256')
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='identity differs'):
        validate_normalization_identity(records, splits[0], path, mode='per_subject')
    with pytest.raises(ValueError, match='identity differs'):
        validate_normalization_identity(records[:1], splits[0], path, mode='per_subject')


def test_legacy_statistics_explicitly_warn_about_unverified_fit(tmp_path):
    _, records = _bundle(tmp_path)
    splits = write_block_cv_splits(records, tmp_path / 'blocks', n_folds=2, seed=42)
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({r.subject_uid: {'mean': [0, 0], 'std': [1, 1]} for r in records}))
    with pytest.warns(RuntimeWarning, match='provenance cannot be verified'):
        validate_normalization_identity(records, splits[0], path, mode='per_subject')
