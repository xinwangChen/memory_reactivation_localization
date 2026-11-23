"""Content identities for the external EEG and behavior files in a manifest.

Schema 1 lives inside artifact_sources.json's manifest_csv entry. No data or
checkpoint config keys change, so historical snapshots keep their identity.
"""

import csv
import hashlib
from pathlib import Path
import warnings


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _digest(path, stamp):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if _stamp(Path(path)) != stamp:
        raise ValueError(f"Source file changed while hashing: {path}")
    return digest.hexdigest()


def content_sha256(path):
    """Read contents at each verification boundary; stat values are not identities.

    Some filesystems can retain the same ctime for rapid writes. Even a cache
    keyed by inode/size/mtime/ctime must not substitute for a content check.
    """
    path = Path(path).expanduser().resolve()
    stamp = _stamp(path)
    result = _digest(str(path), stamp)
    if _stamp(path) != stamp:
        raise ValueError(f"Source file changed while hashing: {path}")
    return result


def stream_sha256(stream):
    """Hash the open input, including an inode no longer reachable at its path."""
    position = stream.tell()
    try:
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        stream.seek(position)


class SourceFileBinding:
    """Bind cached inputs to their original files without hashing per sample.

    Content checks belong at publication/reuse boundaries. Stat checks surround
    lazy reads so a store cannot mix a replacement file with cached metadata.
    Neither check updates the binding: changed inputs require a new dataset.
    """

    def __init__(self, paths):
        self.hashes = {}
        self.stamps = {}
        for value in sorted({str(p) for p in paths}):
            path = Path(value)
            self.stamps[value] = _stamp(path)
            self.hashes[value] = content_sha256(path)
            self.check_stamps([value])

    def check_stamps(self, paths=None):
        for value in self.stamps if paths is None else paths:
            value = str(value)
            if _stamp(Path(value)) != self.stamps[value]:
                raise ValueError(f"Source file changed since loading: {value}; recreate the dataset/input")

    def verify(self, current_hashes=None):
        for path, expected in self.hashes.items():
            actual = content_sha256(path) if current_hashes is None else current_hashes.get(path)
            if actual != expected:
                raise ValueError(f"Source file changed since loading: {path}; recreate the dataset/input")


def manifest_record_files(manifest_path):
    with Path(manifest_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not all(row.get(key) for key in ("subject_uid", "npz_path", "meta_path")) for row in rows):
        raise ValueError("Manifest must identify every record's EEG and metadata files")
    if len({row["subject_uid"] for row in rows}) != len(rows):
        raise ValueError("Manifest has duplicate subject_uid values")
    return [{"subject_uid": row["subject_uid"], **{
        key: str(Path(row[key]).expanduser().resolve()) for key in ("npz_path", "meta_path")}}
        for row in rows]


def snapshot_record_files(manifest_path):
    records = manifest_record_files(manifest_path)
    for record in records:
        for kind in ("npz", "meta"):
            record[kind + "_sha256"] = content_sha256(record[kind + "_path"])
    return records


def warn_legacy_source_files():
    warnings.warn("Legacy run has no EEG/metadata content fingerprints; raw-data provenance "
                  "cannot be verified. Existing artifacts are retained unchanged.", RuntimeWarning, stacklevel=3)


def validate_record_files(manifest_path, descriptor):
    """Old runs stay readable; new runs must match all frozen source contents."""
    if "record_files_schema" not in descriptor and "record_files" not in descriptor:
        warn_legacy_source_files()
        return
    records = descriptor.get("record_files")
    if descriptor.get("record_files_schema") != 1 or not isinstance(records, list):
        raise ValueError("Invalid snapshot record-file identity schema")
    current = manifest_record_files(manifest_path)
    if len(records) != len(current):
        raise ValueError("Snapshot source record count differs from manifest")
    for actual, saved in zip(current, records):
        if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in actual.items()):
            raise ValueError("Snapshot source record identity differs from manifest")
        for kind in ("npz", "meta"):
            if content_sha256(actual[kind + "_path"]) != saved.get(kind + "_sha256"):
                raise ValueError(f"Snapshot source file changed: {actual['subject_uid']} {kind} ({actual[kind + '_path']})")
