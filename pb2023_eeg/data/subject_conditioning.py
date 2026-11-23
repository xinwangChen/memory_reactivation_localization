"""Resolve seen subjects from actual training examples, never validation membership."""
from collections.abc import Mapping, MutableMapping

from pb2023_eeg.models.subject_conditioning import (
    normalize_subject_conditioning_mode, normalize_training_subject_ids,
)


def _participant_names(dataset):
    return dataset.store.participant_names


def _subject_config(model_config: Mapping) -> Mapping:
    if not isinstance(model_config, Mapping):
        raise ValueError("model must be a mapping")
    subject = model_config.get("subject_conditioning")
    if subject is None:
        return {}
    if not isinstance(subject, Mapping):
        raise ValueError("model.subject_conditioning must be a mapping")
    return subject


def _training_names(subject: Mapping) -> dict[str, str]:
    ids = normalize_training_subject_ids(subject.get("resolved_training_subject_ids"))
    names = subject.get("resolved_training_subject_names")
    if (not isinstance(names, Mapping) or set(names) != {str(sid) for sid in ids}
            or not all(isinstance(name, str) and name for name in names.values())
            or len(set(names.values())) != len(names)):
        raise ValueError("resolved_training_subject_names must uniquely match resolved_training_subject_ids")
    return dict(names)


def _validate_namespace(names: Mapping[str, str], datasets: Mapping, message: str) -> None:
    name_to_id = {name: int(sid) for sid, name in names.items()}
    for dataset in datasets.values():
        for sid in set(map(int, dataset.participant_indices)):
            name = str(_participant_names(dataset)[sid])
            # Check both directions: a trained ID cannot name another person, and
            # a trained person cannot silently move to a formerly unseen ID.
            if ((str(sid) in names and names[str(sid)] != name)
                    or (name in name_to_id and name_to_id[name] != sid)):
                raise ValueError(f"{message}: ID {sid} refers to {name!r}; preserve the checkpoint subject mapping")


def validate_pretrained_subject_mapping(saved_config: Mapping, expected_config: Mapping | None) -> None:
    """Require target identity context before transferring shared/individual components."""
    if expected_config is None:
        raise ValueError("expected_config is required to verify pretrained shared subject identities")
    mappings = []
    for role, config in [("checkpoint", saved_config), ("target", expected_config)]:
        if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
            raise ValueError(f"Pretrained DPC {role} model config must be a mapping")
        subject = _subject_config(config["model"])
        if subject.get("mode", "individual") != "shared_plus_residual":
            raise ValueError("Pretrained DPC subject conditioning modes differ")
        mappings.append(_training_names(subject))
    if mappings[0] != mappings[1]:
        raise ValueError("Pretrained DPC training subject names differ from the target subject mapping")


def configure_training_subject_conditioning(model_config: MutableMapping, datasets: Mapping) -> int | None:
    """Save the training-ID namespace and return its required embedding capacity.

    ``None`` preserves the historical capacity resolution for individual mode.
    The namespace must stay consistent when building inference datasets; a genuinely
    unknown identity can also be passed explicitly as -1 or subject_id=None.
    """
    raw = _subject_config(model_config)
    mode = normalize_subject_conditioning_mode(raw.get("mode", "individual"))
    if mode == "individual":
        return None
    active = bool(raw.get("enabled", False)) or model_config.get("attention_mode") in {"query", "gated_query"}
    if not active:
        return None
    train = datasets["train"]
    examples = getattr(train, "dynamic_choice_pool", ()) or train.examples
    seen = sorted({int(train.participant_indices[example.subject_index]) for example in examples})
    if not seen:
        raise ValueError("shared_plus_residual requires actual training examples")
    names = {str(sid): str(_participant_names(train)[sid]) for sid in seen}
    subject_config = dict(raw)
    for key, value in [("resolved_training_subject_ids", seen), ("resolved_training_subject_names", names)]:
        if key in subject_config and subject_config[key] != value:
            raise ValueError(f"{key} differs from the actual training subject mapping")
        subject_config[key] = value
    # Each split must use the same ID namespace, including in-range unseen subjects.
    _validate_namespace(names, datasets, "Subject ID namespaces differ across splits")
    model_config["subject_conditioning"] = subject_config
    return max(seen) + 1


def validate_conditioning_subject_namespace(model_config: Mapping, datasets: Mapping) -> None:
    """Reject inference datasets that silently reuse a trained ID for another person."""
    subject_config = _subject_config(model_config)
    mode = normalize_subject_conditioning_mode(subject_config.get("mode", "individual"))
    if mode != "shared_plus_residual":
        return
    if not (bool(subject_config.get("enabled", False)) or model_config.get("attention_mode") in {"query", "gated_query"}):
        return
    names = _training_names(subject_config)
    _validate_namespace(names, datasets, "Inference subject ID namespace differs from checkpoint")
