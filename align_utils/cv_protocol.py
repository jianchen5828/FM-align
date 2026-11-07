"""Outer patient-level CV with training-only validation for model selection.

No model or dataset is loaded here. The outer test indices are never passed to
checkpoint selection. Split manifests are saved before model training.
"""
import json
import math
from pathlib import Path

PROTOCOL_VERSION = "outer-test-inner-validation-fixed-epochs-v1"


def validate_three_way_split(train_indices, validation_indices, test_indices, n_samples):
    groups = [list(map(int, x)) for x in (train_indices, validation_indices, test_indices)]
    sets = list(map(set, groups))
    if any(not x for x in groups):
        raise ValueError("training, internal validation and outer test must all be nonempty")
    if any(len(a) != len(b) for a, b in zip(groups, sets)):
        raise ValueError("duplicate indices in a split")
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("training, internal validation and outer test must be disjoint")
    if set.union(*sets) != set(range(n_samples)):
        raise ValueError("three-way split must cover the complete patient cohort")


def make_inner_split(development_indices, test_indices, labels, seed, fold, validation_fraction=0.2):
    """Split only the outer development patients, stratified by class label.

    Twenty percent refers to the outer development set, not the full cohort.
    Fail on insufficient class counts; never retry using test performance or
    silently fall back to a non-stratified/data-dropping split.
    """
    from sklearn.model_selection import train_test_split

    development = list(map(int, development_indices))
    inner_seed = int(seed) + 1009 * (int(fold) + 1)
    train, validation = train_test_split(
        development, test_size=validation_fraction, random_state=inner_seed,
        stratify=[labels[i] for i in development],
    )
    train, validation = list(map(int, train)), list(map(int, validation))
    validate_three_way_split(train, validation, test_indices, len(labels))
    return train, validation, inner_seed


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record_fold_split(path, patient_ids, train, validation, test, seed, fold, inner_seed,
                      validation_fraction, monitor, max_epochs):
    patient_ids = list(map(str, patient_ids))
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("duplicate patient IDs; refusing patient-overlapping CV")
    validate_three_way_split(train, validation, test, len(patient_ids))
    payload = {
        "protocol": PROTOCOL_VERSION, "seed": int(seed), "fold": int(fold),
        "inner_seed": int(inner_seed), "patient_order": patient_ids,
        "internal_validation_fraction": validation_fraction,
        "checkpoint_monitor": monitor, "max_epochs": max_epochs, "early_stopping": False,
    }
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        payload[name + "_indices"] = list(map(int, indices))
        payload[name + "_patients"] = [patient_ids[i] for i in indices]
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("existing split manifest differs; use a new output directory")
    save_json(path, payload)
    return payload


class BestCheckpoint:
    """Select on INTERNAL validation; never stop the fixed-length training loop."""
    def __init__(self, allow_ties=False):
        self.allow_ties = bool(allow_ties)
        self.best_score = None
        self.best_epoch = None

    def update(self, validation_score, epoch):
        score = float(validation_score)
        if not math.isfinite(score):
            raise ValueError("internal-validation metric is undefined/nonfinite")
        improved = (self.best_score is None or score > self.best_score
                    or (self.allow_ties and score == self.best_score))
        if improved:
            self.best_score, self.best_epoch = score, int(epoch)
        return improved
