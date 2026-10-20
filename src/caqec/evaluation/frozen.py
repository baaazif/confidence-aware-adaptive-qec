import json
from pathlib import Path

from caqec.decoders.sklearn_decoder import SklearnDecoder
from caqec.evaluation.protocol import file_sha256


def load_frozen_models(models_dir, exp04_results_path, kinds=("logreg", "tree")):
    """Load saved models and check them against the selection recorded in stage 2."""
    selected = json.loads(Path(exp04_results_path).read_text())["selected_1se"]
    models, hashes = {}, {}
    for kind in kinds:
        base = Path(models_dir) / f"exp04_{kind}"
        m = SklearnDecoder.load(base)
        s = selected[kind]
        if (m.kind, m.feature_set, m.params) != (s["kind"], s["feature_set"], s["params"]):
            raise ValueError(f"saved {kind} model does not match the selection recorded in stage 2")
        models[kind] = m
        hashes[kind] = file_sha256(base.with_suffix(".joblib"))
    return models, hashes