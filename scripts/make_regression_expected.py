"""(Re)generate the regression test expectations. Run only after an *intended* change.

    uv run python scripts/make_regression_expected.py --synthetic --reason "first version"
    uv run python scripts/make_regression_expected.py --model --reason "first version"

--synthetic  writes tests/regression/expected_synthetic.json (committed; checked in CI)
--model      writes artifacts/regression/expected_int8_samples.json (local only: it is derived
             from RSNA images, which must not be published)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tests"), str(REPO / "tests" / "regression")]

# These imports need the sys.path entries above.
from brainct.constants import LABELS  # noqa: E402
from brainct.serving.model import InferenceModel  # noqa: E402
from dicom_factory import make_dicom_bytes  # noqa: E402
from synthetic_cases import EXPECTED_PATH, fingerprints  # noqa: E402

MODEL = REPO / "artifacts/resnet50_baseline/onnx/model_int8.onnx"
THRESHOLDS = REPO / "artifacts/resnet50_baseline/metrics_val.json"
MODEL_EXPECTED = REPO / "artifacts/regression/expected_int8_samples.json"
SAMPLE_DIR = Path("samples/dicom")
N_SAMPLES = 50


def write_synthetic(reason: str) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": reason,
        "cases": fingerprints(make_dicom_bytes),
    }
    EXPECTED_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EXPECTED_PATH.relative_to(REPO)} ({len(payload['cases'])} cases)")


def write_model(reason: str) -> None:
    model = InferenceModel(MODEL, THRESHOLDS, "regression", threads=4)
    files = sorted((REPO / SAMPLE_DIR).glob("*.dcm"))[:N_SAMPLES]
    if not files:
        raise SystemExit(f"no samples in {SAMPLE_DIR}")
    probabilities = {}
    for path in files:
        probs = model.predict_batch(model.prepare(path.read_bytes()).x[None])[0]
        probabilities[path.name] = {label: float(p) for label, p in zip(LABELS, probs, strict=True)}
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": reason,
        "model_version": model.version,
        "sample_dir": str(SAMPLE_DIR),
        "probabilities": probabilities,
    }
    MODEL_EXPECTED.parent.mkdir(parents=True, exist_ok=True)
    MODEL_EXPECTED.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {MODEL_EXPECTED.relative_to(REPO)} ({len(files)} samples, model {model.version})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--reason", required=True, help="why the expectations are (re)generated")
    args = parser.parse_args()
    if not (args.synthetic or args.model):
        parser.error("choose --synthetic and/or --model")
    if args.synthetic:
        write_synthetic(args.reason)
    if args.model:
        write_model(args.reason)


if __name__ == "__main__":
    main()
