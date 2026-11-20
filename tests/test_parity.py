"""C/Python parity test.

Builds tests/parity_harness.c, feeds it windows simulated by the Python
training pipeline, and checks that features and class probabilities from the
C code match the Python implementation with the exported weights.

Run:  python tests/test_parity.py   (or: make test)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ml_pipeline"))
import train_export_tinyml as tml  # noqa: E402

N_PER_CLASS = 250


def build_harness(out_dir):
    exe = os.path.join(out_dir, "parity_harness")
    subprocess.run(
        [os.environ.get("CC", "gcc"), "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
         "-I", os.path.join(ROOT, "include"), os.path.join(ROOT, "tests", "parity_harness.c"),
         "-o", exe, "-lm"],
        check=True,
    )
    return exe


def main():
    rng = np.random.default_rng(1234)  # different seed from training
    windows, labels = [], []
    for mode in range(4):
        for _ in range(N_PER_CLASS):
            windows.append(np.concatenate(tml.simulate_window(mode, rng)))
            labels.append(mode)
    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels)

    with tempfile.TemporaryDirectory() as tmp:
        exe = build_harness(tmp)
        out = subprocess.run([exe], input=windows.tobytes(), capture_output=True, check=True).stdout
    c_out = np.frombuffer(out, dtype=np.float32).reshape(len(windows), 10)
    c_feat, c_prob = c_out[:, :6], c_out[:, 6:]

    py_feat = np.array([tml.extract_features(w[:64], w[64:]) for w in windows])
    model = tml.load_header()
    py_prob = tml.predict_proba(model, py_feat)

    feat_err = np.max(np.abs(c_feat - py_feat) / (np.abs(py_feat) + 1e-3))
    prob_err = np.max(np.abs(c_prob - py_prob))
    agree = np.mean(c_prob.argmax(1) == py_prob.argmax(1)) * 100
    c_acc = np.mean(c_prob.argmax(1) == labels) * 100

    print(f"Windows: {len(windows)}")
    print(f"Max relative feature difference C vs Python: {feat_err:.2e}")
    print(f"Max probability difference C vs Python:      {prob_err:.2e}")
    print(f"C/Python prediction agreement: {agree:.2f}%")
    print(f"C accuracy on fresh simulated windows: {c_acc:.2f}%")

    ok = feat_err < 1e-4 and prob_err < 1e-4 and agree == 100.0
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
