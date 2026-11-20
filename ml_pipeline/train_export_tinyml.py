"""Train the TinyML fault classifier and export it as a C header.

Training data is generated the same way the C simulator generates it: raw
64-sample vibration and Iq windows from the fault models in src/main.c
(with randomised amplitudes, phase and noise), passed through a NumPy port of
extract_dsp_features() from include/foc_dsp_engine.h. The model therefore
sees exactly the kind of features the C code produces.

Usage:
    python ml_pipeline/train_export_tinyml.py            # train + export header
    python ml_pipeline/train_export_tinyml.py --check    # train, fail if header differs
"""

import argparse
import os
import sys

import numpy as np

WINDOW = 64            # SAMPLE_WINDOW_SIZE
THETA_STEP = 0.18      # electrical angle step per 1 kHz ISR tick (rad)
LABELS = ["NORMAL", "BEARING", "IMBALANCE", "OVERLOAD"]
HEADER = os.path.join(os.path.dirname(__file__), "..", "include", "tinyml_weights.h")


# ---------------------------------------------------------------------------
# Signal model (mirrors sample_sensors_hil() in src/main.c)
# ---------------------------------------------------------------------------
def simulate_window(mode, rng):
    """Return (vibration_g[64], iq_current[64]) as float32 for one fault mode.

    The C simulator uses the nominal value of every parameter; here each one
    is varied by roughly +/-15 % so the classifier generalises around it.
    """
    j = lambda nominal, spread=0.15: nominal * rng.uniform(1 - spread, 1 + spread)
    idx = np.arange(WINDOW)
    theta = rng.uniform(0, 2 * np.pi) + THETA_STEP * (idx + 1)
    noise = rng.uniform(-0.1, 0.1, WINDOW) * rng.uniform(0.5, 3.0)

    if mode == 0:    # healthy: 1x rotation vibration, flat current
        vib = j(1.0) * np.sin(theta) + noise
        iq = j(2.0) + j(0.05, 0.5) * np.sin(3 * theta)
    elif mode == 1:  # bearing flaking: periodic impacts on an alternating carrier
        period = 8
        offset = rng.integers(0, period)
        spike = np.where((idx + offset) % period == 0, j(7.6), j(0.4))
        vib = spike * np.where(idx % 2 == 0, 1.0, -1.0) + noise
        iq = j(2.3) + j(0.12, 0.5) * np.sin(6 * theta)
    elif mode == 2:  # rotor imbalance: large 1x sinusoid
        vib = j(5.6) * np.sin(theta) + noise
        iq = j(2.8) + j(0.20, 0.5) * np.sin(theta)
    else:            # overload / stall: 2x vibration, high current with ripple
        vib = j(3.1) * np.sin(2 * theta) + noise
        iq = j(6.5) + j(0.70, 0.5) * np.sin(4 * theta)
    return vib.astype(np.float32), iq.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction (NumPy port of extract_dsp_features(), float32 math)
# ---------------------------------------------------------------------------
def extract_features(vib, iq):
    vib = np.asarray(vib, dtype=np.float32)
    iq = np.asarray(iq, dtype=np.float32)
    rms = np.sqrt(np.sum(vib * vib, dtype=np.float32) / np.float32(WINDOW))
    peak = np.max(np.abs(vib))
    crest = peak / rms if rms > 0.001 else np.float32(1.0)
    diff = np.diff(vib)
    hf = np.sqrt(np.sum(diff * diff, dtype=np.float32) / np.float32(WINDOW - 1))
    iq_mean = np.sum(iq, dtype=np.float32) / np.float32(WINDOW)
    ripple = np.max(iq) - np.min(iq)
    return np.array([rms, peak, crest, hf, iq_mean, ripple], dtype=np.float32)


def generate_dataset(samples_per_class=600, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for mode in range(4):
        for _ in range(samples_per_class):
            X.append(extract_features(*simulate_window(mode, rng)))
            y.append(mode)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train_mlp(X, y, hidden=12, epochs=1500, lr=0.05, seed=42):
    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    n, d = Xn.shape
    k = 4
    Y = np.eye(k)[y]

    W1 = (rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = (rng.standard_normal((hidden, k)) * np.sqrt(2.0 / hidden)).astype(np.float32)
    b2 = np.zeros(k, dtype=np.float32)

    for _ in range(epochs):
        z1 = Xn @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        e = np.exp(z2 - z2.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        dz2 = (p - Y) / n
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        W2 -= lr * (a1.T @ dz2)
        b2 -= lr * dz2.sum(axis=0)
        W1 -= lr * (Xn.T @ dz1)
        b1 -= lr * dz1.sum(axis=0)
    return {"mean": mean, "std": std, "W1": W1, "b1": b1, "W2": W2, "b2": b2}


def predict_proba(model, X):
    """Same math as tinyml_predict_fault() in C (float32)."""
    X = np.asarray(X, dtype=np.float32)
    h = np.maximum(0, ((X - model["mean"]) / model["std"]) @ model["W1"] + model["b1"])
    z = h @ model["W2"] + model["b2"]
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_header(path=HEADER):
    """Parse the exported C header back into a model dict."""
    import re
    text = open(path).read()
    arr = lambda name: np.array(
        [float(v.strip().rstrip("f")) for v in re.search(rf"{name}\[\d+\] = \{{(.*?)\}};", text, re.S).group(1).split(",")],
        dtype=np.float32,
    )
    d = int(re.search(r"NN_INPUT_DIM\s+(\d+)", text).group(1))
    h = int(re.search(r"NN_HIDDEN_DIM\s+(\d+)", text).group(1))
    k = int(re.search(r"NN_OUTPUT_DIM\s+(\d+)", text).group(1))
    return {
        "mean": arr("FEATURE_MEAN"), "std": arr("FEATURE_STD"),
        "W1": arr("NN_W1").reshape(d, h), "b1": arr("NN_B1"),
        "W2": arr("NN_W2").reshape(h, k), "b2": arr("NN_B2"),
    }


def header_text(model):
    out = ["#ifndef TINYML_WEIGHTS_H", "#define TINYML_WEIGHTS_H", "",
           "// Auto-generated by ml_pipeline/train_export_tinyml.py - do not edit by hand.",
           "// MLP 6 -> 12 (ReLU) -> 4 (softmax)",
           "#define NN_INPUT_DIM  6", "#define NN_HIDDEN_DIM 12", "#define NN_OUTPUT_DIM 4", ""]
    for name, key in [("FEATURE_MEAN", "mean"), ("FEATURE_STD", "std"), ("NN_W1", "W1"),
                      ("NN_B1", "b1"), ("NN_W2", "W2"), ("NN_B2", "b2")]:
        a = np.asarray(model[key], dtype=np.float32).flatten()
        out.append(f"static const float {name}[{a.size}] = {{ " + ", ".join(f"{v:.6f}f" for v in a) + " };")
        out.append("")
    out.append("#endif // TINYML_WEIGHTS_H")
    return "\n".join(out) + "\n"


def confusion_matrix(y_true, y_pred, k=4):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed header is out of date")
    args = ap.parse_args()

    X, y = generate_dataset()
    idx = np.random.default_rng(0).permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    print(f"Train samples: {len(tr)} | Test samples: {len(te)}")

    model = train_mlp(X[tr], y[tr])      # normalisation stats from training split only
    train_acc = (predict_proba(model, X[tr]).argmax(1) == y[tr]).mean() * 100
    pred = predict_proba(model, X[te]).argmax(1)
    test_acc = (pred == y[te]).mean() * 100
    print(f"Train accuracy: {train_acc:.2f}% | Test accuracy (held-out 20%): {test_acc:.2f}%")

    cm = confusion_matrix(y[te], pred)
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * 11 + "".join(f"{l:>11}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"{l:<11}" + "".join(f"{cm[i, j]:>11}" for j in range(4)))

    text = header_text(model)
    if args.check:
        if open(HEADER).read() != text:
            sys.exit("include/tinyml_weights.h is out of date: run without --check and commit it")
        print("\nHeader is up to date.")
    else:
        with open(HEADER, "w") as f:
            f.write(text)
        print(f"\nWrote {os.path.normpath(HEADER)}")


if __name__ == "__main__":
    main()
