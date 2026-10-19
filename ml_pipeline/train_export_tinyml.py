import os
import numpy as np

np.random.seed(42)

def generate_motor_dataset(samples_per_class=600):
    X, y = [], []

    # Class 0: Healthy Motor
    for _ in range(samples_per_class):
        rms = np.random.normal(1.0, 0.15)
        peak = rms * np.random.normal(1.414, 0.1)
        X.append([rms, peak, peak/rms, np.random.normal(0.5, 0.1), np.random.normal(2.0, 0.2), np.random.normal(0.15, 0.03)])
        y.append(0)

    # Class 1: Bearing Wear (Spiky impacts -> High Crest Factor & HF Energy)
    for _ in range(samples_per_class):
        rms = np.random.normal(1.8, 0.25)
        cf = np.random.normal(4.2, 0.4)
        peak = rms * cf
        X.append([rms, peak, cf, np.random.normal(3.5, 0.4), np.random.normal(2.3, 0.25), np.random.normal(0.30, 0.05)])
        y.append(1)

    # Class 2: Rotor Imbalance (Sinusoidal wobble -> High RMS, low Crest Factor)
    for _ in range(samples_per_class):
        rms = np.random.normal(4.0, 0.4)
        cf = np.random.normal(1.5, 0.15)
        peak = rms * cf
        X.append([rms, peak, cf, np.random.normal(1.1, 0.2), np.random.normal(2.8, 0.3), np.random.normal(0.45, 0.08)])
        y.append(2)

    # Class 3: Overload / Stall (Extreme Iq Current & High Torque Ripple)
    for _ in range(samples_per_class):
        rms = np.random.normal(2.2, 0.3)
        cf = np.random.normal(2.0, 0.2)
        peak = rms * cf
        X.append([rms, peak, cf, np.random.normal(1.5, 0.25), np.random.normal(6.5, 0.5), np.random.normal(1.4, 0.2)])
        y.append(3)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_tinyml_mlp(X, y, epochs=1500, lr=0.05):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    n_samples, n_features = X_norm.shape
    n_hidden = 12
    n_classes = 4
    Y_onehot = np.eye(n_classes)[y]

    W1 = np.random.randn(n_features, n_hidden).astype(np.float32) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(n_hidden, dtype=np.float32)
    W2 = np.random.randn(n_hidden, n_classes).astype(np.float32) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros(n_classes, dtype=np.float32)

    for epoch in range(epochs):
        z1 = X_norm @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        dz2 = (probs - Y_onehot) / n_samples
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_norm.T @ dz1
        db1 = np.sum(dz1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y) * 100.0
    print(f"[TinyML Trainer] Training Complete | Final Accuracy: {acc:.2f}%")
    return mean, std, W1, b1, W2, b2

def export_to_c_header(mean, std, W1, b1, W2, b2, output_header="include/tinyml_weights.h"):
    os.makedirs(os.path.dirname(output_header), exist_ok=True)
    with open(output_header, "w") as f:
        f.write("#ifndef TINYML_WEIGHTS_H\n#define TINYML_WEIGHTS_H\n\n")
        f.write("// Auto-generated TinyML Neural Network Weights (6 -> 12 -> 4)\n")
        f.write("#define NN_INPUT_DIM  6\n#define NN_HIDDEN_DIM 12\n#define NN_OUTPUT_DIM 4\n\n")

        def write_array(name, arr):
            flat = ", ".join(f"{val:.6f}f" for val in arr.flatten())
            f.write(f"static const float {name}[{arr.size}] = {{ {flat} }};\n\n")

        write_array("FEATURE_MEAN", mean)
        write_array("FEATURE_STD", std)
        write_array("NN_W1", W1)
        write_array("NN_B1", b1)
        write_array("NN_W2", W2)
        write_array("NN_B2", b2)
        f.write("#endif // TINYML_WEIGHTS_H\n")
    print(f"[TinyML Exporter] Generated: {os.path.abspath(output_header)}")

if __name__ == "__main__":
    X, y = generate_motor_dataset()
    mean, std, W1, b1, W2, b2 = train_tinyml_mlp(X, y)
    export_to_c_header(mean, std, W1, b1, W2, b2)