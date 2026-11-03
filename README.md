# Embedded FOC Motor Control & TinyML Predictive Maintenance Engine

A zero-dependency C99 implementation of a **Field-Oriented Control (FOC)** motor drive loop paired with an embedded **TinyML** neural network for real-time mechanical and electrical fault classification.

Designed as a deterministic host-side Hardware-in-the-Loop (HIL) simulation, this project validates the complete signal chain—from current/vibration sampling and Space Vector PWM (SVPWM) modulation to DSP feature extraction, neural network inference, and JSON telemetry serialization—prior to target MCU deployment.

---

## System Architecture

1. **1 kHz Fast Control Loop (`tim1_1khz_isr_step`)**
   * Executes at a deterministic step of `dt = 1.0 ms` (1 kHz).
   * Performs Clarke and Park transforms (`ia`, `ib` to `id`, `iq`), dual PI current regulation, and Inverse Park transformation (`valpha`, `vbeta`).
   * Computes 3-phase inverter duty cycles (`duty_a`, `duty_b`, `duty_c`) via **Min-Max Zero-Sequence Injection SVPWM**.
   * Uses a bare-metal `Xorshift32` PRNG for deterministic white-noise injection without relying on `stdlib` heap or `rand()`.

2. **Ping-Pong Double Buffering & Task Notification**
   * Samples 64 points per frame (`64 samples * 1.0 ms = 64 ms` acquisition window) into static ping-pong buffers (`dma_ping_pong_buf[2]`).
   * On window completion, swaps the active write index (`active_write_buf ^= 1`) and asserts `rtos_task_notify_flag` to unblock the background inference task without halting sample acquisition.

3. **DSP Feature Extraction & TinyML Classifier**
   * Extracts 6 time-domain features per 64 ms window:
     * Vibration RMS (`g`), Peak (`g`), Crest Factor, and Kurtosis
     * Quadrature Current (`Iq`) Mean (`A`) and Peak-to-Peak Ripple (`A`)
   * Runs z-score normalization followed by a compact Multi-Layer Perceptron (`6 -> 12 ReLU -> 4 Softmax`) to classify four operating conditions:
     * `NORMAL_OPERATION`
     * `BEARING_FLAKING_WEAR`
     * `ROTOR_IMBALANCE`
     * `OVERLOAD_TORQUE_STALL`

4. **UART JSON Telemetry & Latency Profiling**
   * Measures DSP + inference execution latency using high-resolution performance counters (`QueryPerformanceCounter` on Windows / `clock()` on POSIX).
   * Streams structured diagnostic telemetry (`UART_TX`) containing fault class, confidence, key features, SVPWM duty cycles, and measured execution time in milliseconds.

---

## Repository Structure

    .
    ├── include/
    │   └── foc_dsp_engine.h              # FOC transforms, PI controller, DSP & TinyML weights
    ├── ml_pipeline/
    │   └── train_export_tinyml.py        # Synthetic data generator, MLP trainer & C header exporter
    ├── src/
    │   └── main.c                        # 1 kHz ISR simulation, SVPWM, ping-pong buffer & UART telemetry
    └── README.md

---

## Build & Run Instructions

### 1. Compile and Run the C99 Simulation
Requires `gcc` (MinGW-w64 on Windows or standard GCC on Linux/macOS):

    gcc -std=c99 -Wall -Wextra -Iinclude src/main.c -o foc_tinyml_sim -lm
    ./foc_tinyml_sim

### 2. Sample Output

    === D:\PROJECTS\Embedded RTOS FOC + TINYML PREDICTIVE ENGINE ===
    Config: 1 kHz ISR (dt = 1.0 ms) | Window = 64 samples (64 ms)

    [Injected: NORMAL_OPERATION     ] -> Predicted: NORMAL_OPERATION      (Conf: 100.0%) | Vib_RMS: 0.70g | Crest: 1.54 | Iq: 2.00A | Latency: 0.0042ms
      UART_TX -> {"fault":"NORMAL_OPERATION","conf":1.000,"vib_rms":0.70,"iq_mean":2.00,"svpwm":[0.99,0.01,0.38],"lat_ms":0.0042}
    [Injected: BEARING_FLAKING_WEAR ] -> Predicted: BEARING_FLAKING_WEAR  (Conf:  91.2%) | Vib_RMS: 2.71g | Crest: 2.83 | Iq: 2.30A | Latency: 0.0031ms
      UART_TX -> {"fault":"BEARING_FLAKING_WEAR","conf":0.912,"vib_rms":2.71,"iq_mean":2.30,"svpwm":[0.99,0.01,0.68],"lat_ms":0.0031}
    [Injected: ROTOR_IMBALANCE      ] -> Predicted: ROTOR_IMBALANCE       (Conf:  99.9%) | Vib_RMS: 3.98g | Crest: 1.43 | Iq: 2.81A | Latency: 0.0029ms
      UART_TX -> {"fault":"ROTOR_IMBALANCE","conf":0.999,"vib_rms":3.98,"iq_mean":2.81,"svpwm":[0.99,0.01,0.92],"lat_ms":0.0029}
    [Injected: OVERLOAD_TORQUE_STALL] -> Predicted: OVERLOAD_TORQUE_STALL (Conf:  99.9%) | Vib_RMS: 2.18g | Crest: 1.46 | Iq: 6.49A | Latency: 0.0028ms
      UART_TX -> {"fault":"OVERLOAD_TORQUE_STALL","conf":0.999,"vib_rms":2.18,"iq_mean":6.49,"svpwm":[0.95,0.05,0.99],"lat_ms":0.0028}

    === Execution Complete ===

### 3. Retrain and Export Weights (Optional)

    python ml_pipeline/train_export_tinyml.py

---

## Engineering Notes & Scope

* **Host-Side Verification:** The sensor synthesizer (`sample_sensors_hil`) models parametric fault signatures matching the synthetic training distribution to verify deterministic C execution and numerical parity with Python training. Real-world deployment requires retraining on empirical accelerometer and shunt-current datasets with held-out validation splits.
* **Portability:** All dynamic memory allocation (`calloc`/`malloc`) has been eliminated in favor of static allocation (`SensorFrame_t dma_ping_pong_buf[2]`) to guarantee deterministic memory usage on Cortex-M targets.
