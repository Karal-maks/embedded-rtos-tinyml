# FOC Motor Control + TinyML Fault Classifier (Host-Side HIL Simulation)

C99 simulation of a Field-Oriented Control (FOC) loop with an embedded MLP
that classifies four motor fault conditions. Runs on a PC, with the
structure designed to be portable to a Cortex-M target.
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
     * Vibration RMS (`g`), Peak (`g`), Crest Factor, and High-frequency energy
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
    |-- .gitignore
    |-- include/
    |   |-- foc_dsp_engine.h    # FOC transforms, PI controller, DSP feature extraction
    |   `-- tinyml_weights.h    # Auto-generated MLP weights and normalization constants
    |-- ml_pipeline/
    |   `-- train_export_tinyml.py   # Synthetic data, MLP training, C header export
    |-- src/
    |   `-- main.c              # 1 kHz ISR simulation, SVPWM, ping-pong buffer, UART telemetry
    `-- README.md

---

## Build & Run Instructions

### 1. Compile and Run the C99 Simulation
Requires `gcc` (MinGW-w64 on Windows or standard GCC on Linux/macOS):

    gcc -std=c99 -Wall -Wextra -Iinclude src/main.c -o foc_tinyml_sim -lm
    ./foc_tinyml_sim

### 2. Sample Output

    === FOC + TinyML Predictive Maintenance Engine (Host-Side Simulation) ===
    Config: 1 kHz ISR (dt = 1.0 ms) | Window = 64 samples (64 ms)

    [Injected: NORMAL_OPERATION     ] -> Predicted: NORMAL_OPERATION      (Conf: 100.0%) | Vib_RMS: 0.72g | Crest: 1.52 | Iq: 2.00A | Latency: 0.0037ms
      UART_TX -> {"fault":"NORMAL_OPERATION","conf":1.000,"vib_rms":0.72,"iq_mean":2.00,"svpwm":[0.50,0.50,0.50],"lat_ms":0.0037}
    [Injected: BEARING_FLAKING_WEAR ] -> Predicted: BEARING_FLAKING_WEAR  (Conf:  90.4%) | Vib_RMS: 2.72g | Crest: 2.82 | Iq: 2.30A | Latency: 0.0019ms
      UART_TX -> {"fault":"BEARING_FLAKING_WEAR","conf":0.904,"vib_rms":2.72,"iq_mean":2.30,"svpwm":[0.47,0.53,0.50],"lat_ms":0.0019}
    [Injected: ROTOR_IMBALANCE      ] -> Predicted: ROTOR_IMBALANCE       (Conf:  99.9%) | Vib_RMS: 4.00g | Crest: 1.41 | Iq: 2.81A | Latency: 0.0015ms
      UART_TX -> {"fault":"ROTOR_IMBALANCE","conf":0.999,"vib_rms":4.00,"iq_mean":2.81,"svpwm":[0.50,0.61,0.39],"lat_ms":0.0015}
    [Injected: OVERLOAD_TORQUE_STALL] -> Predicted: OVERLOAD_TORQUE_STALL (Conf:  99.9%) | Vib_RMS: 2.18g | Crest: 1.45 | Iq: 6.53A | Latency: 0.0017ms
      UART_TX -> {"fault":"OVERLOAD_TORQUE_STALL","conf":0.999,"vib_rms":2.18,"iq_mean":6.53,"svpwm":[0.93,0.50,0.07],"lat_ms":0.0017}

    === Execution Complete ===

### 3. Retrain and Export Weights (Optional)

    python ml_pipeline/train_export_tinyml.py

---

## Limitations
- Sensor data is synthetic; there is no motor/plant model, so the FOC loop is not closed on real dynamics.
- The "RTOS task notification" is simulated with a flag. On target, this would use FreeRTOS task notifications from the ISR.
- Accuracy is measured on a held-out split of synthetic data (see Model Evaluation); real-world accuracy is untested.
- The C fault generator does not exactly match the Python training distribution (for example, bearing-wear crest factor is about 2.8 in C versus 4.2 in training), so C/Python numerical parity is not verified.
- Latency is measured on a PC, not on an MCU.
- Control rate is 1 kHz for simulation; production FOC typically runs at 10-20 kHz.

## Engineering Notes & Scope

* **Host-Side Verification:** The sensor synthesizer (`sample_sensors_hil`) models parametric fault signatures to exercise the deterministic C signal chain. These signatures only approximate the Python training distribution; an exact C/Python parity test is future work. Real-world deployment requires retraining on empirical accelerometer and shunt-current datasets with held-out validation splits.
* **Portability:** All dynamic memory allocation (`calloc`/`malloc`) has been eliminated in favor of static allocation (`SensorFrame_t dma_ping_pong_buf[2]`) to guarantee deterministic memory usage on Cortex-M targets.

## Model Evaluation

- Train/test split: 1920 training samples, 480 held-out test samples (20%).
- Test accuracy: 100% on the held-out synthetic set. The four classes are well separated by construction, so this shows the pipeline works, not real-world accuracy.
