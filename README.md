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
   * Measures DSP + inference execution latency, averaged over 1000 runs, with `QueryPerformanceCounter` on Windows and `clock_gettime(CLOCK_MONOTONIC)` on POSIX.
   * Streams structured diagnostic telemetry (`UART_TX`) containing fault class, confidence, key features, SVPWM duty cycles, and measured execution time in milliseconds.

---

## Repository Structure

    .
    |-- .gitignore
    |-- Makefile
    |-- requirements.txt
    |-- include/
    |   |-- foc_dsp_engine.h    # FOC transforms, PI controller, DSP feature extraction
    |   `-- tinyml_weights.h    # Auto-generated MLP weights and normalization constants
    |-- ml_pipeline/
    |   `-- train_export_tinyml.py   # Waveform simulation, feature extraction, MLP training, C header export
    |-- src/
    |   `-- main.c              # 1 kHz ISR simulation, SVPWM, ping-pong buffer, UART telemetry
    |-- tests/
    |   |-- parity_harness.c    # Runs the C DSP + inference on windows from stdin
    |   `-- test_parity.py      # C vs Python parity test
    `-- README.md

---

## Build & Run Instructions

Requires `gcc` or `clang` (MinGW-w64 on Windows) and, for training and tests, Python 3 with NumPy.

    make run        # build and run the simulation
    make test       # simulation + C/Python parity test + weights-up-to-date check
    make train      # retrain the model and regenerate include/tinyml_weights.h

Without `make`:

    gcc -std=c99 -Wall -Wextra -O2 -Iinclude src/main.c -o foc_tinyml_sim -lm
    ./foc_tinyml_sim

The program exits with status 1 if any injected fault is misclassified, so CI catches regressions.

### Sample Output

    === FOC + TinyML Predictive Maintenance Engine (Host-Side Simulation) ===
    Config: 1 kHz ISR (dt = 1.0 ms) | Window = 64 samples (64 ms)

    [Injected: NORMAL_OPERATION     ] -> Predicted: NORMAL_OPERATION      (Conf:  99.8%) | Vib_RMS: 0.72g | Crest: 1.52 | Iq: 2.00A | Latency: 0.17us
      UART_TX -> {"fault":"NORMAL_OPERATION","conf":0.998,"vib_rms":0.72,"iq_mean":2.00,"svpwm":[0.50,0.50,0.50],"lat_us":0.169}
    [Injected: BEARING_FLAKING_WEAR ] -> Predicted: BEARING_FLAKING_WEAR  (Conf:  99.9%) | Vib_RMS: 2.72g | Crest: 2.82 | Iq: 2.30A | Latency: 0.21us
      UART_TX -> {"fault":"BEARING_FLAKING_WEAR","conf":0.999,"vib_rms":2.72,"iq_mean":2.30,"svpwm":[0.47,0.53,0.50],"lat_us":0.214}
    [Injected: ROTOR_IMBALANCE      ] -> Predicted: ROTOR_IMBALANCE       (Conf:  99.9%) | Vib_RMS: 4.00g | Crest: 1.41 | Iq: 2.81A | Latency: 0.22us
      UART_TX -> {"fault":"ROTOR_IMBALANCE","conf":0.999,"vib_rms":4.00,"iq_mean":2.81,"svpwm":[0.50,0.61,0.39],"lat_us":0.216}
    [Injected: OVERLOAD_TORQUE_STALL] -> Predicted: OVERLOAD_TORQUE_STALL (Conf:  99.9%) | Vib_RMS: 2.18g | Crest: 1.45 | Iq: 6.53A | Latency: 0.17us
      UART_TX -> {"fault":"OVERLOAD_TORQUE_STALL","conf":0.999,"vib_rms":2.18,"iq_mean":6.53,"svpwm":[0.93,0.50,0.07],"lat_us":0.167}

    === Execution Complete: 4/4 correct ===

Latency is the average of 1000 runs of DSP + inference on a desktop PC (`-O2`),
measured with `QueryPerformanceCounter` on Windows and `clock_gettime(CLOCK_MONOTONIC)` elsewhere.
It is not an MCU figure.

---

## Limitations
- Sensor data is synthetic; there is no motor/plant model, so the FOC loop is not closed on real dynamics.
- The "RTOS task notification" is simulated with a flag. On target, this would use FreeRTOS task notifications from the ISR.
- The four fault classes are well separated by construction, so 100% accuracy on synthetic data says the pipeline works, not that it would work on a real motor.
- Latency is measured on a PC, not on an MCU.
- Control rate is 1 kHz for simulation; production FOC typically runs at 10-20 kHz.

## Engineering Notes & Scope

* **Training matches the C code:** the Python pipeline simulates the same raw 64-sample waveforms as `sample_sensors_hil` (with randomised amplitudes, phase and noise) and runs a NumPy port of `extract_dsp_features`, so the model is trained on exactly the features the C code computes.
* **C/Python parity test:** `tests/test_parity.py` compiles `tests/parity_harness.c`, feeds it 1000 fresh windows, and requires features and probabilities to match Python within 1e-4 (observed: ~6e-7). It fails on changes as small as a wrong normalisation divisor.
* **Real-world deployment** would still require retraining on recorded accelerometer and shunt-current data.
* **Portability:** All dynamic memory allocation (`calloc`/`malloc`) has been eliminated in favor of static allocation (`SensorFrame_t dma_ping_pong_buf[2]`) to guarantee deterministic memory usage on Cortex-M targets.

## Model Evaluation

- Data: 600 simulated windows per class (2400 total), features computed exactly as in C.
- Train/test split: 1920 training, 480 held-out (20%). Normalisation statistics come from the training split only.
- Held-out test accuracy: 100%. A separate set of 1000 windows generated with a different seed, run through the C code, is also classified 100% correctly.
- Training is deterministic: `make check-weights` retrains and fails if the committed header differs.
