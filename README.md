# Embedded RTOS + TinyML Motor Fault Detection

A C-based motor-control demo that combines a Field-Oriented Control (FOC) loop with lightweight signal processing and an embedded neural-network inference path. The included program simulates sensor readings and injected motor faults on a host computer; it is a proof of concept, not production RTOS firmware or a hardware driver.

## Features

- FOC current transforms and d/q-axis PI controllers.
- 64-sample vibration and q-axis current windows.
- DSP feature extraction: vibration RMS, peak, crest factor, high-frequency energy, mean q-axis current, and current ripple.
- A small 6–12–4 multilayer perceptron (ReLU hidden layer and softmax output) implemented in C.
- A Python script that generates synthetic training data, trains the model, and exports its parameters to a C header.
- A simulation that exercises four operating conditions:
  - Normal operation
  - Bearing flaking/wear
  - Rotor imbalance
  - Overload/torque stall

## Project layout

```text
include/
  foc_dsp_engine.h       FOC, DSP feature extraction, and TinyML inference
  tinyml_weights.h       Exported model weights and feature normalization values
ml_pipeline/
  train_export_tinyml.py Synthetic data generation, training, and C-header export
src/
  main.c                 Host-side motor and fault simulation
```

## Requirements

- A C compiler such as GCC
- Python 3
- NumPy (`python -m pip install numpy`) to train and export the model

## Build and run the simulation

From the repository root, compile with GCC:

```sh
gcc -std=c99 -Wall -Wextra -Iinclude src/main.c -o rtos_tinyml_sim -lm
```

On Windows, you can name the output `rtos_tinyml_sim.exe`:

```powershell
gcc -std=c99 -Wall -Wextra -Iinclude src/main.c -o rtos_tinyml_sim.exe -lm
.\rtos_tinyml_sim.exe
```

On Linux or macOS:

```sh
./rtos_tinyml_sim
```

The program prints the injected condition, predicted condition, confidence, and selected signal features for each simulation run.

## Retrain and export the model

From the repository root:

```sh
python -m pip install numpy
python ml_pipeline/train_export_tinyml.py
```

Training uses generated synthetic examples and writes the trained parameters and feature normalization values to `include/tinyml_weights.h`. Running the trainer replaces the current header. The generated data is for demonstration and does not establish real-world diagnostic accuracy.

## Limitations

- Sensor data is simulated; no physical motor, sensor, or RTOS is required or controlled.
- Training data is synthetic, so model results should not be treated as validated fault diagnosis.
- Hardware-specific sampling, peripheral drivers, scheduling, and safety protections are outside this demo's scope.
