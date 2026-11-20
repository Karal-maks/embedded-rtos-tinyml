CC      ?= gcc
CFLAGS  ?= -std=c99 -Wall -Wextra -Wpedantic -Werror -O2
PYTHON  ?= python3
TARGET  := foc_tinyml_sim

.PHONY: all run test train check-weights clean

all: $(TARGET)

$(TARGET): src/main.c include/foc_dsp_engine.h include/tinyml_weights.h
	$(CC) $(CFLAGS) -Iinclude src/main.c -o $@ -lm

run: $(TARGET)
	./$(TARGET)

# Simulation must classify all four faults, C must match Python,
# and the committed weights must match a fresh training run.
test: $(TARGET) check-weights
	./$(TARGET)
	$(PYTHON) tests/test_parity.py

train:
	$(PYTHON) ml_pipeline/train_export_tinyml.py

check-weights:
	$(PYTHON) ml_pipeline/train_export_tinyml.py --check

clean:
	rm -f $(TARGET) $(TARGET).exe
