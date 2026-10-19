#include <stdio.h>
#include <stdlib.h>
#include "foc_dsp_engine.h"

static const char* FAULT_LABELS[4] = {
    "NORMAL_OPERATION",
    "BEARING_FLAKING_WEAR",
    "ROTOR_IMBALANCE",
    "OVERLOAD_TORQUE_STALL"
};

static SensorFrame_t dma_ping_pong_buf[2];
static volatile uint8_t active_write_buf = 0;
static volatile uint8_t ready_read_buf   = 0;
static volatile uint8_t injected_fault_mode = 0;

static void sample_sensors_hil(uint16_t sample_idx, float theta, float *ia, float *ib, float *vib_g) {
    float noise = ((float)(rand() % 200) - 100.0f) * 0.001f;
    float base_iq = 2.0f;

    if (injected_fault_mode == 0) {
        *vib_g = 1.0f * sinf(theta) + noise;
        base_iq = 2.0f + 0.05f * sinf(theta * 3.0f);
    } else if (injected_fault_mode == 1) {
        float spike = (sample_idx % 8 == 0) ? 7.6f : 0.4f;
        *vib_g = spike * ((sample_idx % 2 == 0) ? 1.0f : -1.0f) + noise;
        base_iq = 2.3f + 0.12f * sinf(theta * 6.0f);
    } else if (injected_fault_mode == 2) {
        *vib_g = 5.6f * sinf(theta) + noise;
        base_iq = 2.8f + 0.20f * sinf(theta);
    } else {
        *vib_g = 3.1f * sinf(theta * 2.0f) + noise;
        base_iq = 6.5f + 0.70f * sinf(theta * 4.0f);
    }

    *ia = -base_iq * sinf(theta);
    *ib =  base_iq * (0.5f * sinf(theta) + 0.866025f * cosf(theta));
}

int main(void) {
    FOC_State_t foc = { .id_ref = 0.0f, .iq_ref = 2.0f };
    PI_Controller_t pi_d = { .kp = 1.5f, .ki = 25.0f, .out_max = 12.0f };
    PI_Controller_t pi_q = { .kp = 1.5f, .ki = 25.0f, .out_max = 12.0f };
    float features[NN_INPUT_DIM];
    float probs[NN_OUTPUT_DIM];

    printf("=== D:\\PROJECTS\\Embedded RTOS FOC + TINYML PREDICTIVE ENGINE ===\n\n");

    for (uint8_t mode = 0; mode < 4; mode++) {
        injected_fault_mode = mode;

        for (uint16_t i = 0; i < SAMPLE_WINDOW_SIZE; i++) {
            foc.theta_elec += 0.18f;
            float vib = 0.0f;
            sample_sensors_hil(i, foc.theta_elec, &foc.ia, &foc.ib, &vib);
            foc_fast_loop_step(&foc, &pi_d, &pi_q, 0.001f);

            dma_ping_pong_buf[active_write_buf].vibration_g[i] = vib;
            dma_ping_pong_buf[active_write_buf].iq_current[i]  = foc.iq;
        }

        ready_read_buf = active_write_buf;
        active_write_buf ^= 1;

        extract_dsp_features(&dma_ping_pong_buf[ready_read_buf], features);
        uint8_t pred = tinyml_predict_fault(features, probs);

        printf("[Injected: %-21s] -> Predicted: %-21s (Conf: %5.1f%%) | Vib_RMS: %4.2fg | Crest: %4.2f | Iq: %4.2fA\n",
               FAULT_LABELS[mode], FAULT_LABELS[pred], probs[pred] * 100.0f,
               features[0], features[2], features[4]);
    }

    printf("\n=== Execution Complete ===\n");
    return 0;
}