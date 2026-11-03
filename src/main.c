#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "foc_dsp_engine.h"

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return ((double)count.QuadPart * 1000.0) / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_ms(void) {
    return ((double)clock() * 1000.0) / (double)CLOCKS_PER_SEC;
}
#endif

#define ISR_DT_SEC 0.001f
#define VDC_BUS    24.0f

typedef struct {
    float duty_a;
    float duty_b;
    float duty_c;
} SVPWM_Duty_t;

static const char* FAULT_LABELS[4] = {
    "NORMAL_OPERATION",
    "BEARING_FLAKING_WEAR",
    "ROTOR_IMBALANCE",
    "OVERLOAD_TORQUE_STALL"
};

/* Static double-buffered DMA ping-pong frames & RTOS task notification state */
static SensorFrame_t dma_ping_pong_buf[2];
static volatile uint8_t active_write_buf      = 0;
static volatile uint8_t ready_read_buf        = 0;
static volatile uint8_t rtos_task_notify_flag = 0;
static volatile uint16_t dma_sample_idx       = 0;
static volatile uint8_t injected_fault_mode   = 0;

/* Deterministic bare-metal PRNG (replaces stdlib rand) */
static uint32_t prng_state = 0xCAFEBABEu;
static float embedded_white_noise(void) {
    uint32_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng_state = x;
    return ((float)(x % 200u) - 100.0f) * 0.001f;
}

/* Space Vector PWM (SVPWM) via Min-Max zero-sequence injection */
static void svpwm_update_duties(float valpha, float vbeta, float vdc, SVPWM_Duty_t *pwm) {
    float va = valpha;
    float vb = -0.5f * valpha + 0.8660254f * vbeta;
    float vc = -0.5f * valpha - 0.8660254f * vbeta;

    float vmin = fminf(va, fminf(vb, vc));
    float vmax = fmaxf(va, fmaxf(vb, vc));
    float voff = -0.5f * (vmin + vmax);

    pwm->duty_a = fminf(1.0f, fmaxf(0.0f, 0.5f + (va + voff) / vdc));
    pwm->duty_b = fminf(1.0f, fmaxf(0.0f, 0.5f + (vb + voff) / vdc));
    pwm->duty_c = fminf(1.0f, fmaxf(0.0f, 0.5f + (vc + voff) / vdc));
}

/* Hardware-in-the-Loop (HIL) sensor synthesizer */
static void sample_sensors_hil(uint16_t sample_idx, float theta, float *ia, float *ib, float *vib_g) {
    float noise = embedded_white_noise();
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
    *ib =  base_iq * (0.5f * sinf(theta) + 0.8660254f * cosf(theta));
}

/* 1 kHz Deterministic Timer ISR + DMA Ping-Pong Buffer Manager */
static void tim1_1khz_isr_step(FOC_State_t *foc, PI_Controller_t *pi_d,
                               PI_Controller_t *pi_q, SVPWM_Duty_t *pwm) {
    float vib = 0.0f;
    foc->theta_elec += 0.18f;

    sample_sensors_hil(dma_sample_idx, foc->theta_elec, &foc->ia, &foc->ib, &vib);
    foc_fast_loop_step(foc, pi_d, pi_q, ISR_DT_SEC);
    svpwm_update_duties(foc->valpha, foc->vbeta, VDC_BUS, pwm);

    dma_ping_pong_buf[active_write_buf].vibration_g[dma_sample_idx] = vib;
    dma_ping_pong_buf[active_write_buf].iq_current[dma_sample_idx]  = foc->iq;
    dma_sample_idx++;

    if (dma_sample_idx >= SAMPLE_WINDOW_SIZE) {
        dma_sample_idx        = 0;
        ready_read_buf        = active_write_buf;
        active_write_buf     ^= 1u;
        rtos_task_notify_flag = 1u; /* Unblock TinyML inference task */
    }
}

/* Simulated UART JSON Telemetry Output */
static void uart_send_json_telemetry(uint8_t mode, uint8_t pred, float conf,
                                     const float *features, const SVPWM_Duty_t *pwm,
                                     double latency_ms) {
    printf("[Injected: %-21s] -> Predicted: %-21s (Conf: %5.1f%%) | "
           "Vib_RMS: %4.2fg | Crest: %4.2f | Iq: %4.2fA | Latency: %.4fms\n",
           FAULT_LABELS[mode], FAULT_LABELS[pred], conf * 100.0f,
           features[0], features[2], features[4], latency_ms);

    printf("  UART_TX -> {\"fault\":\"%s\",\"conf\":%.3f,\"vib_rms\":%.2f,"
           "\"iq_mean\":%.2f,\"svpwm\":[%.2f,%.2f,%.2f],\"lat_ms\":%.4f}\n",
           FAULT_LABELS[pred], conf, features[0], features[4],
           pwm->duty_a, pwm->duty_b, pwm->duty_c, latency_ms);
}

int main(void) {
    FOC_State_t foc      = { .id_ref = 0.0f, .iq_ref = 2.0f };
    PI_Controller_t pi_d = { .kp = 1.5f, .ki = 25.0f, .out_max = 12.0f };
    PI_Controller_t pi_q = { .kp = 1.5f, .ki = 25.0f, .out_max = 12.0f };
    SVPWM_Duty_t pwm     = { 0.5f, 0.5f, 0.5f };
    float features[NN_INPUT_DIM];
    float probs[NN_OUTPUT_DIM];

    printf("=== D:\\PROJECTS\\Embedded RTOS FOC + TINYML PREDICTIVE ENGINE ===\n");
    printf("Config: 1 kHz ISR (dt = 1.0 ms) | Window = %u samples (64 ms)\n\n",
           (unsigned int)SAMPLE_WINDOW_SIZE);

    for (uint8_t mode = 0; mode < 4; mode++) {
        injected_fault_mode = mode;

        /* Run 1 kHz ISR until DMA ping-pong buffer triggers RTOS task notify */
        while (!rtos_task_notify_flag) {
            tim1_1khz_isr_step(&foc, &pi_d, &pi_q, &pwm);
        }
        rtos_task_notify_flag = 0;

        /* Measure TinyML DSP + Inference execution latency */
        double t_start = get_time_ms();
        extract_dsp_features(&dma_ping_pong_buf[ready_read_buf], features);
        uint8_t pred = tinyml_predict_fault(features, probs);
        double latency_ms = get_time_ms() - t_start;

        uart_send_json_telemetry(mode, pred, probs[pred], features, &pwm, latency_ms);
    }

    printf("\n=== Execution Complete ===\n");
    return 0;
}
