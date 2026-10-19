#ifndef FOC_DSP_ENGINE_H
#define FOC_DSP_ENGINE_H

#include <stdint.h>
#include <math.h>
#include "tinyml_weights.h"

#define SAMPLE_WINDOW_SIZE 64
#define INV_SQRT3          0.57735026919f

typedef struct {
    float kp;
    float ki;
    float integral;
    float out_max;
} PI_Controller_t;

typedef struct {
    float ia, ib;
    float theta_elec;
    float id, iq;
    float id_ref, iq_ref;
    float vd, vq;
    float valpha, vbeta;
} FOC_State_t;

typedef struct {
    float vibration_g[SAMPLE_WINDOW_SIZE];
    float iq_current[SAMPLE_WINDOW_SIZE];
} SensorFrame_t;

static inline float run_pi_controller(PI_Controller_t *pi, float error, float dt) {
    pi->integral += error * pi->ki * dt;
    if (pi->integral > pi->out_max)  pi->integral = pi->out_max;
    if (pi->integral < -pi->out_max) pi->integral = -pi->out_max;
    
    float out = (error * pi->kp) + pi->integral;
    if (out > pi->out_max)  out = pi->out_max;
    if (out < -pi->out_max) out = -pi->out_max;
    return out;
}

static inline void foc_fast_loop_step(FOC_State_t *foc, PI_Controller_t *pi_d, PI_Controller_t *pi_q, float dt) {
    float ialpha = foc->ia;
    float ibeta  = (foc->ia + 2.0f * foc->ib) * INV_SQRT3;

    float c = cosf(foc->theta_elec);
    float s = sinf(foc->theta_elec);
    foc->id =  ialpha * c + ibeta * s;
    foc->iq = -ialpha * s + ibeta * c;

    foc->vd = run_pi_controller(pi_d, foc->id_ref - foc->id, dt);
    foc->vq = run_pi_controller(pi_q, foc->iq_ref - foc->iq, dt);

    foc->valpha = foc->vd * c - foc->vq * s;
    foc->vbeta  = foc->vd * s + foc->vq * c;
}

static inline void extract_dsp_features(const SensorFrame_t *frame, float features[NN_INPUT_DIM]) {
    float sum_sq_vib = 0.0f, max_vib = 0.0f, hf_diff_sq = 0.0f;
    float sum_iq = 0.0f, min_iq = 1e6f, max_iq = -1e6f;

    for (uint16_t i = 0; i < SAMPLE_WINDOW_SIZE; i++) {
        float v = frame->vibration_g[i];
        float abs_v = fabsf(v);
        sum_sq_vib += v * v;
        if (abs_v > max_vib) max_vib = abs_v;

        if (i > 0) {
            float diff = v - frame->vibration_g[i - 1];
            hf_diff_sq += diff * diff;
        }

        float iq = frame->iq_current[i];
        sum_iq += iq;
        if (iq < min_iq) min_iq = iq;
        if (iq > max_iq) max_iq = iq;
    }

    float rms_vib = sqrtf(sum_sq_vib / (float)SAMPLE_WINDOW_SIZE);
    float crest_factor = (rms_vib > 0.001f) ? (max_vib / rms_vib) : 1.0f;
    float hf_energy = sqrtf(hf_diff_sq / (float)(SAMPLE_WINDOW_SIZE - 1));
    float iq_mean = sum_iq / (float)SAMPLE_WINDOW_SIZE;
    float iq_ripple = max_iq - min_iq;

    features[0] = rms_vib;
    features[1] = max_vib;
    features[2] = crest_factor;
    features[3] = hf_energy;
    features[4] = iq_mean;
    features[5] = iq_ripple;
}

static inline uint8_t tinyml_predict_fault(const float raw_features[NN_INPUT_DIM], float out_probs[NN_OUTPUT_DIM]) {
    float norm_in[NN_INPUT_DIM];
    float hidden[NN_HIDDEN_DIM];

    for (int i = 0; i < NN_INPUT_DIM; i++) {
        norm_in[i] = (raw_features[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
    }

    for (int j = 0; j < NN_HIDDEN_DIM; j++) {
        float acc = NN_B1[j];
        for (int i = 0; i < NN_INPUT_DIM; i++) {
            acc += norm_in[i] * NN_W1[i * NN_HIDDEN_DIM + j];
        }
        hidden[j] = (acc > 0.0f) ? acc : 0.0f;
    }

    float max_logit = -1e9f;
    for (int k = 0; k < NN_OUTPUT_DIM; k++) {
        float acc = NN_B2[k];
        for (int j = 0; j < NN_HIDDEN_DIM; j++) {
            acc += hidden[j] * NN_W2[j * NN_OUTPUT_DIM + k];
        }
        out_probs[k] = acc;
        if (acc > max_logit) max_logit = acc;
    }

    float sum_exp = 0.0f;
    uint8_t best_class = 0;
    float best_prob = -1.0f;

    for (int k = 0; k < NN_OUTPUT_DIM; k++) {
        out_probs[k] = expf(out_probs[k] - max_logit);
        sum_exp += out_probs[k];
    }
    for (int k = 0; k < NN_OUTPUT_DIM; k++) {
        out_probs[k] /= sum_exp;
        if (out_probs[k] > best_prob) {
            best_prob = out_probs[k];
            best_class = (uint8_t)k;
        }
    }
    return best_class;
}

#endif // FOC_DSP_ENGINE_H