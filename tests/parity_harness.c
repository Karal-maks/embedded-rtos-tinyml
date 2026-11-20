/* Parity harness: runs the production C DSP + inference on windows from stdin.
 *
 * stdin : N records of float32 vibration[64] followed by float32 iq[64]
 * stdout: N records of float32 features[6] followed by float32 probs[4]
 *
 * Used by tests/test_parity.py to check that the C code and the Python
 * training pipeline compute the same features and the same predictions.
 */
#include <stdio.h>
#include "foc_dsp_engine.h"

int main(void) {
    SensorFrame_t frame;
    float features[NN_INPUT_DIM];
    float probs[NN_OUTPUT_DIM];

    while (fread(frame.vibration_g, sizeof(float), SAMPLE_WINDOW_SIZE, stdin) == SAMPLE_WINDOW_SIZE &&
           fread(frame.iq_current, sizeof(float), SAMPLE_WINDOW_SIZE, stdin) == SAMPLE_WINDOW_SIZE) {
        extract_dsp_features(&frame, features);
        tinyml_predict_fault(features, probs);
        fwrite(features, sizeof(float), NN_INPUT_DIM, stdout);
        fwrite(probs, sizeof(float), NN_OUTPUT_DIM, stdout);
    }
    return 0;
}
