#include "activations/Linear.h"

// Activacion lineal (identidad). No realiza ninguna operacion.
void Linear::forward(float* /*d_inout*/, int /*N*/, int /*total*/) {}

void Linear::backward(float* /*d_grad*/, const float* /*d_fwd_output*/, int /*N*/, int /*total*/) {}
