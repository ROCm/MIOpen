#ifndef MIOPEN_NRN_OP_ID
#define MIOPEN_NRN_OP_ID 0
#endif

#define MIOPEN_NEURON_PASTHRU 0      // x
#define MIOPEN_NEURON_LOGISTIC 1     // 1 / (1 + e^-x)  //Sigmoid
#define MIOPEN_NEURON_TANH 2         // beta * tanh(alpha * x)
#define MIOPEN_NEURON_RELU 3         // max(0, x)
#define MIOPEN_NEURON_SOFTRELU 4     // log(1 + e^x)   // bonomial normal log likelihood
#define MIOPEN_NEURON_ABS 5          // abs(x)
#define MIOPEN_NEURON_POWER 6        // (alpha + beta * x )^gamma
#define MIOPEN_NEURON_CLIPPED_RELU 7 // min(alpha, max(0, x))
#define MIOPEN_NEURON_LEAKY_RELU 8   // alpha * x | x <= 0; x | x > 0
#define MIOPEN_NEURON_ELU 9          // alpha * (e^x - 1) | x <= 0; x | x > 0
#define MIOPEN_NEURON_TOTAL 10

#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
#define ACTIVATION_SET() (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
#define ACTIVATION_SET() \
    do                   \
    {                    \
    } while(0);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
#define ACTIVATION_SET() \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
#define ACTIVATION_SET() \
    (void)_beta;         \
    (void)_gamma;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
#define ACTIVATION_SET() \
    (void)_beta;         \
    (void)_gamma;
#endif

#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) out = tmp;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = (_FLOAT_PREC_TYPE)1.f / ((_FLOAT_PREC_TYPE)1.f + exp(-tmp));
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = (_FLOAT_PREC_TYPE)_beta * tanh((_FLOAT_PREC_TYPE)_alpha * tmp);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) out = max(tmp, (_FLOAT_PREC_TYPE)0.);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE)                    \
    out = (tmp > 0) ? (tmp + log((_FLOAT_PREC_TYPE)1.f + exp(-tmp))) \
                    : log((_FLOAT_PREC_TYPE)1.f + exp(tmp));
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) out = fabs(tmp);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE)                                    \
    _FLOAT_PREC_TYPE arg = (_FLOAT_PREC_TYPE)_alpha + tmp * (_FLOAT_PREC_TYPE)_beta; \
    out                  = (arg <= (_FLOAT_PREC_TYPE)EPSILON) ? (_FLOAT_PREC_TYPE)0. \
                                                              : pow(arg, (_FLOAT_PREC_TYPE)_gamma);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = min((_FLOAT_PREC_TYPE)_alpha, max(tmp, (_FLOAT_PREC_TYPE)0.));
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = tmp * ((tmp > 0) ? (_FLOAT_PREC_TYPE)1.f : (_FLOAT_PREC_TYPE)_alpha);
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = (tmp > 0) ? tmp : ((_FLOAT_PREC_TYPE)_alpha * (exp(tmp) - (_FLOAT_PREC_TYPE)1.f));
#endif
