#ifndef MIOPEN_NRN_OP_ID
#define MIOPEN_NRN_OP_ID 0
#endif

// NOTE: these are the only ones batch norm supports with fusion
#define MIOPEN_NEURON_PASTHRU 0      // x
#define MIOPEN_NEURON_RELU 3         // max(0, x)
#define MIOPEN_NEURON_CLIPPED_RELU 7 // max(0, min(alpha, x))
#define MIOPEN_NEURON_CLAMP 10       // max(alpha, min(beta, x))
#define MIOPEN_NEURON_TOTAL 11

#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
#define ACTIVATION_SET() \
    (void)_alpha;        \
    (void)_beta;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
#define ACTIVATION_SET() (void)_beta;
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLAMP
#define ACTIVATION_SET()
#endif

#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) out = tmp;
#define ACTIVATION_OP_BWD(out, xnorm, scale, bias, dy, _FLOAT_PREC_TYPE) out = dy;

#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) out = max((_FLOAT_PREC_TYPE)0., tmp);
#define ACTIVATION_OP_BWD(out, xnorm, scale, bias, dy, _FLOAT_PREC_TYPE)                   \
    _FLOAT_PREC_TYPE macro_tmp = (_FLOAT_PREC_TYPE)scale * xnorm + (_FLOAT_PREC_TYPE)bias; \
    out                        = ((macro_tmp) > 0) ? dy : (_FLOAT_PREC_TYPE)0.;

#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = max((_FLOAT_PREC_TYPE)0., min((_FLOAT_PREC_TYPE)_alpha, tmp));
#define ACTIVATION_OP_BWD(out, xnorm, scale, bias, dy, _FLOAT_PREC_TYPE)                   \
    _FLOAT_PREC_TYPE macro_tmp = (_FLOAT_PREC_TYPE)scale * xnorm + (_FLOAT_PREC_TYPE)bias; \
    out = (macro_tmp > (_FLOAT_PREC_TYPE)0. && macro_tmp <= (_FLOAT_PREC_TYPE)_alpha)      \
              ? dy                                                                         \
              : (_FLOAT_PREC_TYPE)0.f;

#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLAMP
#define ACTIVATION_OP(out, tmp, _FLOAT_PREC_TYPE) \
    out = max((_FLOAT_PREC_TYPE)_alpha, min((_FLOAT_PREC_TYPE)_beta, tmp));
#define ACTIVATION_OP_BWD(out, xnorm, scale, bias, dy, _FLOAT_PREC_TYPE)                   \
    _FLOAT_PREC_TYPE macro_tmp = (_FLOAT_PREC_TYPE)scale * xnorm + (_FLOAT_PREC_TYPE)bias; \
    out = (macro_tmp > (_FLOAT_PREC_TYPE)_alpha && macro_tmp <= (_FLOAT_PREC_TYPE)_beta)   \
              ? dy                                                                         \
              : (_FLOAT_PREC_TYPE)0.f;

#endif
