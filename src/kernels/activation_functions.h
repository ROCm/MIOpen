
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

static __constant _FLOAT kBNLL_THRESHOLD = (_FLOAT)50.;

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

void ActivationFunction(const uint n,
                        _FLOAT_PREC* res,
                        const _FLOAT_PREC* data,
                        const _FLOAT_PREC _gamma,
                        const _FLOAT_PREC _beta,
                        const _FLOAT_PREC _alpha)
{
    ACTIVATION_SET()
    for(uint i = 0; i < n; ++i)
    {
        ACTIVATION_OP(res[i], data[i], _FLOAT_PREC)
    }
}

void ActivationFunction_PassThru_Diff(const uint n,
                                      _FLOAT_PREC* bot_diff,
                                      const _FLOAT_PREC* top_diff,
                                      UNUSED const _FLOAT_PREC* bot_data,
                                      UNUSED const _FLOAT_PREC* top_data,
                                      UNUSED const _FLOAT_PREC diff_scale,
                                      UNUSED const _FLOAT_PREC gamma,
                                      UNUSED const _FLOAT_PREC beta,
                                      UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i];
    }
}

void ActivationFunction_ReLU_Diff(const uint n,
                                  _FLOAT_PREC* bot_diff,
                                  const _FLOAT_PREC* top_diff,
                                  const _FLOAT_PREC* bot_data,
                                  UNUSED const _FLOAT_PREC* top_data,
                                  UNUSED const _FLOAT_PREC diff_scale,
                                  UNUSED const _FLOAT_PREC gamma,
                                  UNUSED const _FLOAT_PREC beta,
                                  UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
    }
}

void ActivationFunction_TanH_Diff(const uint n,
                                  _FLOAT_PREC* bot_diff,
                                  const _FLOAT_PREC* top_diff,
                                  UNUSED const _FLOAT_PREC* bot_data,
                                  const _FLOAT_PREC* top_data,
                                  UNUSED const _FLOAT_PREC diff_scale,
                                  UNUSED const _FLOAT_PREC gamma,
                                  const _FLOAT_PREC beta,
                                  const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // dy/dx = alpha * (beta - y^2 / beta)
        _FLOAT_PREC y = top_data[i];
        bot_diff[i] =
            fabs(beta) <= EPSILON ? (_FLOAT_PREC)0 : (top_diff[i] * alpha * (beta - y * y / beta));
        // fabs(beta) <= EPSILON ? (_FLOAT)0 : (top_diff[i] * alpha * (beta - y * y / beta));
    }
}

void ActivationFunction_Sigmoid_Diff(const uint n,
                                     _FLOAT_PREC* bot_diff,
                                     const _FLOAT_PREC* top_diff,
                                     UNUSED const _FLOAT_PREC* bot_data,
                                     const _FLOAT_PREC* top_data,
                                     UNUSED const _FLOAT_PREC diff_scale,
                                     UNUSED const _FLOAT_PREC gamma,
                                     UNUSED const _FLOAT_PREC beta,
                                     UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = 1/(1 + exp(-x))
        _FLOAT_PREC sigmoid_x = top_data[i];
        bot_diff[i]           = top_diff[i] * sigmoid_x * ((_FLOAT_PREC)1.f - sigmoid_x);
    }
}

void ActivationFunction_Abs_Diff(const uint n,
                                 _FLOAT_PREC* bot_diff,
                                 const _FLOAT_PREC* top_diff,
                                 const _FLOAT_PREC* bot_data,
                                 UNUSED const _FLOAT_PREC* top_data,
                                 UNUSED const _FLOAT_PREC diff_scale,
                                 UNUSED const _FLOAT_PREC gamma,
                                 UNUSED const _FLOAT_PREC beta,
                                 UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : -1);
    }
}

// Compute dy/dx = beta * gamma * (alpha + beta * x)^(gamma - 1)
//               = diff_scale * y / (alpha + beta * x)
void ActivationFunction_Power_Diff(const uint n,
                                   _FLOAT_PREC* bot_diff,
                                   UNUSED const _FLOAT_PREC* top_diff,
                                   const _FLOAT_PREC* bot_data,
                                   const _FLOAT_PREC* top_data,
                                   const _FLOAT_PREC diff_scale,
                                   UNUSED const _FLOAT_PREC gamma,
                                   const _FLOAT_PREC beta,
                                   const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        _FLOAT_PREC arg = alpha + bot_data[i] * beta;
        bot_diff[i]     = arg <= EPSILON ? (_FLOAT_PREC)0 : (diff_scale * top_data[i] / arg);
        // bot_diff[i]     = arg <= EPSILON ? (_FLOAT_PREC)0 : ((diff_scale * top_data[i]) / arg);
    }
}

void ActivationFunction_BNLL_Diff(const uint n,
                                  _FLOAT_PREC* bot_diff,
                                  const _FLOAT_PREC* top_diff,
                                  const _FLOAT_PREC* bot_data,
                                  UNUSED const _FLOAT_PREC* top_data,
                                  UNUSED const _FLOAT_PREC diff_scale,
                                  UNUSED const _FLOAT_PREC gamma,
                                  UNUSED const _FLOAT_PREC beta,
                                  UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = (log(1 + exp(x)))
        // dy/dx = 1/ (1 + exp(-x))
        _FLOAT_PREC expval = exp(fmin((_FLOAT_PREC)bot_data[i], (_FLOAT_PREC)kBNLL_THRESHOLD));
        bot_diff[i]        = top_diff[i] * expval / (expval + (_FLOAT_PREC)1.f);
    }
}

void ActivationFunction_Leaky_ReLU_Diff(const uint n,
                                        _FLOAT_PREC* bot_diff,
                                        const _FLOAT_PREC* top_diff,
                                        const _FLOAT_PREC* bot_data,
                                        UNUSED const _FLOAT_PREC* top_data,
                                        UNUSED const _FLOAT_PREC diff_scale,
                                        UNUSED const _FLOAT_PREC gamma,
                                        UNUSED const _FLOAT_PREC beta,
                                        const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? (_FLOAT_PREC)1.f : alpha);
    }
}

void ActivationFunction_Clipped_ReLU_Diff(const uint n,
                                          _FLOAT_PREC* bot_diff,
                                          const _FLOAT_PREC* top_diff,
                                          const _FLOAT_PREC* bot_data,
                                          UNUSED const _FLOAT_PREC* top_data,
                                          UNUSED const _FLOAT_PREC diff_scale,
                                          UNUSED const _FLOAT_PREC gamma,
                                          UNUSED const _FLOAT_PREC beta,
                                          const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0 && bot_data[i] <= alpha) ? (_FLOAT_PREC)1.f
                                                                               : (_FLOAT_PREC)0.f);
    }
}

void ActivationFunction_ELU_Diff(const uint n,
                                 _FLOAT_PREC* bot_diff,
                                 const _FLOAT_PREC* top_diff,
                                 const _FLOAT_PREC* bot_data,
                                 const _FLOAT_PREC* top_data,
                                 UNUSED const _FLOAT_PREC diff_scale,
                                 UNUSED const _FLOAT_PREC gamma,
                                 UNUSED const _FLOAT_PREC beta,
                                 const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : top_data[i] + alpha);
    }
}

void ActivationFunction_Diff(const uint n,
                             _FLOAT_PREC* bot_diff,
                             const _FLOAT_PREC* top_diff,
                             const _FLOAT_PREC* bot_data,
                             const _FLOAT_PREC* top_data,
                             const _FLOAT_PREC diff_scale,
                             const _FLOAT_PREC gamma,
                             const _FLOAT_PREC beta,
                             const _FLOAT_PREC alpha)
{
#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
    {
        ActivationFunction_PassThru_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
    {
        // y = 1/(1 + exp(-x))
        ActivationFunction_Sigmoid_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
    {
        // y = beta * tanh(alpha * x)
        ActivationFunction_TanH_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
    {
        ActivationFunction_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
    {
        // y = log(1 + exp(x))
        ActivationFunction_BNLL_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
    {
        ActivationFunction_Abs_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
    {
        // y = (alpha + beta * x ) ^ gamma
        ActivationFunction_Power_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
    {
        ActivationFunction_Clipped_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
    {
        ActivationFunction_Leaky_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
    {
        ActivationFunction_ELU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#endif
}
