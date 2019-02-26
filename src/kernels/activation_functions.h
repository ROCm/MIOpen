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

void ActivationFunction_PassThru(const uint n,
                                 _FLOAT_PREC* res,
                                 const _FLOAT_PREC* data,
                                 UNUSED const _FLOAT_PREC gamma,
                                 UNUSED const _FLOAT_PREC beta,
                                 UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i];
    }
}

void ActivationFunction_ReLU(const uint n,
                             _FLOAT_PREC* res,
                             const _FLOAT_PREC* data,
                             UNUSED const _FLOAT_PREC gamma,
                             UNUSED const _FLOAT_PREC beta,
                             UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i] * (data[i] > 0);
    }
}

void ActivationFunction_Sigmoid(const uint n,
                                _FLOAT_PREC* res,
                                const _FLOAT_PREC* data,
                                UNUSED const _FLOAT_PREC gamma,
                                UNUSED const _FLOAT_PREC beta,
                                UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = 1/(1 + exp(-x))
        res[i] = (_FLOAT_PREC)1.f / ((_FLOAT_PREC)1.f + exp(-data[i]));
    }
}

void ActivationFunction_TanH(const uint n,
                             _FLOAT_PREC* res,
                             const _FLOAT_PREC* data,
                             UNUSED const _FLOAT_PREC gamma,
                             const _FLOAT_PREC beta,
                             const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = beta * tanh(alpha * x)
        res[i] = beta * tanh(alpha * data[i]);
    }
}

void ActivationFunction_Abs(const uint n,
                            _FLOAT_PREC* res,
                            const _FLOAT_PREC* data,
                            UNUSED const _FLOAT_PREC gamma,
                            UNUSED const _FLOAT_PREC beta,
                            UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fabs(data[i]);
    }
}

void ActivationFunction_Square(const uint n,
                               _FLOAT_PREC* res,
                               const _FLOAT_PREC* data,
                               UNUSED const _FLOAT_PREC gamma,
                               UNUSED const _FLOAT_PREC beta,
                               UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = data[i] * data[i];
    }
}

void ActivationFunction_Sqrt(const uint n,
                             _FLOAT_PREC* res,
                             const _FLOAT_PREC* data,
                             UNUSED const _FLOAT_PREC gamma,
                             UNUSED const _FLOAT_PREC beta,
                             UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = sqrt(data[i]);
    }
}

void ActivationFunction_Linear(const uint n,
                               _FLOAT_PREC* res,
                               const _FLOAT_PREC* data,
                               UNUSED const _FLOAT_PREC gamma,
                               const _FLOAT_PREC beta,
                               const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = alpha + beta * data[i];
    }
}

void ActivationFunction_Power(const uint n,
                              _FLOAT_PREC* res,
                              const _FLOAT_PREC* data,
                              const _FLOAT_PREC gamma,
                              const _FLOAT_PREC beta,
                              const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = (alpha + beta * x ) ^ gamma
        _FLOAT_PREC arg = alpha + data[i] * beta;
        res[i]          = arg <= EPSILON ? (_FLOAT_PREC)0 : pow(arg, gamma);
    }
}

void ActivationFunction_BNLL(const uint n,
                             _FLOAT_PREC* res,
                             const _FLOAT_PREC* data,
                             UNUSED const _FLOAT_PREC gamma,
                             UNUSED const _FLOAT_PREC beta,
                             UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        //	y = log(1 + exp(x))
        res[i] = (data[i] > 0) ? (data[i] + log((_FLOAT_PREC)1.f + exp(-data[i])))
                               : log((_FLOAT_PREC)(1.f) + exp(data[i]));
    }
}

void ActivationFunction_Leaky_ReLU(const uint n,
                                   _FLOAT_PREC* res,
                                   const _FLOAT_PREC* data,
                                   UNUSED const _FLOAT_PREC gamma,
                                   UNUSED const _FLOAT_PREC beta,
                                   const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i] * ((data[i] > 0) ? (_FLOAT_PREC)1.f : alpha);
    }
}

void ActivationFunction_Clipped_ReLU(const uint n,
                                     _FLOAT_PREC* res,
                                     const _FLOAT_PREC* data,
                                     UNUSED const _FLOAT_PREC gamma,
                                     UNUSED const _FLOAT_PREC beta,
                                     const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fmin(alpha, fmax(data[i], 0));
    }
}

void ActivationFunction_ELU(const uint n,
                            _FLOAT_PREC* res,
                            const _FLOAT_PREC* data,
                            UNUSED const _FLOAT_PREC gamma,
                            UNUSED const _FLOAT_PREC beta,
                            const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : (alpha * (exp(data[i]) - (_FLOAT_PREC)1.f));
    }
}

void ActivationFunction(const uint n,
                        _FLOAT_PREC* res,
                        const _FLOAT_PREC* data,
                        const _FLOAT_PREC gamma,
                        const _FLOAT_PREC beta,
                        const _FLOAT_PREC alpha)
{
#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
    {
        ActivationFunction_PassThru(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
    {
        // y = 1/(1 + exp(-x))
        ActivationFunction_Sigmoid(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
    {
        // y = beta * tanh(alpha * x)
        ActivationFunction_TanH(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
    {
        ActivationFunction_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
    {
        // y = log(1 + exp(x))
        ActivationFunction_BNLL(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
    {
        ActivationFunction_Abs(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
    {
        // y = (alpha + beta * x ) ^ gamma
        ActivationFunction_Power(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
    {
        ActivationFunction_Clipped_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
    {
        ActivationFunction_Leaky_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
    {
        ActivationFunction_ELU(n, res, data, gamma, beta, alpha);
    }
#endif
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
        _FLOAT_PREC expval = exp(fmin(bot_data[i], kBNLL_THRESHOLD));
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
