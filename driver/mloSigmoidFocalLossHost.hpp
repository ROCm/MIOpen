#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
void mloSigmoidFocalLossFwdRunHost(Tgpu* input,
                                   miopenTensorDescriptor_t inputDesc,
                                   Tgpu* target,
                                   miopenTensorDescriptor_t targetDesc,
                                   Tcheck* outputHost,
                                   miopenTensorDescriptor_t outputDesc,
                                   float alpha,
                                   float gamma,
                                   miopenLossReductionMode_t reduction,
                                   float divisor)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto output_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        Tcheck i = static_cast<Tcheck>(input[input_tv.get_tensor_view_idx(idx)]);
        Tcheck t = static_cast<Tcheck>(target[target_tv.get_tensor_view_idx(idx)]);

        Tcheck sig    = 1 / (1 + exp(-i));
        Tcheck ceLoss = -(t * log(sig) + (1 - t) * log(1 - sig));
        Tcheck sigT   = sig * t + (1 - sig) * (1 - t);
        Tcheck loss   = ceLoss * pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            Tcheck alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss          = alphaT * loss;
        }

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            outputHost[output_tv.get_tensor_view_idx(idx)] = loss;
        }
        else
        {
            outputHost[0] += static_cast<Tcheck>(loss / divisor);
        }
    }
}

template <typename Tgpu, typename Tcheck>
void mloSigmoidFocalLossBwdRunHost(Tgpu* input,
                                   miopenTensorDescriptor_t inputDesc,
                                   Tgpu* target,
                                   miopenTensorDescriptor_t targetDesc,
                                   Tgpu* doutput,
                                   miopenTensorDescriptor_t doutputDesc,
                                   Tcheck* dinput,
                                   miopenTensorDescriptor_t dinputDesc,
                                   Tcheck* dtarget,
                                   miopenTensorDescriptor_t dtargetDesc,
                                   float alpha,
                                   float gamma,
                                   miopenLossReductionMode_t reduction,
                                   float divisor)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(dtargetDesc));

    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    tensor_layout_t<5> doIdx(input_tv, 0);
    Tcheck dO = static_cast<Tcheck>(doutput[doutput_tv.get_tensor_view_idx(doIdx)]);

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        Tcheck i = static_cast<Tcheck>(input[input_tv.get_tensor_view_idx(idx)]);
        Tcheck t = static_cast<Tcheck>(target[target_tv.get_tensor_view_idx(idx)]);
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            dO = static_cast<Tcheck>(doutput[doutput_tv.get_tensor_view_idx(idx)]);
        }

        Tcheck p       = 1 / (1 + exp(-i));
        Tcheck ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
        Tcheck pT      = p * t + (1 - p) * (1 - t);
        Tcheck powPt   = pow(1 - pT, gamma);
        Tcheck alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(dinput)
        {
            Tcheck dpdi      = exp(-i) / pow(1 + exp(-i), 2);
            Tcheck dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            Tcheck dpowptdi  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            Tcheck dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            Tcheck grad = dO * dLdi;

            if(alpha >= 0)
            {
                grad *= alpha_t;
            }
            if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
            {
                grad /= divisor;
            }
            dinput[dinput_tv.get_tensor_view_idx(idx)] = static_cast<Tcheck>(grad);
        }

        if(dtarget)
        {
            Tcheck dcelossdt = -log(p) + log(1 - p);
            Tcheck dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            Tcheck dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            Tcheck gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * dL
                gradTarget = alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt;
            }
            if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
            {
                gradTarget /= divisor;
            }
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<Tcheck>(gradTarget);
        }
    }
}
