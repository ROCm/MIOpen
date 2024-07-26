/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/errors.hpp>
#include <miopen/graphapi/conv_bias_res_add_activ_forward_executor.hpp>
#include <miopen/fusion.hpp>
#include <miopen/visit_float.hpp>

namespace miopen {

namespace graphapi {

static std::vector<int> Convert(const std::vector<int64_t>& values)
{
    return {values.begin(), values.end()};
}

static ConvolutionDescriptor Convert(const Convolution& conv, int groupCount)
{
    return {conv.getSpatialDims(),
            conv.getMode(),
            miopenPaddingMode_t::miopenPaddingDefault,
            Convert(conv.getPrePaddings()),
            Convert(conv.getFilterStrides()),
            Convert(conv.getDilations()),
            Convert(conv.getPostPaddings()),
            groupCount};
}

static float Convert(half_float::half value) { return static_cast<float>(value); }

static float Convert(float value) { return value; }

void ConvBiasResAddActivForwardExecutor::execute(miopenHandle_t handle, const VariantPack& vpk)
{
    MIOPEN_THROW_IF(mActivationOp->getPointwise()->getMode() !=
                        miopenPointwiseMode_t::MIOPEN_POINTWISE_RELU_FWD,
                    "invalid activation operation");

    MIOPEN_THROW_IF(mAddOp->getPointwise()->getMode() !=
                        miopenPointwiseMode_t::MIOPEN_POINTWISE_ADD,
                    "invalid pointwise operation for add op");

    MIOPEN_THROW_IF(mBiasOp->getPointwise()->getMode() !=
                        miopenPointwiseMode_t::MIOPEN_POINTWISE_ADD,
                    "invalid pointwise operation for bias op");

    auto* xDesc = mConvOp->getX();
    auto* xData = vpk.getDataPointer(xDesc->getId());

    auto* wDesc = mConvOp->getW();
    auto* wData = vpk.getDataPointer(wDesc->getId());

    std::size_t in_c  = xDesc->GetLengths()[1];
    std::size_t wei_c = wDesc->GetLengths()[1];
    int groupCount    = in_c / wei_c;
    auto convDesc     = Convert(*mConvOp->getConvolution(), groupCount);

    bool xIsVirtual = mAddOp->getX()->isVirtual();
    auto* zDesc     = xIsVirtual ? mAddOp->getB() : mAddOp->getX();
    auto* zData     = vpk.getDataPointer(zDesc->getId());

    auto biasDesc  = mBiasOp->getX()->isVirtual() ? mBiasOp->getB() : mBiasOp->getX();
    auto* biasData = vpk.getDataPointer(biasDesc->getId());

    auto yDesc  = mActivationOp->getY();
    auto* yData = vpk.getDataPointer(yDesc->getId());

    auto convertToFloat = [](auto&& arg) { return Convert(arg); };
    // The virtual tensor for add is the result of the convolution, and combining the alpha1's allow
    // users to specify it through the graphAPI properly.
    float alpha1 =
        mConvOp->getAlpha() * (xIsVirtual ? std::visit(convertToFloat, mAddOp->getAlpha1())
                                          : std::visit(convertToFloat, mAddOp->getAlpha2()));

    // The non-virtual tensor for add should be alpha 2.
    float alpha2 =
        xIsVirtual ? std::get<float>(mAddOp->getAlpha2()) : std::get<float>(mAddOp->getAlpha1());

    ActivationDescriptor activDesc{
        miopenActivationRELU, std::visit(convertToFloat, mActivationOp->getAlpha1()), 1.0, 1.0};

    auto status =
        ConvBiasActivFusion(miopen::deref(handle),
                            &alpha1,
                            *xDesc,
                            xData,
                            *wDesc,
                            wData,
                            convDesc,
                            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM,
                            nullptr,
                            0,
                            &alpha2,
                            *zDesc,
                            zData,
                            *biasDesc,
                            biasData,
                            activDesc,
                            *yDesc,
                            yData);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "execute failed");
}

} // namespace graphapi

} // namespace miopen
