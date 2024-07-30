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

void ConvBiasResAddActivForwardExecutor::execute(miopenHandle_t handle, const VariantPack& vpk)
{
    std::size_t in_c  = mXTensor->GetLengths()[1];
    std::size_t wei_c = mWTensor->GetLengths()[1];
    int groupCount    = in_c / wei_c;
    auto convDesc     = Convert(*mConvolution, groupCount);

    ActivationDescriptor activDesc{miopenActivationRELU, mActivationAlpha, 1.0, 1.0};

    auto* xData    = vpk.getDataPointer(mXTensor->getId());
    auto* wData    = vpk.getDataPointer(mWTensor->getId());
    auto* zData    = vpk.getDataPointer(mZTensor->getId());
    auto* biasData = vpk.getDataPointer(mBiasTensor->getId());
    auto* yData    = vpk.getDataPointer(mYTensor->getId());

    auto status =
        ConvBiasActivFusion(miopen::deref(handle),
                            &mAlpha1,
                            *mXTensor,
                            xData,
                            *mWTensor,
                            wData,
                            convDesc,
                            miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM,
                            nullptr,
                            0,
                            &mAlpha2,
                            *mZTensor,
                            zData,
                            *mBiasTensor,
                            biasData,
                            activDesc,
                            *mYTensor,
                            yData);

    MIOPEN_THROW_IF(status != miopenStatusSuccess, "execute failed");
}

} // namespace graphapi

} // namespace miopen
