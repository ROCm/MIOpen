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

#pragma once

#include <miopen/graphapi/convolution.hpp>
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/variant_pack.hpp>

#include <memory>

namespace miopen {

namespace graphapi {

class ConvBiasResAddActivForwardExecutor : public GraphPatternExecutor
{
    // We dont use pointers here as we need deserialization
    Tensor mXTensor;
    Tensor mWTensor;
    Convolution mConvolution;
    int mGroupCount;
    Tensor mZTensor;
    Tensor mBiasTensor;
    Tensor mYTensor;
    float mAlpha1;
    float mAlpha2;
    float mActivationAlpha;

public:
    ConvBiasResAddActivForwardExecutor(Tensor* xTensor,
                                       Tensor* wTensor,
                                       Convolution* convolution,
                                       int groupCount,
                                       Tensor* zTensor,
                                       Tensor* biasTensor,
                                       Tensor* yTensor,
                                       float alpha1,
                                       float alpha2,
                                       float activationAlpha)
        : GraphPatternExecutor(),
          mXTensor(*xTensor),
          mWTensor(*wTensor),
          mConvolution(*convolution),
          mGroupCount(groupCount),
          mZTensor(*zTensor),
          mBiasTensor(*biasTensor),
          mYTensor(*yTensor),
          mAlpha1(alpha1),
          mAlpha2(alpha2),
          mActivationAlpha(activationAlpha)
    {
    }

    ConvBiasResAddActivForwardExecutor(const nlohmann::json& json);

    void execute(miopenHandle_t handle, const VariantPack& vpk) final;

    size_t getWorkspaceSize() const final { return size_t{0}; }

    nlohmann::json getJson() final;

    static constexpr const char* name = "ConvBiasResAddActivForwardExecutor";

    struct JsonFields
    {
        static constexpr const char* XTensor         = "x_tensor";
        static constexpr const char* WTensor         = "w_tensor";
        static constexpr const char* Convolution     = "convolution";
        static constexpr const char* GroupCount      = "group_count";
        static constexpr const char* ZTensor         = "z_tensor";
        static constexpr const char* BiasTensor      = "bias_tensor";
        static constexpr const char* YTensor         = "y_tensor";
        static constexpr const char* Alpha1          = "alpha1";
        static constexpr const char* Alpha2          = "alpha2";
        static constexpr const char* ActivationAlpha = "activation_alpha";
    };
};

} // namespace graphapi

} // namespace miopen
