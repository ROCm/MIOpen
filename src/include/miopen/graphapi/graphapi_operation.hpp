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

#include <miopen/errors.hpp>
#include <miopen/graphapi/graphapi_tensor.hpp>

#include <algorithm>
#include <tuple>
#include <vector>

namespace miopen {

namespace graphapi {

class Operation
{
public:
    enum class ArgumentId
    {
        X,
        Y,
        W
    };

    Operation& clearInputsOutputs()
    {
        mInputs.clear();
        mOutputs.clear();
        return *this;
    }
    Operation& setInput(ArgumentId argumentId, Tensor* pTensor)
    {
        auto existing =
            std::find_if(mInputs.begin(), mInputs.end(), [argumentId](const auto& item) {
                return std::get<0>(item) == argumentId;
            });
        if(existing == mInputs.end())
        {
            mInputs.emplace_back(argumentId, pTensor);
        }
        else
        {
            std::get<1>(*existing) = pTensor;
        }
        return *this;
    }
    Operation& setOutput(ArgumentId argumentId, Tensor* pTensor)
    {
        auto existing =
            std::find_if(mOutputs.begin(), mOutputs.end(), [argumentId](const auto& item) {
                return std::get<0>(item) == argumentId;
            });
        if(existing == mOutputs.end())
        {
            mOutputs.emplace_back(argumentId, pTensor);
        }
        else
        {
            std::get<1>(*existing) = pTensor;
        }
        return *this;
    }
    Tensor* getInput(ArgumentId argumentId) const
    {
        auto searched =
            std::find_if(mInputs.begin(), mInputs.end(), [argumentId](const auto& item) {
                return std::get<0>(item) == argumentId;
            });
        if(searched == mInputs.end())
        {
            MIOPEN_THROW(miopenStatusInternalError);
        }
        return std::get<1>(*searched);
    }
    Tensor* getOutput(ArgumentId argumentId) const
    {
        auto searched =
            std::find_if(mOutputs.begin(), mOutputs.end(), [argumentId](const auto& item) {
                return std::get<0>(item) == argumentId;
            });
        if(searched == mOutputs.end())
        {
            MIOPEN_THROW(miopenStatusInternalError);
        }
        return std::get<1>(*searched);
    }
    bool hasInput(Tensor* pTensor)
    {
        return std::find_if(mInputs.begin(), mInputs.end(), [pTensor](const auto& item) {
                   return std::get<1>(item) == pTensor;
               }) != mInputs.end();
    }
    bool hasOutput(Tensor* pTensor)
    {
        return std::find_if(mOutputs.begin(), mOutputs.end(), [pTensor](const auto& item) {
                   return std::get<1>(item) == pTensor;
               }) != mOutputs.end();
    }

private:
    std::vector<std::tuple<ArgumentId, Tensor*>> mInputs;
    std::vector<std::tuple<ArgumentId, Tensor*>> mOutputs;
};

} // namespace graphapi

} // namespace miopen
