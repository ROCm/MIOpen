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

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include "mha.hpp"

namespace miopen {

struct NetworkConfig;

namespace mha {

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase
{
    // softmax forward constructor
    ProblemDescription(const MhaInputDescsForward& descs)
        : isForward(true), mhaInputDescsForwardPtr(std::make_shared<MhaInputDescsForward>(descs))
    {
    }

    // softmax backward constructor
    ProblemDescription(const MhaInputDescsBackward& descs)
        : isForward(false), mhaInputDescsBackwardPtr(std::make_shared<MhaInputDescsBackward>(descs))
    {
    }

    bool IsForward() const { return isForward; }
    const MhaInputDescsForward& GetDescsForward() const
    {
        assert(mhaInputDescsForwardPtr && isForward);

        if(mhaInputDescsForwardPtr == nullptr)
        {
            MIOPEN_THROW("Mha ProblemDescription GetDescsForward() failed: PD was initialized with "
                         "a backward direction ctor");
        }

        return *mhaInputDescsForwardPtr;
    }

    const MhaInputDescsBackward& GetDescsBackward() const
    {
        assert(mhaInputDescsBackwardPtr && !isForward);

        if(mhaInputDescsBackwardPtr == nullptr)
        {
            MIOPEN_THROW("Mha ProblemDescription GetDescsBackward() failed: PD was initialized "
                         "with a forward direction ctor");
        }

        return *mhaInputDescsBackwardPtr;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const bool isForward;

    std::shared_ptr<MhaInputDescsForward> mhaInputDescsForwardPtr;
    std::shared_ptr<MhaInputDescsBackward> mhaInputDescsBackwardPtr;
};

} // namespace mha
} // namespace miopen
