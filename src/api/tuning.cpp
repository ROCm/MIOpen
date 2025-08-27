/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>

miopenStatus_t miopenSetTuningPolicy(miopenHandle_t handle, miopenTuningPolicy_t newValue)
{
    return miopen::try_([&] {
        if(newValue < miopenTuningPolicyNone || newValue > miopenTuningPolicyDbClean)
            MIOPEN_THROW(miopenStatusBadParm,
                         "miopenSetTuningPolicy called with invalid value of " +
                             std::to_string(newValue));

        auto& handle_deref = miopen::deref(handle);
        handle_deref.SetTuningPolicy(newValue);
    });
}

miopenStatus_t miopenGetTuningPolicy(miopenHandle_t handle, miopenTuningPolicy_t* value)
{
    return miopen::try_([&] {
        if(value == nullptr)
            MIOPEN_THROW(miopenStatusBadParm, "miopenGetTuningPolicy called with null");

        const auto& handle_deref = miopen::deref(handle);
        *value                   = handle_deref.GetTuningPolicy();
    });
}
