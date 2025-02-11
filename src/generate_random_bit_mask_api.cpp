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

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/generate_random_bit_mask.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

#include <rocrand/rocrand_xorwow.h>

extern "C" miopenStatus_t miopenGetGenerateRandomBitMaskStatesSize(miopenHandle_t handle,
                                                                   size_t* stateSizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, stateSizeInBytes);

    return miopen::try_([&] {
        miopen::deref(stateSizeInBytes) =
            std::min(size_t(MAX_PRNG_STATE), miopen::deref(handle).GetImage3dMaxWidth()) *
            sizeof(rocrand_state_xorwow);
    });
}

extern "C" miopenStatus_t miopenInitPRNGState(miopenHandle_t handle,
                                              void* pstate,
                                              const size_t stateSizeInBytes,
                                              const uint64_t seed)
{
    MIOPEN_LOG_FUNCTION(handle, pstate, stateSizeInBytes, seed);

    return miopen::try_([&] {
        miopen::generate_random_bit_mask::InitPRNGState(
            miopen::deref(handle), DataCast(pstate), stateSizeInBytes, seed);
    });
}

extern "C" miopenStatus_t miopenGenerateRandomBitMask(miopenHandle_t handle,
                                                      const void* pstate,
                                                      const size_t stateSizeInBytes,
                                                      const size_t maskSizeInBytes,
                                                      void* mask,
                                                      const float p)
{
    MIOPEN_LOG_FUNCTION(handle, pstate, stateSizeInBytes, maskSizeInBytes, mask, p);

    return miopen::try_([&] {
        miopen::generate_random_bit_mask::GenerateRandomBitMask(miopen::deref(handle),
                                                                DataCast(pstate),
                                                                stateSizeInBytes,
                                                                maskSizeInBytes,
                                                                DataCast(mask),
                                                                p);
    });
}
