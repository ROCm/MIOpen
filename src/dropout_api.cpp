/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/errors.hpp>
#include <miopen/dropout.hpp>
#include <miopen/tensor_ops.hpp>

// disable __device__ qualifiers
#ifdef FQUALIFIERS
#error rocrand FQUALIFIERS defined externally, probably one of rocrand device header included prior to this
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-macros"
#define FQUALIFIERS inline
#pragma clang diagnostic pop
#include "kernels/miopen_rocrand.hpp"

extern "C" miopenStatus_t miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc)
{

    MIOPEN_LOG_FUNCTION(dropoutDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(dropoutDesc);
        desc       = new miopen::DropoutDescriptor();
    });
}

extern "C" miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc)
{
    MIOPEN_LOG_FUNCTION(dropoutDesc);
    return miopen::try_([&] { miopen_destroy_object(dropoutDesc); });
}

extern "C" miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc,
                                                           size_t* reserveSpaceSizeInBytes)
{
    MIOPEN_LOG_FUNCTION(xDesc);
    return miopen::try_([&] {
        miopen::deref(reserveSpaceSizeInBytes) =
            miopen::deref(xDesc).GetElementSize() * sizeof(bool);
    });
}

extern "C" miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle,
                                                     size_t* stateSizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle);
    return miopen::try_([&] {
        miopen::deref(stateSizeInBytes) =
            std::min(size_t(MAX_PRNG_STATE), miopen::deref(handle).GetImage3dMaxWidth()) *
            sizeof(rocrand_state_xorwow);
    });
}

extern "C" miopenStatus_t miopenGetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t /* handle */,
                                                     float* dropout,
                                                     void** states,
                                                     unsigned long long* seed,
                                                     bool* use_mask,
                                                     bool* state_evo,
                                                     miopenRNGType_t* rng_mode)
{
    MIOPEN_LOG_FUNCTION(dropoutDesc);
    return miopen::try_([&] {
        miopen::deref(dropout)   = miopen::deref(dropoutDesc).dropout;
        miopen::deref(states)    = &(miopen::deref(dropoutDesc).pstates);
        miopen::deref(seed)      = miopen::deref(dropoutDesc).seed;
        miopen::deref(use_mask)  = miopen::deref(dropoutDesc).use_mask;
        miopen::deref(state_evo) = miopen::deref(dropoutDesc).state_evo;
        miopen::deref(rng_mode)  = miopen::deref(dropoutDesc).rng_mode;
    });
}

extern "C" miopenStatus_t miopenRestoreDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                         miopenHandle_t /* handle */,
                                                         float dropout,
                                                         void* states,
                                                         size_t stateSizeInBytes,
                                                         unsigned long long seed,
                                                         bool use_mask,
                                                         bool state_evo,
                                                         miopenRNGType_t rng_mode)
{

    MIOPEN_LOG_FUNCTION(dropoutDesc, dropout, states, stateSizeInBytes, seed, use_mask, state_evo);
    return miopen::try_([&] {
        miopen::deref(dropoutDesc).dropout          = dropout;
        miopen::deref(dropoutDesc).pstates          = DataCast(states);
        miopen::deref(dropoutDesc).stateSizeInBytes = stateSizeInBytes;
        miopen::deref(dropoutDesc).seed             = seed;
        miopen::deref(dropoutDesc).use_mask         = use_mask;
        miopen::deref(dropoutDesc).state_evo        = state_evo;
        miopen::deref(dropoutDesc).rng_mode         = rng_mode;
    });
}

extern "C" miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc,
                                                     miopenHandle_t handle,
                                                     float dropout,
                                                     void* states,
                                                     size_t stateSizeInBytes,
                                                     unsigned long long seed,
                                                     bool use_mask,
                                                     bool state_evo,
                                                     miopenRNGType_t rng_mode)
{

    MIOPEN_LOG_FUNCTION(dropoutDesc, dropout, states, stateSizeInBytes, seed, use_mask, state_evo);
    return miopen::try_([&] {
        miopen::deref(dropoutDesc).dropout          = dropout;
        miopen::deref(dropoutDesc).pstates          = DataCast(states);
        miopen::deref(dropoutDesc).stateSizeInBytes = stateSizeInBytes;
        miopen::deref(dropoutDesc).seed             = seed;
        miopen::deref(dropoutDesc).use_mask         = use_mask;
        miopen::deref(dropoutDesc).state_evo        = state_evo;
        miopen::deref(dropoutDesc).rng_mode         = rng_mode;
        miopen::deref(dropoutDesc)
            .InitPRNGState(miopen::deref(handle), DataCast(states), stateSizeInBytes, seed);
    });
}

static void LogCmdDropout(const miopenDropoutDescriptor_t dropoutDesc,
                          const miopenTensorDescriptor_t xDesc,
                          bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        if(miopen::deref(xDesc).GetType() == miopenFloat)
            ss << "dropout";
        else if(miopen::deref(xDesc).GetType() == miopenHalf)
            ss << "dropoutfp16";

        if(is_fwd)
            ss << " -F 1";
        else
            ss << " -F 2";
        // clang-format off
        ss << " -d " << miopen::deref(xDesc).GetLengths().size()
           << " -e " << std::to_string(static_cast<int>(miopen::deref(dropoutDesc).use_mask))
           << " -l " << (miopen::deref(dropoutDesc).seed & 0xFFFFFFFF)
           << " -m " << ((miopen::deref(dropoutDesc).seed >> 32) & 0xFFFFFFFF)
           << " -p " << std::to_string(miopen::deref(dropoutDesc).dropout);
        // clang-format on 
        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}
extern "C" miopenStatus_t miopenDropoutForward(miopenHandle_t handle,
                                               const miopenDropoutDescriptor_t dropoutDesc,
                                               const miopenTensorDescriptor_t noise_shape,
                                               const miopenTensorDescriptor_t xDesc,
                                               const void* x,
                                               const miopenTensorDescriptor_t yDesc,
                                               void* y,
                                               void* reserveSpace,
                                               size_t reserveSpaceSizeInBytes)
{

    MIOPEN_LOG_FUNCTION(
        dropoutDesc, noise_shape, xDesc, x, yDesc, y, reserveSpace, reserveSpaceSizeInBytes);
    LogCmdDropout(dropoutDesc, xDesc, true);
    return miopen::try_([&] {
        miopen::deref(dropoutDesc)
            .DropoutForward(miopen::deref(handle),
                            miopen::deref(noise_shape),
                            miopen::deref(xDesc),
                            DataCast(x),
                            miopen::deref(yDesc),
                            DataCast(y),
                            DataCast(reserveSpace),
                            reserveSpaceSizeInBytes);
    });
}

extern "C" miopenStatus_t miopenDropoutBackward(miopenHandle_t handle,
                                                const miopenDropoutDescriptor_t dropoutDesc,
                                                const miopenTensorDescriptor_t noise_shape,
                                                const miopenTensorDescriptor_t dyDesc,
                                                const void* dy,
                                                const miopenTensorDescriptor_t dxDesc,
                                                void* dx,
                                                void* reserveSpace,
                                                size_t reserveSpaceSizeInBytes)
{

    MIOPEN_LOG_FUNCTION(dropoutDesc, dyDesc, dy, dxDesc, dx, reserveSpace, reserveSpaceSizeInBytes);
    LogCmdDropout(dropoutDesc, dxDesc, false);
    return miopen::try_([&] {
        miopen::deref(dropoutDesc)
            .DropoutBackward(miopen::deref(handle),
                             miopen::deref(noise_shape),
                             miopen::deref(dyDesc),
                             DataCast(dy),
                             miopen::deref(dxDesc),
                             DataCast(dx),
                             DataCast(reserveSpace),
                             reserveSpaceSizeInBytes);
    });
}
