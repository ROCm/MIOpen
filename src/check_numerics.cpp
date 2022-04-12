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

#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CHECK_NUMERICS)

bool CheckNumericsEnabled(const int bitMask)
{
    return (miopen::Value(MIOPEN_CHECK_NUMERICS{}) & bitMask) != 0;
}

// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult
{
    float sum    = 0.0f;
    float absSum = 0.0f;
    float min    = 0.0f;
    float max    = 0.0f;

    int hasZero = 0;
    int hasNan  = 0;
    int hasInf  = 0;
};

bool checkNumericsImpl(
    const Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput)
{
    int numElements = dDesc.GetElementSize();

    // TODO - some constants we should get from the device:
    const int blockSize             = 256;
    const auto numBlocks            = handle.GetMaxComputeUnits() * 6;
    const size_t numGlobalWorkItems = blockSize * numBlocks;

    const int computeStats = (mode & CheckNumerics::ComputeStats);

    CheckNumericsResult abnormal_h;

    auto abnormal_d =
        handle.Create(sizeof(CheckNumericsResult)); // TODO - someday avoid slow malloc/free here
    handle.WriteTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult));

    std::string params            = GetDataTypeKernelParams(dDesc.GetType());
    std::string program_name      = "MIOpenCheckNumerics.cl";
    std::string kernel_name       = "MIOpenCheckNumerics";
    const std::vector<size_t> vld = {size_t{blockSize}, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {numGlobalWorkItems, size_t{1}, size_t{1}};
    handle.AddKernel("MIOpenCheckNumerics", "", program_name, kernel_name, vld, vgd, params)(
        data, numElements, abnormal_d.get(), computeStats);

    handle.ReadTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult));

    bool isAbnormal = (abnormal_h.hasNan != 0) || (abnormal_h.hasInf != 0);

    if(((mode & CheckNumerics::Info) != 0) || (((mode & CheckNumerics::Warn) != 0) && isAbnormal))
    {
        MIOPEN_LOG((isAbnormal ? miopen::LoggingLevel::Warning : miopen::LoggingLevel::Info),
                   (isInput ? "INPUT " : "OUTPUT")
                       << " ptr=" << data << " zeros=" << abnormal_h.hasZero
                       << " nans=" << abnormal_h.hasNan << " infs=" << abnormal_h.hasInf << "  {"
                       << dDesc << "}");
        if(computeStats != 0)
        {
            assert(numElements != 0);
            MIOPEN_LOG((isAbnormal ? miopen::LoggingLevel::Warning : miopen::LoggingLevel::Info),
                       "Stats: mean=" << (abnormal_h.sum / numElements)
                                      << " absmean=" << (abnormal_h.absSum / numElements)
                                      << " min=" << abnormal_h.min << " max=" << abnormal_h.max);
        }
    }

    if(isAbnormal)
    {

        if((mode & CheckNumerics::Throw) != 0)
        {
            if(isInput)
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on INPUT");
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on OUTPUT");
            }
        }
        if((mode & CheckNumerics::Abort) != 0)
        {
            abort();
        }
    }

    return isAbnormal;
};

// Checks data for input
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsInput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data)
{
    return checkNumericsImpl(
        handle, static_cast<int>(miopen::Value(MIOPEN_CHECK_NUMERICS{})), dDesc, data, true);
}

// Synchronizes to wait for kernel to finish, then checks data for output:
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsOutput(const Handle& handle, const TensorDescriptor& dDesc, ConstData_t data)
{
    handle.Finish();

    return checkNumericsImpl(
        handle, static_cast<int>(miopen::Value(MIOPEN_CHECK_NUMERICS{})), dDesc, data, false);
}

} // namespace miopen
