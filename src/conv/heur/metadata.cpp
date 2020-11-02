/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/conv/heur/metadata.hpp>

#include <cstring>

#ifndef MIOPEN_PARAM_NAME
#define MIOPEN_PARAM_NAME fwd_gfx906_60
#endif

#define BIN_START PPCAT(MIOPEN_PARAM_NAME, _start)
#define BIN_END PPCAT(MIOPEN_PARAM_NAME, _end)

extern char BIN_START;
extern char BIN_END;

extern "C" void* PPCAT(MIOPEN_PARAM_NAME, _getEmbeddedConstPool)(int64_t /*_*/)
{
    auto size    = (unsigned int)(&BIN_END - &BIN_START);
    void* buffer = malloc(size);
    memcpy(buffer, &BIN_START, size);
    return buffer;
}

namespace miopen
{

const std::vector<std::string>& GetFeatureNames(const Handle& /*handle*/)
{
    static const std::vector<std::string> feature_names = {"batchsize",
                                                           "conv_stride_h",
                                                           "conv_stride_w",
                                                           "fil_h",
                                                           "fil_w",
                                                           "in_channels",
                                                           "in_h",
                                                           "in_w",
                                                           "out_channels",
                                                           "pad_h",
                                                           "pad_w"};
    return feature_names;
}

const std::unordered_map<int, std::string>& GetSolverMap(const Handle& /*handle*/, const ProblemDescription& /*problem*/)
{
static const std::unordered_map<int, std::string> solver_map = {
    {0, "ConvAsmImplicitGemmV4R1DynamicFwd"},
    {1, "ConvAsmImplicitGemmV4R1DynamicFwd_1x1"},
    {2, "ConvBinWinograd3x3U"},
    {3, "ConvBinWinogradRxS"},
    {4, "ConvBinWinogradRxSf2x3"},
    {5, "ConvBinWinogradRxSf3x2"},
    {6, "gemm"}};
    return solver_map;
}

const std::vector<float>& GetMu(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    const auto& num_cu = handle.GetMaxComputeUnits();

    if(arch == "gfx906" && num_cu == 60 && problem.direction.IsForward()) 
    {
        static const std::vector<float> fwd_gfx906_60_mu = {3.56184159,
                                         1.04910378,
                                         1.04910378,
                                         1.845967,
                                         1.845967,
                                         7.21834651,
                                         7.34757489,
                                         7.35721592,
                                         6.44103698,
                                         0.06920596,
                                         0.06920596};
        return fwd_gfx906_60_mu;
    }
    else 
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}
const std::vector<float>& GetSig(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    const auto& num_cu = handle.GetMaxComputeUnits();

    if(arch == "gfx906" && num_cu == 60 && problem.direction.IsForward()) 
    {
        static const std::vector<float> fwd_gfx906_60_sig = {2.64897257,
                                          0.2160847,
                                          0.2160847,
                                          1.05505271,
                                          1.05505271,
                                          2.91841399,
                                          1.92969767,
                                          1.93321703,
                                          2.8480272,
                                          0.31618387,
                                          0.31618387};
        return fwd_gfx906_60_sig;
    }
    else 
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}
} // namespacce miopen
