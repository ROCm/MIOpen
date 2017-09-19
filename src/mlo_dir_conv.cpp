/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define MIOPEN

#include <cmath>
#include <iomanip>
#include <sstream>
#include <miopen/solver.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>

#include <cstring>
#include <unordered_map>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)

bool mlo_construct_direct2D::mloIsCompilerWorkarounds() const
{
    bool ret = false;
    return ret;
}

/************************************************************************************************************************
 **
 **			CONSTRUCT CONVOLUTIONAL LAYER
 **
 ************************************************************************************************************************/

/*
   construction has been split into 2
   generic convlution forward
   non-generic stride = 1, forward and backward
   */
int mlo_construct_direct2D::mloConstruct()
{
    const auto no_perf_filtering =
        miopen::IsDisabled(MIOPEN_DEBUG_AMD_ASM_KERNELS_PERF_FILTERING{});

    _search_params.use_binaries = false;
    _search_params.assembler_available = false;
    _search_params.rmv = V3;
    if (mloIsAmdOpenclRocm(_search_params.rmv)) {
        _search_params.assembler_available = !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) &&
                                             ValidateGcnAssembler();
#ifndef HIP_OC_FINALIZER
        _search_params.use_binaries = !miopen::IsDisabled(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES{});
#endif
    }

    for(const miopen::solver::Solver& solver : SolverStore())
    {
        if(solver.IsApplicable(_search_params) &&
           (no_perf_filtering || solver.IsFast(_search_params)))
        {
            const auto perfConfig =
                solver.Find(_search_params);
            miopen::solver::ConvSolution solution;
            solver.GetSolution(solution, _search_params, *perfConfig);

            if(!solution.Succeeded())
                continue;
            if(_search_params.n_passes)
                return solution.passes;

            mloUseSolution(solution);
            return 0;
        }
    }

    return -1;
}

template <class TInstance>
class StaticContainer
{
    public:
    inline static TInstance& Instance()
    {
        static TInstance data{};
        return data;
    }
};

const std::vector<std::reference_wrapper<const miopen::solver::Solver>>&
mlo_construct_direct2D::SolverStore() const
{
    static const std::vector<
        std::reference_wrapper<const miopen::solver::Solver>>
        store({
            StaticContainer<const miopen::solver::ConvAsm3x3U>::Instance(),
            StaticContainer<const miopen::solver::ConvAsm5x10u2v2f1>::Instance(),
            StaticContainer<const miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1>::Instance(),
            StaticContainer<const miopen::solver::ConvAsm5x10u2v2b1>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwd11x11>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwdGen>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwd3x3>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwd1x1>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwdC>::Instance(),
            StaticContainer<const miopen::solver::ConvOclDirectFwd>::Instance(),
        });

    return store;
}

const std::vector<std::reference_wrapper<const miopen::solver::Solver>>&
mlo_construct_winograd::SolverStore() const
{
    static const std::vector<
        std::reference_wrapper<const miopen::solver::Solver>>
        store({
            StaticContainer<const miopen::solver::ConvBinWinograd3x3U>::Instance(),
            StaticContainer<const miopen::solver::ConvBinWinogradRxSFwd>::Instance(),
        });

    return store;
}

const std::vector<std::reference_wrapper<const miopen::solver::Solver>>&
mlo_construct_BwdWrW2D::SolverStore() const
{
    static const std::vector<
        std::reference_wrapper<const miopen::solver::Solver>>
        store({
            StaticContainer<const miopen::solver::ConvAsmBwdWrW3x3>::Instance(),
            StaticContainer<const miopen::solver::ConvOclBwdWrW2>::Instance(),
            StaticContainer<const miopen::solver::ConvOclBwdWrW53>::Instance(),
            StaticContainer<const miopen::solver::ConvOclBwdWrW1x1>::Instance(),
        });

    return store;
}

void mlo_construct_direct2D::mloUseSolution(
    const miopen::solver::ConvSolution& s)
{
    _comp_options = s.construction_params[0].comp_options;
    _kernel_file  = s.construction_params[0].kernel_file;
    _kernel_name  = s.construction_params[0].kernel_name;
    _g_wk         = s.construction_params[0].g_wk;
    _l_wk         = s.construction_params[0].l_wk;

    _workspce_sz     = s.workspce_sz;
    _grp_tile0       = s.grp_tile0;
    _grp_tile1       = s.grp_tile1;
    _in_tile0        = s.in_tile0;
    _in_tile1        = s.in_tile1;
    _out_pix_tile0   = s.out_pix_tile0;
    _out_pix_tile1   = s.out_pix_tile1;
    _n_out_pix_tiles = s.n_out_pix_tiles;
    _n_in_data_tiles = s.n_in_data_tiles;
    _n_stacks        = s.n_stacks;

    for(const auto& params : s.construction_params)
    {
        _mlo_kernels_info.push_back(std::make_tuple(
            params.kernel_name, params.kernel_file, params.comp_options, params.g_wk, params.l_wk));
    }
}

#if MIOPEN_BACKEND_OPENCL
static bool IsTokenInOpenclDriverVersion(const std::string& driver_version, const std::string& s)
{
    // Assume "(, )" are token separators in Driver Version string.
    return (driver_version.find('(' + s + ')') != std::string::npos) ||
           (driver_version.find('(' + s + ',') != std::string::npos) ||
           (driver_version.find('(' + s + ' ') != std::string::npos) ||
           (driver_version.find(',' + s + ')') != std::string::npos) ||
           (driver_version.find(',' + s + ',') != std::string::npos) ||
           (driver_version.find(',' + s + ' ') != std::string::npos) ||
           (driver_version.find(' ' + s + ')') != std::string::npos) ||
           (driver_version.find(' ' + s + ',') != std::string::npos) ||
           (driver_version.find(' ' + s + ' ') != std::string::npos);
}
#endif
bool mlo_construct_direct2D::mloIsAmdOpenclRocm(rocm_meta_version& rmv) const
{
#if MIOPEN_BACKEND_OPENCL
    const auto dev = miopen::GetDevice(_search_params.GetStream().GetStream());

    // Only suitable Opencl platform is from AMD.
    const auto platform        = miopen::GetDeviceInfo<CL_DEVICE_PLATFORM>(dev);
    const auto platform_vendor = miopen::GetPlatformInfo<CL_PLATFORM_VENDOR>(platform);
    if(platform_vendor != "Advanced Micro Devices, Inc.")
    {
        return false;
    }

    // Only AMD devices is suitable
    const auto device_vendor_id = miopen::GetDeviceInfo<CL_DEVICE_VENDOR_ID>(dev);
    if(device_vendor_id != 0x1002)
    {
        return false;
    }

    // Our binaries are in OpenCL-on-ROCm Code Object format.
    // OpenCL-on-ROCm uses Lightning Compiler.
    const auto driver_version = miopen::GetDeviceInfo<CL_DRIVER_VERSION>(dev);
    if(!IsTokenInOpenclDriverVersion(driver_version, "LC"))
    {
        return false;
    }

    // At once, extract version of OpenCL metadata. Keep rmv unchanged if extraction fails.
    const std::string platform_version = miopen::GetPlatformInfo<CL_PLATFORM_VERSION>(
        platform); // e.g. "OpenCL 2.0 AMD-APP.internal (2334.0)"
    size_t num_begin = platform_version.find('(');
    if(num_begin != std::string::npos)
    {
        int num = std::stoi(platform_version.substr(num_begin + 1));
        if(num < 2338)
        {
            rmv = V1; // Switched to V2 somewhere within [2337,2338]
        }
        else if(num < 2389)
        {
            rmv = V2; // Switched to V3 somewhere within [2388,2389]
        }
        else
        {
            rmv = V3;
        }
    }
    return true;
#else
    (void)rmv; // We don't care about metada version
    return true;
#endif // MIOPEN_BACKEND_OPENCL
}

bool mlo_construct_BwdWrW2D::mloIsCompilerWorkarounds() const
{
    bool ret = false;
    ret =
        (_search_params.in_height == 227 && _search_params.in_width == 227 &&
         _search_params.n_inputs == 1 && _search_params.kernel_size0 == 3 &&
         _search_params.kernel_size1 == 3 && _search_params.pad0 == 1 && _search_params.pad1 == 1 &&
         _search_params.kernel_stride0 == 1 && _search_params.kernel_stride1 == 1) ||
        (_search_params.in_height == 231 && _search_params.in_width == 231 &&
         _search_params.n_inputs == 1 && _search_params.kernel_size0 == 3 &&
         _search_params.kernel_size1 == 3 && _search_params.pad0 == 1 && _search_params.pad1 == 1 &&
         _search_params.kernel_stride0 == 1 && _search_params.kernel_stride1 == 1);

    return ret;
}

bool mlo_construct_direct2D::mloIsFastBinaryWinograd3x3U() const
{
    return StaticContainer<const miopen::solver::ConvBinWinograd3x3U>::Instance().IsFast(_search_params);
}

int mlo_construct_BwdWrW2D::mloMultiStep()
{
    _search_params.n_passes = true;
    const auto ret          = mloConstruct();
    _search_params.n_passes = false;

    return (ret);
}

/***********************************************************************************************************

 * Internal implementation of the direct conv configuration search

 ************************************************************************************************************/

/*
   the search db is a text file with the name defined by the device characteristics.
   each line is a key/value pair, separated by a space:
   32x16x16x3x3x64x16x16x100xNCHWxFP32x1 16.16.16.16.1.4.8.4.1
   or
   64x8x8x5x5x32x8x8x100xNCHWxFP32x0 16.16.8.8.2.4.1.1.4

   key format (all values are separted by x):
   n input maps
   input height
   input width
   filter height
   filter width
   n output maps
   output height
   output width
   batch size
   tensors' layout
   tensprs' data type
   direction (1 - forward, 0 - backward)

Note:
for backward direction - input and output are reversed.

value format (all values are separated by .):
vertical group size
horizontal group size
input block vertical size
input block horizontal size
output tile vertical size
output tile horizaontal size
n of output tiles
n of input blocks
n batchs (stacks) processed by the group
*/

int mlo_construct_direct2D::mloBuildConf_Key(std::string& conf_key) const
{

    conf_key =
        std::to_string(static_cast<long long>(_search_params.n_inputs)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.in_height)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.in_width)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.kernel_size1)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.kernel_size0)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.n_outputs)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.out_height)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.out_width)) + std::string("x") +
        std::to_string(static_cast<long long>(_search_params.batch_sz)) + std::string("x") +
        _search_params.in_layout + std::string("x") + _search_params.in_data_type +
        std::string("x") + std::to_string(static_cast<long long>(_search_params.forward));
    return (0);
}

// Tensor Helper APIs

size_t
mlo_construct_direct2D::setWeightDescFromMLDesc(const miopen::TensorDescriptor& weight_tensor)
{

    int nWei;
    int cWei;
    int hWei;
    int wWei;
    int nWeiStride;
    int cWeiStride;
    int hWeiStride;
    int wWeiStride;

    std::tie(nWei, cWei, hWei, wWei) = miopen::tie4(weight_tensor.GetLengths());
    std::tie(nWeiStride, cWeiStride, hWeiStride, wWeiStride) =
        miopen::tie4(weight_tensor.GetStrides());

    setWeightsDescr(
        "NCHW", "FP32", nWei, cWei, hWei, wWei, nWeiStride, cWeiStride, hWeiStride, wWeiStride);

    size_t weights_sz = nWei * cWei * hWei * wWei * sizeof(float);
    return weights_sz;
}

size_t
mlo_construct_direct2D::setOutputDescFromMLDesc(const miopen::TensorDescriptor& output_tensor)
{

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut) = miopen::tie4(output_tensor.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) =
        miopen::tie4(output_tensor.GetStrides());

    setOutputDescr(
        "NCHW", "FP32", nOut, cOut, hOut, wOut, nOutStride, cOutStride, hOutStride, wOutStride);

    size_t output_sz = nOut * cOut * hOut * wOut * sizeof(float);
    return output_sz;
}

size_t mlo_construct_direct2D::setInputDescFromMLDesc(const miopen::TensorDescriptor& input_tensor)
{

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = miopen::tie4(input_tensor.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = miopen::tie4(input_tensor.GetStrides());

    setInputDescr("NCHW", "FP32", nIn, cIn, hIn, wIn, nInStride, cInStride, hInStride, wInStride);

    size_t input_sz = nIn * cIn * hIn * wIn * sizeof(float);

    return input_sz;
}
