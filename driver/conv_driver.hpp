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
#ifndef GUARD_MIOPEN_CONV_DRIVER_HPP
#define GUARD_MIOPEN_CONV_DRIVER_HPP

/*#ifdef MIOPEN_NEURON_SOFTRELU /// \todo This needs to be explained or rewritten in clear manner.
#undef MIOPEN_NEURON_SOFTRELU
#endif

#ifdef MIOPEN_NEURON_POWER /// \todo This needs to be explained or rewritten in clear manner.
#undef MIOPEN_NEURON_POWER
#endif
*/
#include "InputFlags.hpp"
#include "conv_verify.hpp"
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <fstream>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/miopen_internal.h>
#include <miopen/tensor.hpp>
#include <miopen/env.hpp>
#include <miopen/algorithm.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/logger.hpp>
#include <miopen/convolution.hpp>
#include <miopen/solver.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/problem_description.hpp>
#include "random.hpp"
#include <numeric>
#include <sstream>
#include <vector>
#include <type_traits>
#include <boost/range/adaptors.hpp>
#include <boost/optional/optional_io.hpp>
#include <../test/verify.hpp>
#include <../test/serialize.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/cpu_conv.hpp>
#include <../test/cpu_bias.hpp>

#include <boost/optional.hpp>

// Declare hidden function for MIGraphX to smoke test it.
extern "C" miopenStatus_t miopenHiddenSetConvolutionFindMode(miopenConvolutionDescriptor_t convDesc,
                                                             int findMode);

#define WORKAROUND_ISSUE_2176 1 // https://github.com/AMDComputeLibraries/MLOpen/issues/2176

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_PAD_BUFFERS_2M)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DRIVER_USE_GPU_REFERENCE)

#if MIOPEN_BACKEND_OPENCL
#define STATUS_SUCCESS CL_SUCCESS
typedef cl_int status_t;
#else // MIOPEN_BACKEND_HIP
#define STATUS_SUCCESS 0
typedef int status_t;
#endif

struct AutoMiopenWarmupMode
{
    AutoMiopenWarmupMode()
    {
        debug_logging_quiet_prev          = miopen::debug::LoggingQuiet;
        debug_find_enforce_disable_prev   = miopen::debug::FindEnforceDisable;
        miopen::debug::LoggingQuiet       = true;
        miopen::debug::FindEnforceDisable = true;
    }
    AutoMiopenWarmupMode(const AutoMiopenWarmupMode&) = delete;
    AutoMiopenWarmupMode(AutoMiopenWarmupMode&&)      = delete;
    AutoMiopenWarmupMode& operator=(const AutoMiopenWarmupMode&) = delete;
    AutoMiopenWarmupMode& operator=(AutoMiopenWarmupMode&&) = delete;
    ~AutoMiopenWarmupMode()
    {
        miopen::debug::LoggingQuiet       = debug_logging_quiet_prev;
        miopen::debug::FindEnforceDisable = debug_find_enforce_disable_prev;
    }

private:
    bool debug_logging_quiet_prev;
    bool debug_find_enforce_disable_prev;
};

struct AutoConvDirectNaiveAlwaysEnable
{
    AutoConvDirectNaiveAlwaysEnable()
    {
        prev                                       = miopen::debug::AlwaysEnableConvDirectNaive;
        miopen::debug::AlwaysEnableConvDirectNaive = true;
    }
    ~AutoConvDirectNaiveAlwaysEnable() { miopen::debug::AlwaysEnableConvDirectNaive = prev; }

private:
    bool prev;
};

template <typename T>
void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        printf("Wrote output to file %s\n", fileName);
    }
    else
    {
        printf("Could not open file %s for writing\n", fileName);
    }
}

template <typename T>
bool readBufferFromFile(T* data, size_t dataNumItems, const char* fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        infile.close();
        printf("Read data from input file %s\n", fileName);
        return true;
    }
    else
    {
        printf("Could not open file %s for reading\n", fileName);
        return false;
    }
}

// Tgpu and Tref are the data-type in GPU memory and CPU memory respectively.
// They are not necessarily the same as the computation type on GPU or CPU
template <typename Tgpu, typename Tref>
class ConvDriver : public Driver
{
#if MIOPEN_BACKEND_OPENCL
    typedef cl_context context_t;
#elif MIOPEN_BACKEND_HIP
    typedef uint32_t context_t;
#endif
public:
    ConvDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&biasTensor);
        miopenCreateTensorDescriptor(&inputTensor_vect4);
        miopenCreateTensorDescriptor(&weightTensor_vect4);
        miopenCreateConvolutionDescriptor(&convDesc);

        {
            AutoMiopenWarmupMode warmupMode;
            miopenCreateTensorDescriptor(&warmupInputTensor);
            miopenCreateTensorDescriptor(&warmupWeightTensor);
            miopenCreateTensorDescriptor(&warmupOutputTensor);
            miopenCreateConvolutionDescriptor(&warmupConvDesc);
        }

        workspace_dev = nullptr;
        // the variable name is implementation dependent, checking size instead
        InitDataType<Tgpu>();
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    // function to validate the Layout type parameters.
    // Layout types are -In,Out,Fil etc.This function validates the
    // layout parameter value to std (NCHW/NHWC/NCDHW/NDHWC) values,
    // defined in MIOpen lib.
    // layout_type - input value supplied with MIOpen driver command.
    void ValidateLayoutInputParameters(std::string layout_type);
    void ValidateVectorizedParameters(int vector_dim, int vector_length);

    // Helper function to check the Layout type short names
    // Short names are defined as I,O,f. W.r.t In/Out/fil layout
    // types.
    int ChkLayout_ShortName();

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetWeightTensorLengthsFromCmdLine();
    std::vector<int> GetBiasTensorLengthsFromCmdLine();

    int SetConvDescriptorFromCmdLineArgs();

    std::vector<int> GetOutputTensorLengths();

    int AllocateBuffersAndCopy() override;

    bool UseGPUReference();

    int FindForward(int& ret_algo_count,
                    int request_algo_count,
                    std::vector<miopenConvAlgoPerf_t>& perf_results,
                    context_t ctx);
    int RunForwardGPU() override;
    int RunForwardCPU();
    int RunForwardGPUReference();
    int RunWarmupFindForwardGPU();

    int FindBackwardData(int& ret_algo_count,
                         int request_algo_count,
                         std::vector<miopenConvAlgoPerf_t>& perf_results,
                         context_t ctx);
    int FindBackwardWeights(int& ret_algo_count,
                            int request_algo_count,
                            std::vector<miopenConvAlgoPerf_t>& perf_results,
                            context_t ctx);
    int RunBackwardGPU() override;
    int RunBackwardDataCPU();
    int RunBackwardWeightsCPU();
    int RunBackwardBiasCPU();
    int RunBackwardDataGPUReference();
    int RunBackwardWeightsGPUReference();
    // int RunBackwardBiasGPUReference();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~ConvDriver() override
    {
        miopenDestroyTensorDescriptor(biasTensor);
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(inputTensor_vect4);
        miopenDestroyTensorDescriptor(weightTensor_vect4);
        miopenDestroyConvolutionDescriptor(convDesc);

        miopenDestroyTensorDescriptor(warmupInputTensor);
        miopenDestroyTensorDescriptor(warmupWeightTensor);
        miopenDestroyTensorDescriptor(warmupOutputTensor);
        miopenDestroyConvolutionDescriptor(warmupConvDesc);
    }

private:
    const miopenDataType_t warmup_data_type = miopenFloat;
    typedef float warmup_Tgpu;

    InputFlags inflags;

    boost::optional<uint64_t> immediate_solution;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t outputTensor;
    miopenTensorDescriptor_t biasTensor;
    miopenTensorDescriptor_t inputTensor_vect4;
    miopenTensorDescriptor_t weightTensor_vect4;
    miopenTensorDescriptor_t warmupInputTensor;
    miopenTensorDescriptor_t warmupWeightTensor;
    miopenTensorDescriptor_t warmupOutputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> in_vect4_dev;
    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> wei_dev;
    std::unique_ptr<GPUMem> wei_vect4_dev;
    std::unique_ptr<GPUMem> dwei_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> dout_dev;
    std::unique_ptr<GPUMem> b_dev;
    std::unique_ptr<GPUMem> db_dev;
    std::unique_ptr<GPUMem> warmup_in_dev;
    std::unique_ptr<GPUMem> warmup_wei_dev;
    std::unique_ptr<GPUMem> warmup_out_dev;

    std::unique_ptr<GPUMem> workspace_dev;
    std::size_t ws_sizeof_find_fwd;
    std::size_t ws_sizeof_find_bwd;
    std::size_t ws_sizeof_find_wrw;
    std::size_t warmup_ws_sizeof_find;

    tensor<Tgpu> in;
    tensor<Tgpu> wei;
    tensor<Tgpu> out;
    tensor<Tgpu> dout;
    tensor<Tgpu> b;
    tensor<Tref> outhost;
    tensor<Tref> dwei_host;
    tensor<Tref> din_host;
    tensor<Tref> db_host;
    tensor<warmup_Tgpu> warmup_in;
    tensor<warmup_Tgpu> warmup_wei;
    tensor<warmup_Tgpu> warmup_out;

    std::vector<Tgpu> din;
    std::vector<Tgpu> dwei;
    std::vector<int32_t> out_int8;
    std::vector<Tgpu> db;
    std::vector<float> b_int8;

    miopenConvolutionDescriptor_t convDesc;
    miopenConvolutionDescriptor_t warmupConvDesc;

    bool is_wrw = true, is_bwd = true, is_fwd = true;
    bool is_wrw_winograd = false;
    bool is_wrw_igemm    = false;
    bool is_fwd_igemm    = false;
    bool is_bwd_igemm    = false;
    bool time_enabled    = false;
    bool wall_enabled    = false;
    bool warmup_enabled  = false;
    int num_iterations   = 1;

    // Used to avoid wasting time for verification after failure of Run*GPU().
    // We can't properly control this from the main() level.
    // RunBackwardGPU() and VerifyBackward() do the job for both Bwd and WrW.
    // If RunBackwardGPU() fails, then main() doesn't know if Bwd or WrW has failed.
    // Also main() has no ways to for controlling how Verify works except skipping the whole call.
    bool is_fwd_run_failed = false, is_bwd_run_failed = false, is_wrw_run_failed = false;

    Timer wall;
    Timer2 fwd_auxiliary;
    Timer2 bwd_auxiliary;
    Timer2 wrw_auxiliary;
    Timer2 fwd_auxiliary_gwss;
    Timer2 bwd_auxiliary_gwss;
    Timer2 wrw_auxiliary_gwss;
    Timer2 warmup_wall_total; // Counts also auxiliary time.

    void PrintForwardTime(float kernel_total_time, float kernel_first_time) const;
    int RunForwardGpuImmed(bool is_transform);
    int RunForwardGpuFind(bool is_transform);
    void PrintBackwardDataTime(float kernel_total_time, float kernel_first_time);
    int RunBackwardDataGpuImmed();
    int RunBackwardDataGpuFind();
    void PrintBackwardWrwTime(float kernel_total_time, float kernel_first_time);
    int RunBackwardWrwGpuImmed();
    int RunBackwardWrwGpuFind();

    double GetDefaultTolerance() const
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        auto tolerance = (sizeof(Tgpu) == 4 || sizeof(Tgpu) == 1) ? 1.5e-6 : 8.2e-3;
        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<Tgpu, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    enum class Direction
    {
        Fwd,
        Bwd,
        WrW,
        BwdBias
    };

    std::string GetVerificationCacheFileName(const Direction& direction) const;
    bool IsInputTensorTransform() const;

    bool TryReadVerificationCache(const Direction& direction,
                                  miopenTensorDescriptor_t& tensorDesc,
                                  Tref* data) const;
    void TrySaveVerificationCache(const Direction& direction, std::vector<Tref>& data) const;

    void ResizeWorkspaceDev(context_t ctx, std::size_t size)
    {
        workspace_dev.reset();
        if(size > 0)
            workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, size, 1));
    }

    // Helper functions, can be moved out of class.
    void PrintImmedSolutionInfo(const miopenConvSolution_t& s) const
    {
        std::cout << "- id: " << s.solution_id << " algo: " << s.algorithm << ", time: " << s.time
                  << " ms, ws: " << s.workspace_size
                  << ", name: " << miopen::solver::Id(s.solution_id).ToString() << std::endl;
    }

    std::string AlgorithmSolutionToString(const miopenConvSolution_t& s) const
    {
        std::ostringstream oss;
        oss << "Algorithm: " << s.algorithm << ", Solution: " << s.solution_id << '/'
            << ((s.solution_id != 0) ? miopen::solver::Id(s.solution_id).ToString()
                                     : std::string("UNKNOWN"));
        return oss.str();
    }

    /// Find() updates find-db with the most recent information (unless find-db is disabled).
    /// Therefore, after Find(), Immediate mode returns the "best" found solution
    /// as the 1st solution in the list, and we can use Immediate mode to find out
    /// the name of the Solver selected during Find() and then used in Run().
    void GetSolutionAfterFind(const miopenConvAlgoPerf_t& found,
                              const Direction& direction,
                              const miopenTensorDescriptor_t& inTensor,
                              const miopenTensorDescriptor_t& weiTensor,
                              const miopenTensorDescriptor_t& outTensor,
                              miopenConvSolution_t& solution);
};

// Check if int8 type tensor x and w need to be transformed to a pack of 4 elements along channel
// (NCHW_VECT_C format)
template <typename Tgpu, typename Tref>
bool ConvDriver<Tgpu, Tref>::IsInputTensorTransform() const
{
    return (data_type == miopenInt8 && inflags.GetValueInt("in_channels") % 4 != 0) ||
           data_type == miopenInt8x4;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{

    inflags.Parse(argc, argv);

    // try to set a default layout value for 3d conv if not specified from cmd line
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    const std::string default_layout = (spatial_dim == 2) ? "NCHW" : "NCDHW";

    // inflags value is empty, default value is used
    // if it is supplied via cmd line, check the value.
    if(inflags.GetValueStr("in_layout").empty())
    {
        inflags.SetValue("in_layout", default_layout);
    }
    else
    {
        std::string in_layoutValue = inflags.GetValueStr("in_layout");
        ValidateLayoutInputParameters(in_layoutValue);
        inflags.SetValue("in_layout", in_layoutValue);
    }
    // fil layout argument value check
    if(inflags.GetValueStr("fil_layout").empty())
    {
        inflags.SetValue("fil_layout", default_layout);
    }
    else
    {
        std::string fil_layoutValue = inflags.GetValueStr("fil_layout");
        ValidateLayoutInputParameters(fil_layoutValue);
        inflags.SetValue("fil_layout", fil_layoutValue);
    }
    // out layout argument check
    if(inflags.GetValueStr("out_layout").empty())
    {
        inflags.SetValue("out_layout", default_layout);
    }
    else
    {
        std::string out_layoutValue = inflags.GetValueStr("out_layout");
        ValidateLayoutInputParameters(out_layoutValue);
        inflags.SetValue("out_layout", out_layoutValue);
    }

    // vectorized tensor Dimension & Length check
    int vector_dim    = inflags.GetValueInt("tensor_vect");
    int vector_length = inflags.GetValueInt("vector_length");

    ValidateVectorizedParameters(vector_dim, vector_length);
    if(vector_length != 1 && vector_dim == 1)
    {
        inflags.SetValue("in_layout",
                         inflags.GetValueStr("in_layout") + "c" + std::to_string(vector_length));
        inflags.SetValue("fil_layout",
                         inflags.GetValueStr("fil_layout") + "c" + std::to_string(vector_length));
        inflags.SetValue("out_layout",
                         inflags.GetValueStr("out_layout") + "c" + std::to_string(vector_length));
    }

    num_iterations = inflags.GetValueInt("iter");
    if(num_iterations < 1)
    {
        std::cout << "Fatal: Number of iterations must be > 0: " << num_iterations << std::endl;
        return 1;
    }
    time_enabled = (inflags.GetValueInt("time") != 0);
    {
        const int val = inflags.GetValueInt("wall");
        if(val >= 1)
        {
            if(!time_enabled)
            {
                std::cout << "Info: '--wall " << val << "' is ignored because '--time' is not set"
                          << std::endl;
            }
            else
            {
                wall_enabled   = (val >= 1);
                warmup_enabled = (val >= 2);
            }
        }
    }

    if(time_enabled)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    is_fwd = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 1);
    is_bwd = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 2);
    is_wrw = (inflags.GetValueInt("forw") == 0 || inflags.GetValueInt("forw") & 4);

    const auto solution_str = inflags.GetValueStr("solution");
    auto solution_value     = static_cast<int>(miopen::solver::Id(solution_str.c_str()).Value());
    if(solution_value == 0) // Assume number on input
    {
        solution_value = std::strtol(solution_str.c_str(), nullptr, 10);
        if(errno == ERANGE)
        {
            errno          = 0;
            solution_value = 0;
        }
    }
    if(solution_value >= 0)
        immediate_solution = solution_value;

    return 0;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::ValidateLayoutInputParameters(std::string layout_value)
{
    if((ChkLayout_ShortName()))
    {
        std::cerr << " Invalid Layout Short Name = " << ChkLayout_ShortName() << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        if((layout_value.compare("NCHW") == 0) || (layout_value.compare("NHWC") == 0) ||
           (layout_value.compare("CHWN") == 0) || (layout_value.compare("NCDHW") == 0) ||
           (layout_value.compare("NDHWC") == 0))
        {
            // do nothing,Values are matching as defined in Lib.
        }
        else
        {
            std::cerr << "Invalid Layout Parameter Value - " << layout_value << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::ValidateVectorizedParameters(int vector_dim, int vector_length)
{
    if(((vector_length == 4 || vector_length == 8) && vector_dim == 1) ||
       (vector_length == 1 && vector_dim == 0))
    {
        // do nothing,Values are matching as defined in Lib.
    }
    else
    {
        std::cerr << "Invalid Tensor Vectorization Parameter Value - "
                  << "vector_dim:" << vector_dim << ", vector_length:" << vector_length
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::ChkLayout_ShortName()
{
    // check for short name of layout type
    if((inflags.FindShortName("in_layout") == 'I') &&
       (inflags.FindShortName("out_layout") == 'O') && (inflags.FindShortName("fil_layout") == 'f'))
    {
        // do noting
        // found valid short names
        return 0;
    }
    else
    {
        std::cerr << "Error:Invalid Short Name!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

    SetTensorNd(inputTensor, in_len, inflags.GetValueStr("in_layout"), data_type);
    SetTensorNd(weightTensor, wei_len, inflags.GetValueStr("fil_layout"), data_type);

    if(inflags.GetValueInt("tensor_vect") == 1 && data_type == miopenInt8)
    {
        data_type = miopenInt8x4;
    }

    if(IsInputTensorTransform())
    {
        std::vector<int> in_len_vect4(in_len.begin(), in_len.end()),
            wei_len_vect4(wei_len.begin(), wei_len.end());
        in_len_vect4[1] = ((in_len[1] + 3) / 4) * 4;
        SetTensorNd(inputTensor_vect4, in_len_vect4, data_type);
        wei_len_vect4[1] = ((wei_len[1] + 3) / 4) * 4;
        SetTensorNd(weightTensor_vect4, wei_len_vect4, data_type);
    }
    SetConvDescriptorFromCmdLineArgs();

    std::vector<int> out_len = GetOutputTensorLengths();
    if(miopen::deref(inputTensor).GetLayout_t() == miopenTensorNCHWc4 ||
       miopen::deref(inputTensor).GetLayout_t() == miopenTensorNCHWc8)
    {
        out_len[1] *= miopen::deref(inputTensor).GetVectorLength();
    }
    if(miopen::deref(inputTensor).GetLayout_t() == miopenTensorCHWNc4 ||
       miopen::deref(inputTensor).GetLayout_t() == miopenTensorCHWNc8)
    {
        out_len[0] *= miopen::deref(inputTensor).GetVectorLength();
    }
    miopenDataType_t y_type =
        (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenInt32 : data_type;
    SetTensorNd(outputTensor, out_len, inflags.GetValueStr("out_layout"), y_type);

    if(inflags.GetValueInt("bias") != 0)
    {
        std::vector<int> bias_len = GetBiasTensorLengthsFromCmdLine();
        SetTensorNd(biasTensor, bias_len, data_type);
    }

    if(warmup_enabled)
    {
        AutoMiopenWarmupMode warmupMode;
        std::vector<int> warmup_in_len  = {1, 1, 16, 16}; // NCHW
        std::vector<int> warmup_wei_len = {1, 1, 1, 1};   // KCYX
        SetTensorNd(warmupInputTensor, warmup_in_len, warmup_data_type);
        SetTensorNd(warmupWeightTensor, warmup_wei_len, warmup_data_type);

        const int spatial_dim           = 2;
        const int group_count           = 1;
        std::vector<int> pads           = {0, 0};
        std::vector<int> conv_strides   = {1, 1};
        std::vector<int> conv_dilations = {1, 1};
        miopenConvolutionMode_t mode    = miopenConvolution;
        miopenInitConvolutionNdDescriptor(warmupConvDesc,
                                          spatial_dim,
                                          pads.data(),
                                          conv_strides.data(),
                                          conv_dilations.data(),
                                          mode);
        miopenSetConvolutionFindMode(warmupConvDesc, miopenConvolutionFindModeNormal);
        miopenHiddenSetConvolutionFindMode(
            warmupConvDesc,
            static_cast<int>(miopenConvolutionFindModeNormal)); // Repeat via hidden API.
        miopenSetConvolutionGroupCount(warmupConvDesc, group_count);

        int warmup_out_len_size = miopen::deref(warmupInputTensor).GetSize();
        std::vector<int> warmup_out_len(warmup_out_len_size);
        miopenGetConvolutionNdForwardOutputDim(warmupConvDesc,
                                               warmupInputTensor,
                                               warmupWeightTensor,
                                               &warmup_out_len_size,
                                               warmup_out_len.data());
        SetTensorNd(warmupOutputTensor, warmup_out_len, warmup_data_type);
    }
    return (0);
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::AddCmdLineArgs()
{

    inflags.AddInputFlag("in_layout",
                         'I',
                         "",
                         "Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag("out_layout",
                         'O',
                         "",
                         "Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag("fil_layout",
                         'f',
                         "",
                         "Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag(
        "spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Flag enables fwd, bwd, wrw convolutions"
                         "\n0 fwd+bwd+wrw (default)"
                         "\n1 fwd only"
                         "\n2 bwd only"
                         "\n4 wrw only"
                         "\n3 fwd+bwd"
                         "\n5 fwd+wrw"
                         "\n6 bwd+wrw",
                         "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", '!', "32", "Input Depth (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_d", '#', "1", "Convolution Stride for Depth (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_h", 'u', "1", "Convolution Stride for Height (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_w", 'v', "1", "Convolution Stride for Width (Default=1)", "int");
    inflags.AddInputFlag("pad_d", '$', "0", "Zero Padding for Depth (Default=0)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding for Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding for Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_d", '%', "0", "Zero Padding Output for Depth (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_h", 'Y', "0", "Zero Padding Output for Height (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_w", 'X', "0", "Zero Padding Output for Width (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("wall",
                         'w',
                         "0",
                         "Wall-clock Time Each Layer"
                         "\n0 Off (Default)"
                         "\n1 On, requires '--time 1')"
                         "\n2 On, warm-up the library (prefetch db caches), requires '--time 1')",
                         "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
    inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "conv", "Convolution Mode (conv, trans) (Default=conv)", "str");

    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
    inflags.AddInputFlag("tensor_vect",
                         'Z',
                         "0",
                         "tensor vectorization type (none, vect_c, vect_n) (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "vector_length", 'L', "1", "tensor vectorization length (Default=1)", "int");
    inflags.AddInputFlag("dilation_d", '^', "1", "Dilation of Filter Depth (Default=1)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("in_bias", 'a', "", "Input bias filename (Default=)", "string");
    inflags.AddInputFlag("group_count", 'g', "1", "Number of Groups (Default=1)", "int");
    inflags.AddInputFlag("dout_data",
                         'D',
                         "",
                         "dy data filename for backward weight computation (Default=)",
                         "string");
    inflags.AddInputFlag("solution",
                         'S',
                         "-1",
                         "Use immediate mode, run solution with specified id."
                         "\nAccepts integer argument N:"
                         "\n=0 Immediate mode, build and run fastest solution"
                         "\n>0 Immediate mode, build and run solution_id = N"
                         "\n<0 Use Find() API (Default=-1)"
                         "\nAlso accepts symbolic name of solution:"
                         "\n<valid name>   Immediate mode, build and run specified solution"
                         "\n<invalid name> Use Find() API",
                         "string");

    return 0;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::vector<int> in_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = inflags.GetValueInt("batchsize");
    in_lens[1] = inflags.GetValueInt("in_channels");

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_h");
        in_spatial_lens[1] = inflags.GetValueInt("in_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_d");
        in_spatial_lens[1] = inflags.GetValueInt("in_h");
        in_spatial_lens[2] = inflags.GetValueInt("in_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetWeightTensorLengthsFromCmdLine()
{
    std::vector<int> wei_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    int wei_k_len = inflags.GetValueInt("out_channels");
    int wei_c_len = inflags.GetValueInt("in_channels");

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_w");
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2] = inflags.GetValueInt("fil_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            MIOPEN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        MIOPEN_THROW("Incorrect Convolution Mode\n");
    }

    if(mode == miopenTranspose)
    {
        wei_lens[0] = wei_c_len;
        wei_lens[1] = wei_k_len / group_count;
    }
    else
    {
        wei_lens[0] = wei_k_len;
        wei_lens[1] = wei_c_len / group_count;
    }

    return wei_lens;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetBiasTensorLengthsFromCmdLine()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = inflags.GetValueInt("out_channels");

    return bias_lens;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::SetConvDescriptorFromCmdLineArgs()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_h");
        in_spatial_lens[1]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_h");
        pads[1]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_h");
        conv_dilations[1]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_d");
        in_spatial_lens[1]   = inflags.GetValueInt("in_h");
        in_spatial_lens[2]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_d");
        pads[1]              = inflags.GetValueInt("pad_h");
        pads[2]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_d");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[2]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_d");
        conv_dilations[1]    = inflags.GetValueInt("dilation_h");
        conv_dilations[2]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_d");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[2] = inflags.GetValueInt("trans_output_pad_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    int out_c       = inflags.GetValueInt("out_channels");
    int in_c        = inflags.GetValueInt("in_channels");
    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0); // NOLINT (concurrency-mt-unsafe)
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    // adjust padding based on user-defined padding mode
    if(mode == miopenConvolution &&
       (miopen::all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        miopen::all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if((inflags.GetValueStr("pad_mode")) == "same")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] =
                    (in_spatial_lens[i] % conv_strides[i] == 0)
                        ? (std::max((wei_spatial_lens[i] - conv_strides[i]), 0))
                        : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % conv_strides[i])),
                                    0));
                pads[i] /= 2;
            }
        }
        else if((inflags.GetValueStr("pad_mode")) == "valid")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    miopenInitConvolutionNdDescriptor(
        convDesc, spatial_dim, pads.data(), conv_strides.data(), conv_dilations.data(), mode);

    miopenSetConvolutionGroupCount(convDesc, group_count);

    if(mode == miopenTranspose)
    {
        miopenSetTransposeConvNdOutputPadding(convDesc, spatial_dim, trans_output_pads.data());
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> ConvDriver<Tgpu, Tref>::GetOutputTensorLengths()
{
    int ndim = miopen::deref(inputTensor).GetSize();

    std::vector<int> out_lens(ndim);

    miopenGetConvolutionNdForwardOutputDim(
        convDesc, inputTensor, weightTensor, &ndim, out_lens.data());

    return out_lens;
}

namespace detail {

template <typename T>
T RanGenWeights()
{
    return RAN_GEN<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
float16 RanGenWeights()
{
    return RAN_GEN<float16>(static_cast<float16>(-1.0 / 3.0), static_cast<float16>(0.5));
}

} // namespace detail

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    if(wall_enabled)
    {
        fwd_auxiliary.start();
        fwd_auxiliary.pause();
        bwd_auxiliary.start();
        bwd_auxiliary.pause();
        wrw_auxiliary.start();
        wrw_auxiliary.pause();
        fwd_auxiliary_gwss.start();
        fwd_auxiliary_gwss.pause();
        bwd_auxiliary_gwss.start();
        bwd_auxiliary_gwss.pause();
        wrw_auxiliary_gwss.start();
        wrw_auxiliary_gwss.pause();
        if(warmup_enabled)
        {
            warmup_wall_total.start();
            warmup_wall_total.pause();
        }
    }

    bool is_transform = IsInputTensorTransform();
    bool is_int8      = data_type == miopenInt8 || data_type == miopenInt8x4;
    size_t in_sz      = GetTensorSize(inputTensor);
    size_t wei_sz     = GetTensorSize(weightTensor);
    size_t out_sz     = GetTensorSize(outputTensor);

    // Workaround: Pad buffers allocations to be a multiple of 2M
    if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
    {
        // PadBufferSize(in_sz, sizeof(Tgpu));
        PadBufferSize(wei_sz, sizeof(Tgpu));
        PadBufferSize(out_sz, sizeof(Tgpu));
    }

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif
    ws_sizeof_find_fwd = 0;
    ws_sizeof_find_wrw = 0;
    ws_sizeof_find_bwd = 0;

    if(warmup_enabled)
    {
        do
        {
            AutoMiopenWarmupMode warmupMode;
            size_t warmup_in_sz  = GetTensorSize(warmupInputTensor);
            size_t warmup_wei_sz = GetTensorSize(warmupWeightTensor);
            size_t warmup_out_sz = GetTensorSize(warmupOutputTensor);
            if(miopen::IsEnabled(MIOPEN_DRIVER_PAD_BUFFERS_2M{}))
            {
                PadBufferSize(warmup_wei_sz, sizeof(warmup_Tgpu));
                PadBufferSize(warmup_out_sz, sizeof(warmup_Tgpu));
            }

            warmup_ws_sizeof_find = 0;
            warmup_wall_total.resume(wall_enabled);
            miopenStatus_t rc = miopenConvolutionForwardGetWorkSpaceSize(GetHandle(),
                                                                         warmupWeightTensor,
                                                                         warmupInputTensor,
                                                                         warmupConvDesc,
                                                                         warmupOutputTensor,
                                                                         &warmup_ws_sizeof_find);
            warmup_wall_total.pause(wall_enabled);
            if(rc != miopenStatusSuccess)
            {
                std::cout << "Warm-up: Error getting workspace size, status = " << rc
                          << ". Warm-up disabled." << std::endl;
                warmup_enabled = false;
                break;
            }
            if(warmup_ws_sizeof_find != 0)
            {
                std::cout << "Warm-up: This step should not require workspace, but asks for "
                          << warmup_ws_sizeof_find << ". Warm-up disabled." << std::endl;
                warmup_enabled = false;
                break;
            }

            warmup_in  = tensor<warmup_Tgpu>(miopen::deref(warmupInputTensor).GetLengths(),
                                            miopen::deref(warmupInputTensor).GetStrides());
            warmup_wei = tensor<warmup_Tgpu>(miopen::deref(warmupWeightTensor).GetLengths(),
                                             miopen::deref(warmupWeightTensor).GetStrides());
            warmup_out = tensor<warmup_Tgpu>(miopen::deref(warmupOutputTensor).GetLengths(),
                                             miopen::deref(warmupOutputTensor).GetStrides());

            warmup_in_dev =
                std::unique_ptr<GPUMem>(new GPUMem(ctx, warmup_in_sz, sizeof(warmup_Tgpu)));
            warmup_wei_dev =
                std::unique_ptr<GPUMem>(new GPUMem(ctx, warmup_wei_sz, sizeof(warmup_Tgpu)));
            warmup_out_dev =
                std::unique_ptr<GPUMem>(new GPUMem(ctx, warmup_out_sz, sizeof(warmup_Tgpu)));

            status_t status = STATUS_SUCCESS;
            status |= warmup_in_dev->ToGPU(q, warmup_in.data.data());
            status |= warmup_wei_dev->ToGPU(q, warmup_wei.data.data());
            status |= warmup_out_dev->ToGPU(q, warmup_out.data.data());

            if(status != STATUS_SUCCESS)
            {
                std::cout << "Warm-up: Error copying data to GPU, status = " << status
                          << ". Warm-up disabled." << std::endl;
                warmup_enabled = false;
                break;
            }

            const int rcf = RunWarmupFindForwardGPU();
            if(rcf != 0)
            {
                std::cout << "Warm-up: RunWarmupFindForwardGPU() FAILED, rcf = " << rcf
                          << ". Warm-up disabled." << std::endl;
                warmup_enabled = false;
                break;
            }
        } while(false);
    }

    if(!immediate_solution)
    {
        miopenStatus_t rc = miopenStatusSuccess;
        if(is_wrw && rc == miopenStatusSuccess)
        {
            wrw_auxiliary.resume(wall_enabled);
            wrw_auxiliary_gwss.resume(wall_enabled);
            rc = miopenConvolutionBackwardWeightsGetWorkSpaceSize(GetHandle(),
                                                                  outputTensor,
                                                                  inputTensor,
                                                                  convDesc,
                                                                  weightTensor,
                                                                  &ws_sizeof_find_wrw);
            wrw_auxiliary_gwss.pause(wall_enabled);
            wrw_auxiliary.pause(wall_enabled);
        }
        if(is_bwd && rc == miopenStatusSuccess)
        {
            bwd_auxiliary.resume(wall_enabled);
            bwd_auxiliary_gwss.resume(wall_enabled);
            rc = miopenConvolutionBackwardDataGetWorkSpaceSize(GetHandle(),
                                                               outputTensor,
                                                               weightTensor,
                                                               convDesc,
                                                               inputTensor,
                                                               &ws_sizeof_find_bwd);
            bwd_auxiliary_gwss.pause(wall_enabled);
            bwd_auxiliary.pause(wall_enabled);
        }
        if(is_fwd && rc == miopenStatusSuccess)
        {
            fwd_auxiliary.resume(wall_enabled);
            fwd_auxiliary_gwss.resume(wall_enabled);
            rc = miopenConvolutionForwardGetWorkSpaceSize(
                GetHandle(),
                (is_transform ? weightTensor_vect4 : weightTensor),
                (is_transform ? inputTensor_vect4 : inputTensor),
                convDesc,
                outputTensor,
                &ws_sizeof_find_fwd);
            fwd_auxiliary_gwss.pause(wall_enabled);
            fwd_auxiliary.pause(wall_enabled);
        }
        if(rc != miopenStatusSuccess)
        {
            std::cout << "Error getting workspace size, status = " << rc << std::endl;
            return rc;
        }
    }

    if(is_fwd || is_wrw)
        in = tensor<Tgpu>(miopen::deref(inputTensor));
    if(is_fwd || is_bwd)
        wei = tensor<Tgpu>(miopen::deref(weightTensor));
    if(is_fwd)
        out = tensor<Tgpu>(miopen::deref(outputTensor));
    if(is_bwd || is_wrw)
        dout = tensor<Tgpu>(miopen::deref(outputTensor));

    if(is_bwd)
        din = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    if(is_wrw)
        dwei = std::vector<Tgpu>(wei_sz, static_cast<Tgpu>(0));
    if(is_int8)
        out_int8 = std::vector<int32_t>(out_sz, 0);
    if(is_transform)
    {
        in_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(inputTensor_vect4), sizeof(Tgpu)));
        wei_vect4_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, GetTensorSize(weightTensor_vect4), sizeof(Tgpu)));
    }

    outhost   = tensor<Tref>(miopen::deref(outputTensor));
    din_host  = tensor<Tref>(miopen::deref(inputTensor));
    dwei_host = tensor<Tref>(miopen::deref(weightTensor));

    std::string inFileName   = inflags.GetValueStr("in_data");
    std::string weiFileName  = inflags.GetValueStr("weights");
    std::string biasFileName = inflags.GetValueStr("in_bias");
    std::string doutFileName = inflags.GetValueStr("dout_data");

    /* Unless seed is persistent between runs validation using cache stored in file is impossible.
     */
    srand(0);

    bool dataRead = false;
    if(is_fwd || is_wrw)
        if(!inFileName.empty())
            dataRead = readBufferFromFile<Tgpu>(in.data.data(), in_sz, inFileName.c_str());

    bool weiRead = false;
    if(is_fwd || is_bwd)
        if(!weiFileName.empty())
            weiRead = readBufferFromFile<Tgpu>(wei.data.data(), wei_sz, weiFileName.c_str());

    if(is_int8)
    {
        float Data_scale = 127.0;

        if(!dataRead)
        {
            for(int i = 0; i < in_sz; i++)
            {
                if(is_fwd || is_wrw)
                    in.data[i] =
                        static_cast<Tgpu>(Data_scale * RAN_GEN<float>(static_cast<float>(0.0),
                                                                      static_cast<float>(1.0)));
                else /// \anchor move_rand
                    /// Move rand() forward, even if buffer is unused. This provides the same
                    /// initialization of input buffers regardless of which kinds of
                    /// convolutions are currently selectedfor testing (see the "-F" option).
                    /// Verification cache would be broken otherwise.
                    GET_RAND();
            }
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = GetTensorSize(biasTensor);
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(float)));
            b_int8      = std::vector<float>(b_sz, static_cast<float>(0));
            for(int i = 0; i < b_sz; i++)
            {
                b_int8[i] = static_cast<float>(i % 8) +
                            RAN_GEN<float>(static_cast<float>(0.0), static_cast<float>(1.0));
            }

            if(!biasFileName.empty())
            {
                readBufferFromFile<float>(b_int8.data(), b_sz, biasFileName.c_str());
            }

            b_dev->ToGPU(q, b_int8.data());
        }

        if(!weiRead)
        {
            for(int i = 0; i < wei_sz; i++)
                if(is_fwd || is_bwd)
                    wei.data[i] =
                        static_cast<Tgpu>(Data_scale * 2 * detail::RanGenWeights<float>());
                else /// \ref move_rand
                    GET_RAND();
        }
    }
    else
    {
        Tgpu Data_scale = static_cast<Tgpu>(0.01);

        bool doutRead = false;
        if(is_bwd || is_wrw)
            if(!doutFileName.empty())
                doutRead = readBufferFromFile<Tgpu>(dout.data.data(), out_sz, doutFileName.c_str());

        if(!dataRead)
        {
            for(int i = 0; i < in_sz; i++)
            {
                if(is_fwd || is_wrw)
                    in.data[i] =
                        Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
                else /// \ref move_rand
                    GET_RAND();
            }
        }

        if(!doutRead)
        {
            for(int i = 0; i < out_sz; i++)
                if(is_bwd || is_wrw)
                    dout.data[i] =
                        Data_scale * RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
                else /// \ref move_rand
                    GET_RAND();
        }

        if(inflags.GetValueInt("bias") != 0)
        {
            size_t b_sz = GetTensorSize(biasTensor);
            b_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            db_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, b_sz, sizeof(Tgpu)));
            b           = tensor<Tgpu>(miopen::deref(biasTensor));
            db          = std::vector<Tgpu>(b_sz, static_cast<Tgpu>(0));
            db_host     = tensor<Tref>(miopen::deref(biasTensor));
            for(int i = 0; i < b_sz; i++)
            {
                b.data[i] = static_cast<Tgpu>(i % 8) +
                            RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
                db[i] = static_cast<Tgpu>(i % 8) +
                        RAN_GEN<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
            }

            if(!biasFileName.empty())
            {
                readBufferFromFile<Tgpu>(b.data.data(), b_sz, biasFileName.c_str());
            }

            b_dev->ToGPU(q, b.data.data());
            db_dev->ToGPU(q, db.data());
        }

        if(!weiRead)
        {
            for(int i = 0; i < wei_sz; i++)
                if(is_fwd || is_bwd)
                    wei.data[i] = Data_scale * detail::RanGenWeights<Tgpu>();
                else /// \ref move_rand
                    GET_RAND();
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        if(is_fwd || is_wrw)
            dumpBufferToFile<Tgpu>("dump_in.bin", in.data.data(), in_sz);
        if(is_fwd || is_bwd)
            dumpBufferToFile<Tgpu>("dump_wei.bin", wei.data.data(), wei_sz);
        if(inflags.GetValueInt("bias") != 0)
            dumpBufferToFile<Tgpu>("dump_bias.bin", b.data.data(), b.data.size());

        if(is_bwd || is_wrw)
            dumpBufferToFile<Tgpu>("dump_dout.bin", dout.data.data(), out_sz);
    }

    status_t status = STATUS_SUCCESS;

    if(is_fwd || is_wrw)
    {
        in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        status |= in_dev->ToGPU(q, in.data.data());
    }
    if(is_bwd)
    {
        din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        status |= din_dev->ToGPU(q, din.data());
    }
    if(is_fwd || is_bwd)
    {
        wei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
        status |= wei_dev->ToGPU(q, wei.data.data());
    }
    if(is_wrw)
    {
        dwei_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, wei_sz, sizeof(Tgpu)));
        status |= dwei_dev->ToGPU(q, dwei.data());
    }
    if(is_bwd || is_wrw)
    {
        dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        status |= dout_dev->ToGPU(q, dout.data.data());
    }
    if(is_fwd)
    {
        out_dev = std::unique_ptr<GPUMem>(
            new GPUMem(ctx, out_sz, is_int8 ? sizeof(float) : sizeof(Tgpu)));
        status |=
            (is_int8 ? out_dev->ToGPU(q, out_int8.data()) : out_dev->ToGPU(q, out.data.data()));
    }

    if(status != STATUS_SUCCESS)
    {
        std::cout << "Error copying data to GPU, status = " << status << std::endl;
        return miopenStatusNotInitialized;
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
bool ConvDriver<Tgpu, Tref>::UseGPUReference()
{
    if(miopen::IsEnabled(MIOPEN_DRIVER_USE_GPU_REFERENCE{}))
    {
        if((miopen_type<Tref>{} == miopenFloat &&
            (miopen_type<Tgpu>{} == miopenFloat || miopen_type<Tgpu>{} == miopenHalf ||
             miopen_type<Tgpu>{} == miopenBFloat16)) ||
           (miopen_type<Tref>{} == miopenInt32 && miopen_type<Tgpu>{} == miopenInt8))
            return true;
        else
            return false;
    }
    else
        return false;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindForward(int& ret_algo_count,
                                        int request_algo_count,
                                        std::vector<miopenConvAlgoPerf_t>& perf_results,
                                        context_t ctx)
{
    bool is_transform = IsInputTensorTransform();
    fwd_auxiliary.resume(wall_enabled);
    ResizeWorkspaceDev(ctx, ws_sizeof_find_fwd);
    const auto rc = miopenFindConvolutionForwardAlgorithm(
        GetHandle(),
        (is_transform ? inputTensor_vect4 : inputTensor),
        (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem()),
        (is_transform ? weightTensor_vect4 : weightTensor),
        (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem()),
        convDesc,
        outputTensor,
        out_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        workspace_dev != nullptr ? workspace_dev->GetMem() : nullptr,
        ws_sizeof_find_fwd,
        (inflags.GetValueInt("search") == 1) ? true : false);
    fwd_auxiliary.pause(wall_enabled);
    return rc;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::PrintForwardTime(const float kernel_total_time,
                                              const float kernel_first_time) const
{
    float kernel_average_time = num_iterations > 1
                                    ? (kernel_total_time - kernel_first_time) / (num_iterations - 1)
                                    : kernel_first_time;
    printf("GPU Kernel Time Forward Conv. Elapsed: %f ms (average)\n", kernel_average_time);

    const auto num_dim = miopen::deref(inputTensor).GetSize() - 2;
    if(num_dim != 2 && num_dim != 3)
    {
        printf("stats: <not implemented> for conv%dd\n", num_dim);
        return;
    }

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(num_dim == 2)
    {
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_h, wei_w) =
            miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) =
            miopen::tien<4>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / group_count;
        size_t inputBytes =
            in_n * in_c * in_h * in_w * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        size_t weightBytes = wei_n * wei_c * wei_h * wei_w *
                             miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
        size_t readBytes = inputBytes + weightBytes;

        size_t outputBytes = 1.0 * out_n * out_c * out_h * out_w *
                             miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

        printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
               "GB/s, timeMs\n");
        printf("stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
               "fwd-conv",
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               out_h,
               out_w,
               wei_h,
               wei_w,
               out_c,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
    else
    { // 3d
        int in_n, in_c, in_d, in_h, in_w;
        std::tie(in_n, in_c, in_d, in_h, in_w) =
            miopen::tien<5>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_d, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_d, wei_h, wei_w) =
            miopen::tien<5>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_d, out_h, out_w;
        std::tie(out_n, out_c, out_d, out_h, out_w) =
            miopen::tien<5>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt = 2L * in_n * in_c * in_d * wei_h * wei_w * wei_d * out_c * out_d * out_h *
                         out_w / group_count;
        size_t inputBytes = in_n * in_c * in_d * in_h * in_w *
                            miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        size_t weightBytes = wei_n * wei_c * wei_d * wei_h * wei_w *
                             miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
        size_t readBytes = inputBytes + weightBytes;

        size_t outputBytes = 1.0 * out_n * out_c * out_d * out_h * out_w *
                             miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

        printf("stats: name  , n, c, do, ho, wo, z, y, x, k, flopCnt, bytesRead, bytesWritten, "
               "GFLOPs, "
               "GB/s, timeMs\n");
        printf("stats: %s%dx%dx%du%d, %u, %u, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, "
               "%.0f, %.0f, %f\n",
               "fwd-conv",
               wei_d,
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               out_d,
               out_h,
               out_w,
               wei_d,
               wei_h,
               wei_w,
               out_c,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
}

/// Always warm-ups Find API. Why: this is definitely required for Find mode.
/// For Immediate mode, this guarantees that we won't hit fallback.
/// Immediate mode API is warmed-up only when driver is used in Immediate mode.
template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunWarmupFindForwardGPU()
{
    if(!warmup_enabled)
        return 0;

    AutoMiopenWarmupMode warmupMode;

    int find_count;
    miopenConvAlgoPerf_t find_result;
    warmup_wall_total.resume(wall_enabled);
    auto rc = miopenFindConvolutionForwardAlgorithm(GetHandle(),
                                                    warmupInputTensor,
                                                    warmup_in_dev->GetMem(),
                                                    warmupWeightTensor,
                                                    warmup_wei_dev->GetMem(),
                                                    warmupConvDesc,
                                                    warmupOutputTensor,
                                                    warmup_out_dev->GetMem(),
                                                    1,
                                                    &find_count,
                                                    &find_result,
                                                    nullptr,
                                                    0,
                                                    false);
    warmup_wall_total.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return 10;
    if(find_count == 0)
        return 20;

    miopenConvSolution_t solution;
    if(immediate_solution)
    {
        std::size_t immed_count;
        warmup_wall_total.resume(wall_enabled);
        rc = miopenConvolutionForwardGetSolutionCount(handle,
                                                      warmupWeightTensor,
                                                      warmupInputTensor,
                                                      warmupConvDesc,
                                                      warmupOutputTensor,
                                                      &immed_count);
        warmup_wall_total.pause(wall_enabled);
        if(rc != miopenStatusSuccess)
            return 30;
        if(immed_count < 1)
            return 40;

        warmup_wall_total.resume(wall_enabled);
        rc = miopenConvolutionForwardGetSolution(handle,
                                                 warmupWeightTensor,
                                                 warmupInputTensor,
                                                 warmupConvDesc,
                                                 warmupOutputTensor,
                                                 1,
                                                 &immed_count,
                                                 &solution);
        warmup_wall_total.pause(wall_enabled);
        if(rc != miopenStatusSuccess)
            return 50;
        if(immed_count < 1)
            return 60;
        if(solution.workspace_size != 0)
            return 70;

        warmup_wall_total.resume(wall_enabled);
        rc = miopenConvolutionForwardImmediate(handle,
                                               warmupWeightTensor,
                                               warmup_wei_dev->GetMem(),
                                               warmupInputTensor,
                                               warmup_in_dev->GetMem(),
                                               warmupConvDesc,
                                               warmupOutputTensor,
                                               warmup_out_dev->GetMem(),
                                               nullptr,
                                               0,
                                               solution.solution_id);
        warmup_wall_total.pause(wall_enabled);
        if(rc != miopenStatusSuccess)
            return 80;
    }

    warmup_wall_total.stop(wall_enabled);
    std::ostringstream ss;
    ss << "Warm-up: ";
    if(wall_enabled)
        ss << "Wall-clock Total Time: " << warmup_wall_total.gettime_ms() << " ms, ";
    ss << "Find Algorithm: " << find_result.fwd_algo;
    if(immediate_solution)
        ss << ", Immediate Algorithm: " << miopen::ConvolutionAlgoToString(solution.algorithm)
           << '[' << solution.solution_id << ']';
    ss << std::endl;
    std::cout << ss.str();
    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardGPU()
{
    if(!is_fwd)
        return 0;

    int rc;
    bool is_transform = IsInputTensorTransform();

    if(is_transform)
    {
        float aph = 1.0;
        float bta = 0.0;
        miopenTransformTensor(GetHandle(),
                              &aph,
                              inputTensor,
                              in_dev->GetMem(),
                              &bta,
                              inputTensor_vect4,
                              in_vect4_dev->GetMem());

        miopenTransformTensor(GetHandle(),
                              &aph,
                              weightTensor,
                              wei_dev->GetMem(),
                              &bta,
                              weightTensor_vect4,
                              wei_vect4_dev->GetMem());
    }

    if(immediate_solution)
        rc = RunForwardGpuImmed(is_transform);
    else
        rc = RunForwardGpuFind(is_transform);

    is_fwd_run_failed = (rc != 0);

    if(rc != miopenStatusSuccess)
        return rc;

    if(inflags.GetValueInt("bias") != 0)
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);

        miopenConvolutionForwardBias(GetHandle(),
                                     &alpha,
                                     biasTensor,
                                     b_dev->GetMem(),
                                     &beta,
                                     outputTensor,
                                     out_dev->GetMem());

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);

            printf("GPU Kernel Time Forward Conv. Bias Elapsed: %f ms\n", time);
        }
    }

    bool is_int8 = data_type == miopenInt8 || data_type == miopenInt8x4;
    if(is_int8)
        out_dev->FromGPU(GetStream(), out_int8.data());
    else
        out_dev->FromGPU(GetStream(), out.data.data());

    if(inflags.GetValueInt("dump_output"))
    {
        if(is_int8)
            dumpBufferToFile<int32_t>("dump_fwd_out_gpu.bin", out_int8.data(), out_int8.size());
        else
            dumpBufferToFile<Tgpu>("dump_fwd_out_gpu.bin", out.data.data(), out.data.size());
    }

    return rc;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::GetSolutionAfterFind(
    const miopenConvAlgoPerf_t& found,
    const ConvDriver<Tgpu, Tref>::Direction& direction,
    const miopenTensorDescriptor_t& in_tensor,
    const miopenTensorDescriptor_t& wei_tensor,
    const miopenTensorDescriptor_t& out_tensor,
    miopenConvSolution_t& solution)
{
    AutoMiopenWarmupMode warmupMode; // Shut logging.
    miopenConvAlgorithm_t found_algo;
    switch(direction)
    {
    case Direction::Fwd: found_algo = static_cast<miopenConvAlgorithm_t>(found.fwd_algo); break;
    case Direction::Bwd:
        found_algo = static_cast<miopenConvAlgorithm_t>(found.bwd_data_algo);
        break;
    case Direction::WrW:
        found_algo = static_cast<miopenConvAlgorithm_t>(found.bwd_weights_algo);
        break;
    case Direction::BwdBias: // nop
        MIOPEN_THROW("BwdBias is not supported");
    }
    std::size_t immed_count = 0;
    miopenStatus_t rc       = miopenStatusUnknownError;
    switch(direction)
    {
    case Direction::Fwd:
        rc = miopenConvolutionForwardGetSolution(
            handle, wei_tensor, in_tensor, convDesc, out_tensor, 1, &immed_count, &solution);
        break;
    case Direction::Bwd:
        rc = miopenConvolutionBackwardDataGetSolution(
            handle, out_tensor, wei_tensor, convDesc, in_tensor, 1, &immed_count, &solution);
        break;
    case Direction::WrW:
        rc = miopenConvolutionBackwardWeightsGetSolution(
            handle, out_tensor, in_tensor, convDesc, wei_tensor, 1, &immed_count, &solution);
        break;
    case Direction::BwdBias: // nop
        break;
    }
    if(rc != miopenStatusSuccess           // (formatting)
       || immed_count < 1                  // It should not be so if Find succeeded.
       || found_algo != solution.algorithm // These must match.
       || solution.time < 0)               // Fallback mode (no entry in find-db -- disabled?)
    {
        // Ignore errors, just skip printing the solver information.
        solution.solution_id = 0;
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardGpuFind(const bool is_transform)
{
    int ret_algo_count;
    int request_algo_count = 2;
    // The library returns `request_algo_count` algorithms to the caller. However this does
    // not affect auto-tuning and find-db updates. Internally, the library searches for
    // *all* available algorithms during Find(), -- regardless of how many algorithms
    // requested, -- so perf-db and find-db are fully updated.
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto rc = FindForward(ret_algo_count, request_algo_count, perf_results, ctx);
    if(rc != miopenStatusSuccess)
        return rc;

    if(ret_algo_count == 0)
        throw std::runtime_error("Find Forward Conv. ret_algo_count == 0");

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    const auto algo    = perf_results[0].fwd_algo; // use the fastest algo
    const auto ws_size = perf_results[0].memory;
    is_fwd_igemm       = (algo == miopenConvolutionFwdAlgoImplicitGEMM);

    ResizeWorkspaceDev(ctx, ws_size);
    wall.start(wall_enabled);

    auto in_tens  = (is_transform ? inputTensor_vect4 : inputTensor);
    auto in_buff  = (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem());
    auto wei_tens = (is_transform ? weightTensor_vect4 : weightTensor);
    auto wei_buff = (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem());

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionForward(GetHandle(),
                                      &alpha,
                                      in_tens,
                                      in_buff,
                                      wei_tens,
                                      wei_buff,
                                      convDesc,
                                      algo,
                                      &beta,
                                      outputTensor,
                                      out_dev->GetMem(),
                                      workspace_dev != nullptr ? workspace_dev->GetMem() : nullptr,
                                      ws_size);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
                kernel_first_time = time;
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        fwd_auxiliary.stop();
        fwd_auxiliary_gwss.stop();
        std::cout << "Wall-clock Time Forward Conv. Elapsed: "
                  << (wall.gettime_ms() / num_iterations) << " ms"
                  << ", Auxiliary API calls: " << fwd_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << fwd_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        miopenConvSolution_t solution;
        GetSolutionAfterFind(
            perf_results[0], Direction::Fwd, in_tens, wei_tens, outputTensor, solution);
        std::cout << "MIOpen Forward Conv. " << AlgorithmSolutionToString(solution) << std::endl;
        PrintForwardTime(kernel_total_time, kernel_first_time);
    }

    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardGpuImmed(const bool is_transform)
{
    std::size_t count;
    fwd_auxiliary.resume(wall_enabled);
    auto rc =
        miopenConvolutionForwardGetSolutionCount(handle,
                                                 (is_transform ? weightTensor_vect4 : weightTensor),
                                                 (is_transform ? inputTensor_vect4 : inputTensor),
                                                 convDesc,
                                                 outputTensor,
                                                 &count);
    fwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;
    if(count < 1)
        return miopenStatusNotImplemented;

    auto solutions = std::vector<miopenConvSolution_t>(count);
    fwd_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionForwardGetSolution(handle,
                                             (is_transform ? weightTensor_vect4 : weightTensor),
                                             (is_transform ? inputTensor_vect4 : inputTensor),
                                             convDesc,
                                             outputTensor,
                                             count,
                                             &count,
                                             solutions.data());
    fwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;

    std::cout << "Forward Conv solutions available: " << count << std::endl;
    if(count < 1)
        return miopenStatusNotImplemented;

    solutions.resize(count);
    const miopenConvSolution_t* selected = nullptr;

    for(const auto& s : solutions)
        PrintImmedSolutionInfo(s);

    if(*immediate_solution == 0)
        selected = &solutions.front();
    else
        for(const auto& s : solutions)
            if(*immediate_solution == s.solution_id)
            {
                selected = &s;
                break;
            }

    miopenConvSolution_t voluntary = {
        -1.0, 0, *immediate_solution, static_cast<miopenConvAlgorithm_t>(-1)};
    if(selected == nullptr)
    {
        std::cout << "Warning: Solution id (" << *immediate_solution
                  << ") is not reported by the library. Trying it anyway..." << std::endl;
        selected = &voluntary;
    }

    std::size_t ws_size;

    fwd_auxiliary.resume(wall_enabled);
    fwd_auxiliary_gwss.resume(wall_enabled);
    rc = miopenConvolutionForwardGetSolutionWorkspaceSize(
        handle,
        (is_transform ? weightTensor_vect4 : weightTensor),
        (is_transform ? inputTensor_vect4 : inputTensor),
        convDesc,
        outputTensor,
        selected->solution_id,
        &ws_size);
    fwd_auxiliary_gwss.pause(wall_enabled);
    fwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto ws = std::unique_ptr<GPUMem>{ws_size > 0 ? new GPUMem{ctx, ws_size, 1} : nullptr};

    fwd_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionForwardCompileSolution(handle,
                                                 (is_transform ? weightTensor_vect4 : weightTensor),
                                                 (is_transform ? inputTensor_vect4 : inputTensor),
                                                 convDesc,
                                                 outputTensor,
                                                 selected->solution_id);
    fwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    wall.start(wall_enabled);

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionForwardImmediate(
            handle,
            (is_transform ? weightTensor_vect4 : weightTensor),
            (is_transform ? wei_vect4_dev->GetMem() : wei_dev->GetMem()),
            (is_transform ? inputTensor_vect4 : inputTensor),
            (is_transform ? in_vect4_dev->GetMem() : in_dev->GetMem()),
            convDesc,
            outputTensor,
            out_dev->GetMem(),
            ws ? ws->GetMem() : nullptr,
            ws_size,
            selected->solution_id);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
            {
                kernel_first_time = time;
                if(wall_enabled && num_iterations > 1)
                    wall.start(); // The 1st is warm-up. Disregard it in wall time.
            }
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        fwd_auxiliary.stop();
        fwd_auxiliary_gwss.stop();
        const auto wall_iterations = (num_iterations > 1 ? num_iterations - 1 : 1);
        std::cout << "Wall-clock Time Forward Conv. Elapsed: "
                  << (wall.gettime_ms() / wall_iterations) << " ms"
                  << ", Auxiliary API calls: " << fwd_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << fwd_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        std::cout << "MIOpen Forward Conv. " << AlgorithmSolutionToString(*selected) << std::endl;
        PrintForwardTime(kernel_total_time, kernel_first_time);
    }

    is_fwd_igemm = (selected->algorithm == miopenConvolutionAlgoImplicitGEMM);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                      outhost,
                                      wei,
                                      in,
                                      miopen::deref(convDesc).GetConvPads(),
                                      miopen::deref(convDesc).GetConvStrides(),
                                      miopen::deref(convDesc).GetConvDilations(),
                                      miopen::deref(convDesc).GetGroupCount());

        if(inflags.GetValueInt("bias") != 0)
        {
            cpu_bias_forward(outhost, b);
        }
    }
    else
    {
        cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                in,
                                wei,
                                outhost,
                                miopen::deref(convDesc).GetConvPads(),
                                miopen::deref(convDesc).GetConvStrides(),
                                miopen::deref(convDesc).GetConvDilations(),
                                miopen::deref(convDesc).GetGroupCount());

        if(inflags.GetValueInt("bias") != 0)
        {
            outhost.par_for_each([&](auto out_n_id, auto out_k_id, auto... out_spatial_id_pack) {
                outhost(out_n_id, out_k_id, out_spatial_id_pack...) =
                    double(outhost(out_n_id, out_k_id, out_spatial_id_pack...)) +
                    double(b.data[out_k_id]);
            });
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_fwd_out_cpu.bin", outhost.data.data(), outhost.data.size());
    }

    TrySaveVerificationCache(Direction::Fwd, outhost.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunForwardGPUReference()
{
    AutoConvDirectNaiveAlwaysEnable naive_conv_enable;

    if(inflags.GetValueInt("bias") != 0)
    {
        std::cout << "gpu reference convolution does not support bias yet" << std::endl;
        return -1;
    }
    auto ref_solution_id = miopen::deref(convDesc).mode == miopenTranspose
                               ? miopen::solver::Id("ConvDirectNaiveConvBwd").Value()
                               : miopen::solver::Id("ConvDirectNaiveConvFwd").Value();
    auto rc = miopenConvolutionForwardImmediate(handle,
                                                weightTensor,
                                                wei_dev->GetMem(),
                                                inputTensor,
                                                in_dev->GetMem(),
                                                convDesc,
                                                outputTensor,
                                                out_dev->GetMem(),
                                                nullptr,
                                                0,
                                                ref_solution_id);
    if(rc != miopenStatusSuccess)
    {
        std::cout << "reference kernel fail to run "
                  << miopen::solver::Id(ref_solution_id).ToString() << std::endl;
        return rc;
    }

    if(miopen_type<Tgpu>{} == miopen_type<Tref>{})
        out_dev->FromGPU(GetStream(), outhost.data.data());
    else
    {
        auto out_tmp = tensor<Tgpu>(miopen::deref(outputTensor));
        out_dev->FromGPU(GetStream(), out_tmp.data.data());
        for(int i = 0; i < out_tmp.data.size(); i++)
        {
            outhost.data[i] = static_cast<Tref>(out_tmp.data[i]);
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>(
            "dump_fwd_out_gpu_ref.bin", outhost.data.data(), outhost.data.size());
    }

    // TrySaveVerificationCache(Direction::Fwd, outhost.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindBackwardData(int& ret_algo_count,
                                             int request_algo_count,
                                             std::vector<miopenConvAlgoPerf_t>& perf_results,
                                             context_t ctx)
{
    bwd_auxiliary.resume(wall_enabled);
    ResizeWorkspaceDev(ctx, ws_sizeof_find_bwd);
    const auto rc = miopenFindConvolutionBackwardDataAlgorithm(
        GetHandle(),
        outputTensor,
        dout_dev->GetMem(),
        weightTensor,
        wei_dev->GetMem(),
        convDesc,
        inputTensor,
        din_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        workspace_dev != nullptr ? workspace_dev->GetMem() : nullptr,
        ws_sizeof_find_bwd,
        (inflags.GetValueInt("search") == 1) ? true : false);
    bwd_auxiliary.pause(wall_enabled);
    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::FindBackwardWeights(int& ret_algo_count,
                                                int request_algo_count,
                                                std::vector<miopenConvAlgoPerf_t>& perf_results,
                                                context_t ctx)
{
    wrw_auxiliary.resume(wall_enabled);
    ResizeWorkspaceDev(ctx, ws_sizeof_find_wrw);
    const auto rc = miopenFindConvolutionBackwardWeightsAlgorithm(
        GetHandle(),
        outputTensor,
        dout_dev->GetMem(),
        inputTensor,
        in_dev->GetMem(),
        convDesc,
        weightTensor,
        dwei_dev->GetMem(),
        request_algo_count,
        &ret_algo_count,
        perf_results.data(),
        workspace_dev != nullptr ? workspace_dev->GetMem() : nullptr,
        ws_sizeof_find_wrw,
        (inflags.GetValueInt("search") == 1) ? true : false);
    wrw_auxiliary.pause(wall_enabled);
    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardGPU()
{
    if(data_type == miopenInt8 || data_type == miopenInt8x4)
    {
        std::cout << "Int8 Backward Convolution is not supported" << std::endl;
        return 0;
    }

    if(!(is_bwd || is_wrw))
        return 0;

    int ret = 0;

    if(is_bwd)
    {
        auto rc = immediate_solution ? RunBackwardDataGpuImmed() : RunBackwardDataGpuFind();
        is_bwd_run_failed = (rc != 0);
        ret |= rc;
    }

    if(is_wrw)
    {
        auto rc           = immediate_solution ? RunBackwardWrwGpuImmed() : RunBackwardWrwGpuFind();
        is_wrw_run_failed = (rc != 0);
        ret |= (rc << 16); // Differentiate WrW and Bwd error codes.
    }

    if(inflags.GetValueInt("dump_output"))
    {
        if(is_bwd)
            dumpBufferToFile<Tgpu>("dump_bwd_din_gpu.bin", din.data(), din.size());
        if(is_wrw)
            dumpBufferToFile<Tgpu>("dump_bwd_dwei_gpu.bin", dwei.data(), dwei.size());
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);

        ret |= miopenConvolutionBackwardBias(GetHandle(),
                                             &alpha,
                                             outputTensor,
                                             dout_dev->GetMem(),
                                             &beta,
                                             biasTensor,
                                             db_dev->GetMem());

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            printf("GPU Kernel Time Backward Bias Conv. Elapsed: %f ms\n", time);
        }

        db_dev->FromGPU(GetStream(), db.data());
        if(inflags.GetValueInt("dump_output"))
        {
            dumpBufferToFile<Tgpu>("dump_bwd_db_gpu.bin", db.data(), db.size());
        }
    }
    return ret;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardDataGpuFind()
{
    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results_data(request_algo_count);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto rc = FindBackwardData(ret_algo_count, request_algo_count, perf_results_data, ctx);
    if(rc != miopenStatusSuccess)
        return rc;

    if(ret_algo_count == 0)
        throw std::runtime_error("Find Backward Data Conv. ret_algo_count == 0");

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;
    float alpha = static_cast<float>(1), beta = static_cast<float>(0);

    const auto algo    = perf_results_data[0].bwd_data_algo;
    const auto ws_size = perf_results_data[0].memory;
    is_bwd_igemm       = (algo == miopenConvolutionBwdDataAlgoImplicitGEMM);

    ResizeWorkspaceDev(ctx, ws_size);
    wall.start(wall_enabled);

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionBackwardData(GetHandle(),
                                           &alpha,
                                           outputTensor,
                                           dout_dev->GetMem(),
                                           weightTensor,
                                           wei_dev->GetMem(),
                                           convDesc,
                                           algo,
                                           &beta,
                                           inputTensor,
                                           din_dev->GetMem(),
                                           workspace_dev != nullptr ? workspace_dev->GetMem()
                                                                    : nullptr,
                                           ws_size);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
                kernel_first_time = time;
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        bwd_auxiliary.stop();
        bwd_auxiliary_gwss.stop();
        std::cout << "Wall-clock Time Backward Data Conv. Elapsed: "
                  << (wall.gettime_ms() / num_iterations) << " ms"
                  << ", Auxiliary API calls: " << bwd_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << bwd_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        miopenConvSolution_t solution;
        GetSolutionAfterFind(perf_results_data[0],
                             Direction::Bwd,
                             inputTensor,
                             weightTensor,
                             outputTensor,
                             solution);
        std::cout << "MIOpen Backward Data Conv. " << AlgorithmSolutionToString(solution)
                  << std::endl;
        PrintBackwardDataTime(kernel_total_time, kernel_first_time);
    }

    din_dev->FromGPU(GetStream(), din.data());
    return rc;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::PrintBackwardDataTime(float kernel_total_time, float kernel_first_time)
{
    float kernel_average_time = num_iterations > 1
                                    ? (kernel_total_time - kernel_first_time) / (num_iterations - 1)
                                    : kernel_first_time;

    printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms (average)\n", kernel_average_time);

    const auto num_dim = miopen::deref(inputTensor).GetSize() - 2;
    if(num_dim != 2 && num_dim != 3)
    {
        printf("stats: <not implemented> for conv%dd\n", num_dim);
        return;
    }

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(num_dim == 2)
    {
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_h, wei_w) =
            miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) =
            miopen::tien<4>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt     = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / group_count;
        size_t weightBytes = wei_n * wei_c * wei_h * wei_w *
                             miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
        size_t inputBytes =
            in_n * in_c * out_c * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        size_t readBytes = inputBytes + weightBytes;

        size_t outputBytes = 1.0 * out_n * out_c * out_h * out_w *
                             miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

        printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
               "GB/s, timeMs\n");
        printf("stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
               "bwdd-conv",
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               wei_h,
               wei_w,
               out_c,
               out_h,
               out_w,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
    else
    { // 3d
        int in_n, in_c, in_d, in_h, in_w;
        std::tie(in_n, in_c, in_d, in_h, in_w) =
            miopen::tien<5>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_d, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_d, wei_h, wei_w) =
            miopen::tien<5>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_d, out_h, out_w;
        std::tie(out_n, out_c, out_d, out_h, out_w) =
            miopen::tien<5>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt =
            2L * in_n * in_c * wei_d * wei_h * wei_w * out_c * out_d * out_h * out_w / group_count;
        size_t weightBytes = wei_n * wei_c * wei_d * wei_h * wei_w *
                             miopen::GetTypeSize(miopen::deref(weightTensor).GetType());
        size_t inputBytes =
            in_n * in_c * out_c * miopen::GetTypeSize(miopen::deref(inputTensor).GetType());
        size_t readBytes = inputBytes + weightBytes;

        size_t outputBytes = 1.0 * out_n * out_c * out_d * out_h * out_w *
                             miopen::GetTypeSize(miopen::deref(outputTensor).GetType());

        printf(
            "stats: name, n, c, do, ho, wo, z, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
            "GB/s, timeMs\n");
        printf("stats: %s%dx%dx%du%d, %u, %u, %u, %u, %u, %u, %u, %u, %u  %zu, %zu, %zu, %.0f, "
               "%.0f, %f\n",
               "bwdd-conv",
               wei_d,
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               wei_d,
               wei_h,
               wei_w,
               out_c,
               out_d,
               out_h,
               out_w,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardWrwGpuFind()
{
    int ret_algo_count;
    int request_algo_count = 2;

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    float alpha = static_cast<float>(1), beta = static_cast<float>(0);
    std::vector<miopenConvAlgoPerf_t> perf_results_weights(request_algo_count);

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto rc = FindBackwardWeights(ret_algo_count, request_algo_count, perf_results_weights, ctx);
    if(rc != miopenStatusSuccess)
        return rc;

    if(ret_algo_count == 0)
        throw std::runtime_error("Find Backward Weights Conv. ret_algo_count == 0");

    kernel_total_time = 0.0;
    kernel_first_time = 0.0;

    const auto algo    = perf_results_weights[0].bwd_weights_algo;
    const auto ws_size = perf_results_weights[0].memory;
    is_wrw_winograd    = (algo == miopenConvolutionBwdWeightsAlgoWinograd);
    is_wrw_igemm       = (algo == miopenConvolutionBwdWeightsAlgoImplicitGEMM);

    ResizeWorkspaceDev(ctx, ws_size);
    wall.start(wall_enabled);

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionBackwardWeights(GetHandle(),
                                              &alpha,
                                              outputTensor,
                                              dout_dev->GetMem(),
                                              inputTensor,
                                              in_dev->GetMem(),
                                              convDesc,
                                              algo,
                                              &beta,
                                              weightTensor,
                                              dwei_dev->GetMem(),
                                              workspace_dev != nullptr ? workspace_dev->GetMem()
                                                                       : nullptr,
                                              ws_size);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
                kernel_first_time = time;
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        wrw_auxiliary.stop();
        wrw_auxiliary_gwss.stop();
        std::cout << "Wall-clock Time Backward Weights Conv. Elapsed: "
                  << (wall.gettime_ms() / num_iterations) << " ms"
                  << ", Auxiliary API calls: " << wrw_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << wrw_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        miopenConvSolution_t solution;
        GetSolutionAfterFind(perf_results_weights[0],
                             Direction::WrW,
                             inputTensor,
                             weightTensor,
                             outputTensor,
                             solution);
        std::cout << "MIOpen Backward Weights Conv. " << AlgorithmSolutionToString(solution)
                  << std::endl;
        PrintBackwardWrwTime(kernel_total_time, kernel_first_time);
    }

    dwei_dev->FromGPU(GetStream(), dwei.data());
    return rc;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::PrintBackwardWrwTime(float kernel_total_time, float kernel_first_time)
{
    float time = 0.0;
    miopenGetKernelTime(GetHandle(), &time);

    float kernel_average_time = num_iterations > 1
                                    ? (kernel_total_time - kernel_first_time) / (num_iterations - 1)
                                    : kernel_first_time;

    printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms (average)\n",
           kernel_average_time);

    const auto num_dim = miopen::deref(inputTensor).GetSize() - 2;
    if(num_dim != 2 && num_dim != 3)
    {
        printf("stats: <not implemented> for conv%dd\n", num_dim);
        return;
    }

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(num_dim == 2)
    {
        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_h, wei_w) =
            miopen::tien<4>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) =
            miopen::tien<4>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt     = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / group_count;
        size_t readBytes   = 0;
        size_t outputBytes = 0;

        printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
               "GB/s, timeMs\n");
        printf("stats: %s%dx%du%d, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, %.0f, %f\n",
               "bwdw-conv",
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               out_h,
               out_w,
               wei_h,
               wei_w,
               out_c,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
    else
    { // 3d
        int in_n, in_c, in_d, in_h, in_w;
        std::tie(in_n, in_c, in_d, in_h, in_w) =
            miopen::tien<5>(miopen::deref(inputTensor).GetLengths());
        int wei_c, wei_n, wei_d, wei_h, wei_w;
        std::tie(wei_c, wei_n, wei_d, wei_h, wei_w) =
            miopen::tien<5>(miopen::deref(weightTensor).GetLengths());
        int out_n, out_c, out_d, out_h, out_w;
        std::tie(out_n, out_c, out_d, out_h, out_w) =
            miopen::tien<5>(miopen::deref(outputTensor).GetLengths());

        size_t flopCnt =
            2L * in_n * in_c * wei_d * wei_h * wei_w * out_c * out_d * out_h * out_w / group_count;
        size_t readBytes   = 0;
        size_t outputBytes = 0;

        printf(
            "stats: name, n, c, do, ho, wo, z, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
            "GB/s, timeMs\n");
        printf("stats: %s%dx%dx%du%d, %u, %u, %u, %u, %u, %u, %u, %u, %u,  %zu, %zu, %zu, %.0f, "
               "%.0f, %f\n ",
               "bwdw-conv",
               wei_d,
               wei_h,
               wei_w,
               miopen::deref(convDesc).GetConvStrides()[0],
               in_n,
               in_c,
               out_d,
               out_h,
               out_w,
               wei_d,
               wei_h,
               wei_w,
               out_c,
               flopCnt,
               readBytes,
               outputBytes,
               flopCnt / kernel_average_time / 1e6,
               (readBytes + outputBytes) / kernel_average_time / 1e6,
               kernel_average_time);
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardDataGpuImmed()
{
    std::size_t count;

    bwd_auxiliary.resume(wall_enabled);
    auto rc = miopenConvolutionBackwardDataGetSolutionCount(
        handle, outputTensor, weightTensor, convDesc, inputTensor, &count);
    bwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;
    if(count < 1)
        return miopenStatusNotImplemented;

    auto solutions = std::vector<miopenConvSolution_t>(count);
    bwd_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionBackwardDataGetSolution(
        handle, outputTensor, weightTensor, convDesc, inputTensor, count, &count, solutions.data());
    bwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;
    std::cout << "Backward Data Conv solutions available: " << count << std::endl;
    if(count < 1)
        return miopenStatusNotImplemented;

    solutions.resize(count);
    const miopenConvSolution_t* selected = nullptr;

    for(const auto& s : solutions)
        PrintImmedSolutionInfo(s);

    if(*immediate_solution == 0)
        selected = &solutions.front();
    else
        for(const auto& s : solutions)
            if(*immediate_solution == s.solution_id)
            {
                selected = &s;
                break;
            }

    miopenConvSolution_t voluntary = {
        -1.0, 0, *immediate_solution, static_cast<miopenConvAlgorithm_t>(-1)};
    if(selected == nullptr)
    {
        std::cout << "Warning: Solution id (" << *immediate_solution
                  << ") is not reported by the library. Trying it anyway..." << std::endl;
        selected = &voluntary;
    }

    std::size_t ws_size;

    bwd_auxiliary.resume(wall_enabled);
    bwd_auxiliary_gwss.resume(wall_enabled);
    rc = miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
        handle, outputTensor, weightTensor, convDesc, inputTensor, selected->solution_id, &ws_size);
    bwd_auxiliary_gwss.pause(wall_enabled);
    bwd_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto ws = std::unique_ptr<GPUMem>{ws_size > 0 ? new GPUMem{ctx, ws_size, 1} : nullptr};

    bwd_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionBackwardDataCompileSolution(
        handle, outputTensor, weightTensor, convDesc, inputTensor, selected->solution_id);
    bwd_auxiliary.pause(wall_enabled);

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    wall.start(wall_enabled);

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionBackwardDataImmediate(handle,
                                                    outputTensor,
                                                    dout_dev->GetMem(),
                                                    weightTensor,
                                                    wei_dev->GetMem(),
                                                    convDesc,
                                                    inputTensor,
                                                    din_dev->GetMem(),
                                                    ws ? ws->GetMem() : nullptr,
                                                    ws_size,
                                                    selected->solution_id);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
            {
                kernel_first_time = time;
                if(wall_enabled && num_iterations > 1)
                    wall.start();
            }
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        bwd_auxiliary.stop();
        bwd_auxiliary_gwss.stop();
        const auto wall_iterations = (num_iterations > 1 ? num_iterations - 1 : 1);
        std::cout << "Wall-clock Time Backward Data Conv. Elapsed: "
                  << (wall.gettime_ms() / wall_iterations) << " ms"
                  << ", Auxiliary API calls: " << bwd_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << bwd_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        std::cout << "MIOpen Backward Data Conv. " << AlgorithmSolutionToString(*selected)
                  << std::endl;
        PrintBackwardDataTime(kernel_total_time, kernel_first_time);
    }

    is_bwd_igemm = (selected->algorithm == miopenConvolutionAlgoImplicitGEMM);
    din_dev->FromGPU(GetStream(), din.data());
    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardWrwGpuImmed()
{
    std::size_t count;
    wrw_auxiliary.resume(wall_enabled);
    auto rc = miopenConvolutionBackwardWeightsGetSolutionCount(
        handle, outputTensor, inputTensor, convDesc, weightTensor, &count);
    wrw_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;
    if(count < 1)
        return miopenStatusNotImplemented;

    auto solutions = std::vector<miopenConvSolution_t>(count);
    wrw_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionBackwardWeightsGetSolution(
        handle, outputTensor, inputTensor, convDesc, weightTensor, count, &count, solutions.data());
    wrw_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;
    std::cout << "Backward Weights Conv solutions available: " << count << std::endl;
    if(count < 1)
        return miopenStatusNotImplemented;

    solutions.resize(count);
    const miopenConvSolution_t* selected = nullptr;

    for(const auto& s : solutions)
        PrintImmedSolutionInfo(s);

    if(*immediate_solution == 0)
        selected = &solutions.front();
    else
        for(const auto& s : solutions)
            if(*immediate_solution == s.solution_id)
            {
                selected = &s;
                break;
            }

    miopenConvSolution_t voluntary = {
        -1.0, 0, *immediate_solution, static_cast<miopenConvAlgorithm_t>(-1)};
    if(selected == nullptr)
    {
        std::cout << "Warning: Solution id (" << *immediate_solution
                  << ") is not reported by the library. Trying it anyway..." << std::endl;
        selected = &voluntary;
    }

    std::size_t ws_size;

    wrw_auxiliary.resume(wall_enabled);
    wrw_auxiliary_gwss.resume(wall_enabled);
    rc = miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
        handle, outputTensor, inputTensor, convDesc, weightTensor, selected->solution_id, &ws_size);
    wrw_auxiliary_gwss.pause(wall_enabled);
    wrw_auxiliary.pause(wall_enabled);
    if(rc != miopenStatusSuccess)
        return rc;

#if MIOPEN_BACKEND_OPENCL
    cl_context ctx;

    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#elif MIOPEN_BACKEND_HIP
    uint32_t ctx = 0;
#endif

    auto ws = std::unique_ptr<GPUMem>{ws_size > 0 ? new GPUMem{ctx, ws_size, 1} : nullptr};

    wrw_auxiliary.resume(wall_enabled);
    rc = miopenConvolutionBackwardWeightsCompileSolution(
        handle, outputTensor, inputTensor, convDesc, weightTensor, selected->solution_id);
    wrw_auxiliary.pause(wall_enabled);

    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    wall.start(wall_enabled);

    for(int i = 0; i < num_iterations; i++)
    {
        rc = miopenConvolutionBackwardWeightsImmediate(handle,
                                                       outputTensor,
                                                       dout_dev->GetMem(),
                                                       inputTensor,
                                                       in_dev->GetMem(),
                                                       convDesc,
                                                       weightTensor,
                                                       dwei_dev->GetMem(),
                                                       ws ? ws->GetMem() : nullptr,
                                                       ws_size,
                                                       selected->solution_id);
        if(rc != miopenStatusSuccess)
            return rc;

        if(time_enabled)
        {
            float time = 0.0;
            miopenGetKernelTime(GetHandle(), &time);
            kernel_total_time += time;
            if(i == 0)
            {
                kernel_first_time = time;
                if(wall_enabled && num_iterations > 1)
                    wall.start();
            }
        }
    }

    if(wall_enabled)
    {
        wall.stop();
        wrw_auxiliary.stop();
        wrw_auxiliary_gwss.stop();
        const auto wall_iterations = (num_iterations > 1 ? num_iterations - 1 : 1);
        std::cout << "Wall-clock Time Backward Weights Conv. Elapsed: "
                  << (wall.gettime_ms() / wall_iterations) << " ms"
                  << ", Auxiliary API calls: " << wrw_auxiliary.gettime_ms() << " ms"
                  << " (GWSS: " << wrw_auxiliary_gwss.gettime_ms() << ')' << std::endl;
    }
    if(time_enabled)
    {
        std::cout << "MIOpen Backward Weights Conv. " << AlgorithmSolutionToString(*selected)
                  << std::endl;
        PrintBackwardWrwTime(kernel_total_time, kernel_first_time);
    }

    is_wrw_winograd = (selected->algorithm == miopenConvolutionAlgoWinograd);
    is_wrw_igemm    = (selected->algorithm == miopenConvolutionAlgoImplicitGEMM);
    dwei_dev->FromGPU(GetStream(), dwei.data());
    return rc;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardWeightsCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                        dout,
                                        dwei_host,
                                        in,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());
    }
    else
    {
        cpu_convolution_backward_weight(miopen::deref(convDesc).GetSpatialDimension(),
                                        in,
                                        dwei_host,
                                        dout,
                                        miopen::deref(convDesc).GetConvPads(),
                                        miopen::deref(convDesc).GetConvStrides(),
                                        miopen::deref(convDesc).GetConvDilations(),
                                        miopen::deref(convDesc).GetGroupCount());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>(
            "dump_bwd_dwei_cpu.bin", dwei_host.data.data(), dwei_host.data.size());
    }

    TrySaveVerificationCache(Direction::WrW, dwei_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardDataCPU()
{
    if(miopen::deref(convDesc).mode == miopenTranspose)
    {
        cpu_convolution_forward(miopen::deref(convDesc).GetSpatialDimension(),
                                dout,
                                wei,
                                din_host,
                                miopen::deref(convDesc).GetConvPads(),
                                miopen::deref(convDesc).GetConvStrides(),
                                miopen::deref(convDesc).GetConvDilations(),
                                miopen::deref(convDesc).GetGroupCount());
    }
    else
    {
        cpu_convolution_backward_data(miopen::deref(convDesc).GetSpatialDimension(),
                                      din_host,
                                      wei,
                                      dout,
                                      miopen::deref(convDesc).GetConvPads(),
                                      miopen::deref(convDesc).GetConvStrides(),
                                      miopen::deref(convDesc).GetConvDilations(),
                                      miopen::deref(convDesc).GetGroupCount());
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_bwd_din_cpu.bin", din_host.data.data(), din_host.data.size());
    }

    TrySaveVerificationCache(Direction::Bwd, din_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardBiasCPU()
{
    cpu_bias_backward_data(dout, db_host);

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>("dump_bwd_db_cpu.bin", db_host.data.data(), db_host.data.size());
    }

    TrySaveVerificationCache(Direction::BwdBias, db_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardWeightsGPUReference()
{
    AutoConvDirectNaiveAlwaysEnable naive_conv_enable;

    auto ref_solution_id = miopen::solver::Id("ConvDirectNaiveConvWrw").Value();
    auto rc              = miopenConvolutionBackwardWeightsImmediate(handle,
                                                        outputTensor,
                                                        dout_dev->GetMem(),
                                                        inputTensor,
                                                        in_dev->GetMem(),
                                                        convDesc,
                                                        weightTensor,
                                                        dwei_dev->GetMem(),
                                                        nullptr,
                                                        0,
                                                        ref_solution_id);
    if(rc != miopenStatusSuccess)
    {
        std::cout << "reference kernel fail to run "
                  << miopen::solver::Id(ref_solution_id).ToString() << std::endl;
        return rc;
    }

    if(miopen_type<Tgpu>{} == miopen_type<Tref>{})
        dwei_dev->FromGPU(GetStream(), dwei_host.data.data());
    else
    {
        auto dwei_tmp = tensor<Tgpu>(miopen::deref(weightTensor));
        dwei_dev->FromGPU(GetStream(), dwei_tmp.data.data());
        for(int i = 0; i < dwei_tmp.data.size(); i++)
        {
            dwei_host.data[i] = static_cast<Tref>(dwei_tmp.data[i]);
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>(
            "dump_bwd_dwei_gpu_ref.bin", dwei_host.data.data(), dwei_host.data.size());
    }

    // TrySaveVerificationCache(Direction::WrW, dwei_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::RunBackwardDataGPUReference()
{
    AutoConvDirectNaiveAlwaysEnable naive_conv_enable;

    auto ref_solution_id = miopen::deref(convDesc).mode == miopenTranspose
                               ? miopen::solver::Id("ConvDirectNaiveConvFwd").Value()
                               : miopen::solver::Id("ConvDirectNaiveConvBwd").Value();
    auto rc = miopenConvolutionBackwardDataImmediate(handle,
                                                     outputTensor,
                                                     dout_dev->GetMem(),
                                                     weightTensor,
                                                     wei_dev->GetMem(),
                                                     convDesc,
                                                     inputTensor,
                                                     din_dev->GetMem(),
                                                     nullptr,
                                                     0,
                                                     ref_solution_id);
    if(rc != miopenStatusSuccess)
    {
        std::cout << "reference kernel fail to run "
                  << miopen::solver::Id(ref_solution_id).ToString() << std::endl;
        return rc;
    }

    if(miopen_type<Tgpu>{} == miopen_type<Tref>{})
        din_dev->FromGPU(GetStream(), din_host.data.data());
    else
    {
        auto din_tmp = tensor<Tgpu>(miopen::deref(inputTensor));
        din_dev->FromGPU(GetStream(), din_tmp.data.data());
        for(int i = 0; i < din_tmp.data.size(); i++)
        {
            din_host.data[i] = static_cast<Tref>(din_tmp.data[i]);
        }
    }

    if(inflags.GetValueInt("dump_output"))
    {
        dumpBufferToFile<Tref>(
            "dump_bwd_din_gpu_ref.bin", din_host.data.data(), din_host.data.size());
    }

    // TrySaveVerificationCache(Direction::Bwd, din_host.data);
    return 0;
}

template <typename Tgpu, typename Tref>
std::string ConvDriver<Tgpu, Tref>::GetVerificationCacheFileName(
    const ConvDriver<Tgpu, Tref>::Direction& direction) const
{
    std::ostringstream ss;

    miopenConvolutionMode_t mode;

    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    miopenGetConvolutionNdDescriptor(convDesc,
                                     spatial_dim,
                                     &spatial_dim,
                                     pads.data(),
                                     conv_strides.data(),
                                     conv_dilations.data(),
                                     &mode);

    auto get_basename_string = [&]() {
        switch(direction)
        {
        case Direction::Fwd: return "conv_fwd_out";
        case Direction::Bwd: return "conv_bwd_dat";
        case Direction::WrW: return "conv_bwd_wei";
        case Direction::BwdBias: return "bias_bwd_dat";
        }
        return "<error in get_basename_string>"; // For gcc.
    };

    auto get_datatype_string = [](auto type) {
        if(std::is_same<decltype(type), int8_t>::value)
        {
            return "int8";
        }
        else if(std::is_same<decltype(type), float16>::value)
        {
            return "float16";
        }
        else if(std::is_same<decltype(type), float>::value)
        {
            return "float";
        }
        else if(std::is_same<decltype(type), double>::value)
        {
            return "double";
        }
        else if(std::is_same<decltype(type), bfloat16>::value)
        {
            return "bfloat16";
        }
        else
        {
            MIOPEN_THROW("unknown data type");
        }
    };

    ss << get_basename_string();
    ss << "_" << mode;
    ss << "_" << spatial_dim;
    ss << "_" << miopen::deref(convDesc).paddingMode;
    ss << "_" << miopen::deref(convDesc).GetGroupCount();
    miopen::LogRange(ss << "_", miopen::deref(inputTensor).GetLengths(), "x");
    miopen::LogRange(ss << "_", miopen::deref(weightTensor).GetLengths(), "x");
    miopen::LogRange(ss << "_", pads, "x");
    miopen::LogRange(ss << "_", conv_strides, "x");
    miopen::LogRange(ss << "_", conv_dilations, "x");
    miopen::LogRange(ss << "_", trans_output_pads, "x");
    ss << "_" << inflags.GetValueInt("pad_val");
    ss << "_" << inflags.GetValueInt("bias");
    ss << "_"
       << "GPU" << get_datatype_string(Tgpu{});
    ss << "_"
       << "REF" << get_datatype_string(Tref{});

    return ss.str();
}

template <typename Tgpu, typename Tref>
bool ConvDriver<Tgpu, Tref>::TryReadVerificationCache(
    const ConvDriver<Tgpu, Tref>::Direction& direction,
    miopenTensorDescriptor_t& tensorDesc,
    Tref* data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");

    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + GetVerificationCacheFileName(direction);

        if(std::ifstream(file_path).good())
        {
            if(readBufferFromFile<Tref>(data, GetTensorSize(tensorDesc), file_path.c_str()))
            {
                return true;
            }
        }
    }

    return false;
}

template <typename Tgpu, typename Tref>
void ConvDriver<Tgpu, Tref>::TrySaveVerificationCache(
    const ConvDriver<Tgpu, Tref>::Direction& direction, std::vector<Tref>& data) const
{
    const auto verification_cache_path = inflags.GetValueStr("verification_cache");
    if(!verification_cache_path.empty())
    {
        const auto file_path =
            verification_cache_path + "/" + GetVerificationCacheFileName(direction);
        dumpBufferToFile<Tref>(file_path.c_str(), data.data(), data.size());
    }
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::VerifyForward()
{
    if(!is_fwd)
        return 0;

    if(!is_fwd_run_failed)
        if(!TryReadVerificationCache(Direction::Fwd, outputTensor, outhost.data.data()))
        {
            if(UseGPUReference())
                RunForwardGPUReference();
            else
                RunForwardCPU();
        }

    const auto isInt8 = (data_type == miopenInt8 || data_type == miopenInt8x4);
    auto error        = is_fwd_run_failed ? std::numeric_limits<double>::max()
                                   : (isInt8 ? miopen::rms_range(outhost.data, out_int8)
                                             : miopen::rms_range(outhost.data, out.data));

    auto tolerance = GetDefaultTolerance();
    // iGemm's deviation is higher than other algorithms.
    // The reason is most likely different order of computations.
    if(is_fwd_igemm)
        tolerance = tolerance * 10;

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Convolution FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }

    std::cout << "Forward Convolution Verifies OK on " << (UseGPUReference() ? "GPU" : "CPU")
              << " reference (" << error << ')' << std::endl;
    return 0;
}

template <typename Tgpu, typename Tref>
int ConvDriver<Tgpu, Tref>::VerifyBackward()
{
    if(data_type == miopenInt8 || data_type == miopenInt8x4)
    {
        std::cout << "Int8 Backward Convolution is not supported" << std::endl;
        return 0;
    }

    if(!(is_bwd || is_wrw))
        return 0;

    int cumulative_rc = 0;
    if(is_bwd)
    {
        if(!is_bwd_run_failed)
            if(!TryReadVerificationCache(Direction::Bwd, inputTensor, din_host.data.data()))
            {
                if(UseGPUReference())
                    RunBackwardDataGPUReference();
                else
                    RunBackwardDataCPU();
            }

        auto error_data = is_bwd_run_failed ? std::numeric_limits<double>::max()
                                            : miopen::rms_range(din_host.data, din);

        auto tolerance = GetDefaultTolerance();
        // iGemm's deviation is higher than other algorithms.
        // The reason is most likely different order of computations.
        if(is_bwd_igemm)
            tolerance = tolerance * 10;

        if(!std::isfinite(error_data) || error_data > tolerance)
        {
            std::cout << "Backward Convolution Data FAILED: " << error_data << " > " << tolerance
                      << std::endl;
            cumulative_rc |= EC_VerifyBwd;
        }
        else
        {
            std::cout << "Backward Convolution Data Verifies OK on "
                      << (UseGPUReference() ? "GPU" : "CPU") << " reference (" << error_data << ')'
                      << std::endl;
        }
    }

    if(is_wrw)
    {
        if(!is_wrw_run_failed)
            if(!TryReadVerificationCache(Direction::WrW, weightTensor, dwei_host.data.data()))
            {
                if(UseGPUReference())
                    RunBackwardWeightsGPUReference();
                else
                    RunBackwardWeightsCPU();
            }

        // WrW deviation is ~twice worse than Bwd due to more FP computations involved,
        // which means more roundings, so GPU amd CPU computations diverge more.
        auto tolerance = 2 * GetDefaultTolerance();
        // Winograd and iGemm WrW algorithms reveal bigger deviation than other algos.
        if(is_wrw_winograd && std::is_same<Tgpu, float>::value)
        {
            tolerance *= 10;
        }
        else if(is_wrw_igemm)
        {
            if(std::is_same<Tgpu, float>::value)
#if WORKAROUND_ISSUE_2176
                tolerance = 0.01;
#else
                tolerance *= 10;
#endif
            else if(std::is_same<Tgpu, float16>::value)
                tolerance *= 5;
        }

        auto error_weights = is_wrw_run_failed ? std::numeric_limits<double>::max()
                                               : miopen::rms_range(dwei_host.data, dwei);

        if(!std::isfinite(error_weights) || error_weights > tolerance)
        {
            std::cout << "Backward Convolution Weights FAILED: " << error_weights << " > "
                      << tolerance << std::endl;
            cumulative_rc |= EC_VerifyWrw;
        }
        else
        {
            std::cout << "Backward Convolution Weights Verifies OK on "
                      << (UseGPUReference() ? "GPU" : "CPU") << " reference (" << error_weights
                      << ')' << std::endl;
        }
    }

    if(inflags.GetValueInt("bias") != 0)
    {
        if(!TryReadVerificationCache(Direction::BwdBias, biasTensor, db_host.data.data()))
        {
            RunBackwardBiasCPU();
        }

        auto error_bias      = miopen::rms_range(db_host.data, db);
        const auto tolerance = GetDefaultTolerance();
        if(!std::isfinite(error_bias) || error_bias > tolerance)
        {
            std::cout << "Backward Convolution Bias FAILED: " << error_bias << " > " << tolerance
                      << std::endl;
            cumulative_rc |= EC_VerifyBwdBias;
        }
        else
        {
            std::cout << "Backward Convolution Bias Verifies OK on "
                      << (UseGPUReference() ? "GPU" : "CPU") << " reference (" << error_bias << ')'
                      << std::endl;
        }
    }

    return cumulative_rc;
}

#endif // GUARD_MIOPEN_CONV_DRIVER_HPP
