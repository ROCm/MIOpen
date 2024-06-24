/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/pooling/solvers.hpp>

#include <miopen/pooling/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/pooling.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace pooling {

namespace {

constexpr int top_w_per_work = 1;
constexpr int top_h_per_work = 4;
constexpr int top_d_per_work = 2;

struct kernel_params
{
    uint32_t stride_d;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t kernel_sz_d;
    uint32_t kernel_sz_h;
    uint32_t kernel_sz_w;

    kernel_params(const miopen::pooling::ProblemDescription& p)
    {
        const auto& pd = p.GetPooling();
        stride_d       = pd.strides[0];
        stride_h       = pd.strides[1];
        stride_w       = pd.strides[2];
        kernel_sz_d    = pd.lens[0];
        kernel_sz_h    = pd.lens[1];
        kernel_sz_w    = pd.lens[2];
    }
};

std::size_t sizeof_kernel_FLOAT(const miopen::pooling::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

inline std::size_t RoundUpToMultiple(std::size_t v, std::size_t m)
{
    assert(m > 0);
    return ((v + m - 1) / m) * m;
}

// Compute amount of private memory required for holding the arrays defined
// in the "mloPoolingCkFwd" kernel:
//
// #define BOT_TILE_W ((TOP_W_PER_WORK - 1) * STRIDE_W + KERNEL_SZ_W)
// #define BOT_TILE_H ((TOP_H_PER_WORK - 1) * STRIDE_H + KERNEL_SZ_H)
// #define BOT_TILE_D ((TOP_D_PER_WORK - 1) * STRIDE_D + KERNEL_SZ_D)
//
// _FLOAT bot_data[BOT_TILE_D][BOT_TILE_H][BOT_TILE_W];
//
std::size_t sizeof_private_memory(const miopen::pooling::ProblemDescription& problem)
{
    const kernel_params kp(problem);

    const std::size_t bot_tile_w = ((top_w_per_work - 1) * kp.stride_w + kp.kernel_sz_w);
    const std::size_t bot_tile_h = ((top_h_per_work - 1) * kp.stride_h + kp.kernel_sz_h);
    const std::size_t bot_tile_d = ((top_d_per_work - 1) * kp.stride_d + kp.kernel_sz_d);

    const auto sizeof_bot_data =
        sizeof_kernel_FLOAT(problem) * bot_tile_d * bot_tile_h * bot_tile_w;
    MIOPEN_LOG_T("sizeof_bot_data " << sizeof_bot_data);

    /// \ref alignment_of_arrays_in_gpu_memory
    return RoundUpToMultiple(sizeof_bot_data, 16);
}

} // namespace

bool PoolingForwardCk::IsApplicable(const ExecutionContext& context,
                                    const miopen::pooling::ProblemDescription& problem) const
{

    return problem.GetDirection() == miopen::pooling::Direction::Forward                      //
           && problem.GetXDesc().GetNumDims() == 5                                            //
           && problem.GetXDesc().GetLayout("NCDHW") == "NCDHW"                                //
           && problem.GetYDesc().GetLayout("NCDHW") == "NCDHW"                                //
           && problem.GetXDesc().GetType() == problem.GetYDesc().GetType()                    //
           && (problem.GetXDesc().GetType() == miopenFloat                                    //
               || problem.GetXDesc().GetType() == miopenHalf)                                 //
           && (problem.GetPooling().GetMode() == miopenPoolingMax                             //
               || problem.GetPooling().GetMode() == miopenPoolingAverage                      //
               || problem.GetPooling().GetMode() == miopenPoolingAverageInclusive)            //
           && sizeof_private_memory(problem) <= TargetProperties::GetMaxWaveScratchSize()     //
                                                    / context.GetStream().GetWavefrontWidth() //
           /// \todo This solver does not support workspace index mask mode yet.
           &&
           !(problem.GetPooling().GetMode() == miopenPoolingMax                                 //
             && problem.GetPooling().GetWorkspaceIndexMode() == miopenPoolingWorkspaceIndexMask //
             && problem.SaveIndex() == true);
}

ConvSolution PoolingForwardCk::GetSolution(const ExecutionContext&,
                                           const miopen::pooling::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const int batch = problem.GetXDesc().GetLengths()[0];
    const int chal  = problem.GetXDesc().GetLengths()[1];

    const kernel_params kp(problem);

    const int top_d = *(problem.GetYDesc().GetLengths().rbegin() + 2);
    const int top_h = *(problem.GetYDesc().GetLengths().rbegin() + 1);
    const int top_w = *(problem.GetYDesc().GetLengths().rbegin());

    const int top_blk_w = std::max((top_w + top_w_per_work - 1) / top_w_per_work, 1);
    const int top_blk_h = std::max((top_h + top_h_per_work - 1) / top_h_per_work, 1);
    const int top_blk_d = std::max((top_d + top_d_per_work - 1) / top_d_per_work, 1);

    const int max_activ_workitem = 65536;
    const int total_work         = batch * chal * top_blk_w * top_blk_h * top_blk_d;
    const int activ_work         = std::min(total_work, max_activ_workitem);

    {
        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenPoolingCK.cl";
        kernel.kernel_name = "mloPoolingCKFwd";

        int pooling_method = (problem.GetPooling().mode == miopenPoolingMax)
                                 ? MLO_POOLING_OP_MAX
                                 : ((problem.GetPooling().mode == miopenPoolingAverage)
                                        ? MLO_POOLING_OP_AVE
                                        : MLO_POOLING_OP_AVE_INCLUSIVE);

        const size_t lcl_work = 64;
        const size_t grp_num  = (activ_work + lcl_work - 1) / lcl_work;

        auto build_params = KernelBuildParameters{
            {"MLO_POOLING_OP_ID", static_cast<long long>(pooling_method)},
            {"MAX_ACTIV_WORKITEM", static_cast<unsigned>(max_activ_workitem)},
            {"MLO_POOLING_GROUP_SZ0", static_cast<long long>(lcl_work)},
            {"MLO_POOLING_GROUP_SZ1", 1},
            {"MLO_POOLING_GROUP_SZ2", 1},
            {"TOP_W_PER_WORK", top_w_per_work},
            {"TOP_H_PER_WORK", top_h_per_work},
            {"TOP_D_PER_WORK", top_d_per_work},
            {"KERNEL_SZ_D", kp.kernel_sz_d},
            {"KERNEL_SZ_H", kp.kernel_sz_h},
            {"KERNEL_SZ_W", kp.kernel_sz_w},
            {"STRIDE_D", kp.stride_d},
            {"STRIDE_H", kp.stride_h},
            {"STRIDE_W", kp.stride_w},
            {"MLO_POOLING_INDEX_TYPE",
             get_pooling_index_type_name(problem.GetPooling().GetIndexType())},
            {"MLO_POOLING_INDEX_MAX",
             get_pooling_index_type_max_name(problem.GetPooling().GetIndexType())},
        };

        if(problem.SaveIndex())
        {
            build_params << KernelBuildParameters{
                {"MLO_POOLING_SAVE_INDEX"},
            };
        }

        build_params << GetDataTypeKBP(problem.GetXDesc().GetType());

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk = {lcl_work, 1, 1};
        kernel.g_wk = {lcl_work * grp_num, 1, 1};

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::pooling::FwdInvokeParams>();

            const int batch_ = params.xDesc.GetLengths()[0];
            const int chal_  = params.xDesc.GetLengths()[1];

            const int top_d_ = *(params.yDesc.GetLengths().rbegin() + 2);
            const int top_h_ = *(params.yDesc.GetLengths().rbegin() + 1);
            const int top_w_ = *(params.yDesc.GetLengths().rbegin());

            const int top_blk_w_ = std::max((top_w_ + top_w_per_work - 1) / top_w_per_work, 1);
            const int top_blk_h_ = std::max((top_h_ + top_h_per_work - 1) / top_h_per_work, 1);
            const int top_blk_d_ = std::max((top_d_ + top_d_per_work - 1) / top_d_per_work, 1);

            const int total_work_ = batch_ * chal_ * top_blk_w_ * top_blk_h_ * top_blk_d_;

            kernel(params.x,
                   params.y,
                   params.workspace,
                   static_cast<unsigned>(params.pooling.pads[0]),
                   static_cast<unsigned>(params.pooling.pads[1]),
                   static_cast<unsigned>(params.pooling.pads[2]),
                   static_cast<unsigned>(batch_),
                   static_cast<unsigned>(chal_),
                   static_cast<unsigned>(params.xDesc.GetLengths()[2]),
                   static_cast<unsigned>(params.xDesc.GetLengths()[3]),
                   static_cast<unsigned>(params.xDesc.GetLengths()[4]),
                   static_cast<unsigned>(top_d_),
                   static_cast<unsigned>(top_h_),
                   static_cast<unsigned>(top_w_),
                   static_cast<unsigned>(params.xDesc.GetStrides()[0]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[1]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[2]),
                   static_cast<unsigned>(params.xDesc.GetStrides()[3]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[0]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[1]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[2]),
                   static_cast<unsigned>(params.yDesc.GetStrides()[3]),
                   static_cast<unsigned>(total_work_));
        };
    };

    return result;
}

std::size_t
PoolingForwardCk::GetWorkspaceSize(const ExecutionContext&,
                                   const miopen::pooling::ProblemDescription& problem) const
{
    if(problem.GetPooling().GetMode() != miopenPoolingMax)
        return 0;
    return problem.GetYDesc().GetElementSize() * get_data_size(problem.GetPooling().GetIndexType());
}

} // namespace pooling

} // namespace solver

} // namespace miopen







#include <vector>
#include <cstdint>

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpGWrw = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::tensor_layout::convolution::NHWGK,
    DataType,
    DataType,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

template <typename DataType>
using DeviceOpGWrwPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGWrw<DataType>>;

namespace {

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        // CK strides are in GNCDHW order
        if(problem.IsLayoutNHWC())
        {
            // first entry reserved for G's stride
            auto copy_strides = [](const auto& src, auto& dst) {
                assert(dst.size() == (src.size() + 1));
                std::copy(src.begin(), src.end(), dst.begin() + 1);
            };
            copy_strides(problem.GetIn().GetStrides(), in_strides);
            copy_strides(problem.GetOut().GetStrides(), out_strides);
            copy_strides(problem.GetWeights().GetStrides(), wei_strides);

            // On a backward pass, problem.GetIn() means y(or out),
            // and problem.GetOut means x(or in)
            /// \todo remove this when we stop swapping in and out tensors/descriptors
            std::swap(in_strides, out_strides);

            // Now compute G's stride
            in_strides[0]  = C;
            out_strides[0] = K;
            wei_strides[0] = K * wei_strides[1];
        }
        else
        {
            assert(problem.IsLayoutDefault()); // already checked in IsApplicable
            // for default layout, we produce packed strides for NHWC layout
            // because we transpose to NHWC layout before calling CK kernel
            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        }

        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }
    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t x,
                    Data_t dw,
                    ConstData_t dy,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const ConvWrwTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.x, tensors.dw, tensors.dy, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        // Creat dummy workspace to pass the ck IsSupportedArgument check.
        int dummy_var = 1;
        conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &dummy_var);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    ck::index_t split_k = 1;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemmGroupWrwXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmGroupWrwXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
}

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
static std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(
        kernel.substr(kernel.find('<') + 1, kernel.find('>') - kernel.find('<') - 1));
    while(std::getline(tokenStream, token, ','))
    {
        token.erase(remove_if(token.begin(), token.end(), isspace),
                    token.end()); // strip whitespace
        tokens.push_back(token);
    }
    return tokens;
}

void PerformanceConfigHipImplicitGemmGroupWrwXdlops::InitHeuristicKernelIDs()
{
    for(int i = 0; i < valid_kernels.size(); i++)
    {
        if(valid_kernels[i].find("DeviceGroupedConvBwdWeight_Xdl_CShuffle") != std::string::npos)
        {
            heuristic_indexes.push_back(i);
            heuristic_kernels.push_back(GetKernelAsTokens(valid_kernels[i]));
        }
    }
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::ModelApplyToken(int idx, std::string value)
{
    if(idx == 13)
        idx += 1; // skip

    auto eraseBegin = std::remove_if(
        heuristic_indexes.begin(), heuristic_indexes.end(), [&](int heuristic_index) {
            return heuristic_kernels[heuristic_index][idx] != value;
        });

    if(eraseBegin != heuristic_indexes.begin())
    {
        heuristic_indexes.erase(eraseBegin, heuristic_indexes.end());
        return true;
    }
    return false;
}

static std::vector<float> GetFeatures(const ProblemDescription& problem)
{
    std::size_t n = 18;
    std::vector<float> features(n * n, 0.0f);
    features[0]           = 1.0;
    features[n + 1]       = problem.GetOutChannels();
    features[2 * n + 2]   = problem.GetOutHeight();
    features[3 * n + 3]   = problem.GetOutWidth();
    features[4 * n + 4]   = problem.GetInChannels();
    features[5 * n + 5]   = problem.GetInHeight();
    features[6 * n + 6]   = problem.GetInWidth();
    features[7 * n + 7]   = problem.GetWeightsHeight();
    features[8 * n + 8]   = problem.GetWeightsWidth();
    features[9 * n + 9]   = problem.GetPadH();
    features[10 * n + 10] = problem.GetPadW();
    features[11 * n + 11] = problem.GetKernelStrideH();
    features[12 * n + 12] = problem.GetKernelStrideW();
    features[13 * n + 13] = problem.GetDilationH();
    features[14 * n + 14] = problem.GetDilationW();
    features[15 * n + 15] = problem.GetBatchSize();
    features[16 * n + 16] = problem.GetInDataType() == miopenFloat ? 2.0 : 1.0;
    features[17 * n + 17] = problem.GetGroupCount();
    return features;
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::RunParameterPredictionModel(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs>(
        problem); // filter valid_kernel ID's
    InitHeuristicKernelIDs();
    static const std::string& arch  = ctx.GetStream().GetDeviceName();
    static const std::string solver = "ConvHipIgemmGroupXdlops";
    std::vector<float> features     = GetFeatures(problem);
    if(ai::tuning::ModelSetParams(
           arch, solver, problem.GetDirection(), features, true, [&](int idx, std::string value) {
               return this->ModelApplyToken(idx, value);
           }))
    {
        index     = heuristic_indexes[0];
        kernel_id = valid_kernels[index];
        MIOPEN_LOG_I("Params set by AI: " << ToString());
        return true;
    }
    return false;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR))
        return false;
    return true;
}

void PerformanceConfigHipImplicitGemmGroupWrwXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    // these seem redundant
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(IsModelApplicable(ctx, problem))
    {
        if(problem.GetInDataType() == miopenFloat)
        {
            if(RunParameterPredictionModel<float>(ctx, problem))
                return;
        }
        else
        {
            if(RunParameterPredictionModel<ck::half_t>(ctx, problem))
                return;
        }
    }
#endif
    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::SetNextValue(const ProblemDescription& problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        switch(problem.GetInDataType())
        {
        case miopenHalf: Init<ck::half_t>(problem); break;
        case miopenFloat: Init<float>(problem); break;
        case miopenInt8: Init<int8_t>(problem); break;
        case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8:
        case miopenBFloat8:
        case miopenDouble: break;
        }
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
#endif
        return false;
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(problem);
    case miopenInt8: return CheckIsSupportCKArgs<int8_t>(problem);
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(problem);
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupWrwXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupWrwXdlops
ConvHipImplicitGemmGroupWrwXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupWrwXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupWrwXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupWrwXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemmGroupWrwXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupWrwXdlops
ConvHipImplicitGemmGroupWrwXdlops::Search(const ExecutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupWrwXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS))
        return false;
    if(env::enabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC))
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat: return CheckCKApplicability<float>(problem);
    case miopenInt8: return CheckCKApplicability<int8_t>(problem);
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(problem);
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmGroupWrwXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupWrwXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryWrwNCHW<2,
                                             DeviceOpGWrwPtrs<T>,
                                             CKArgs,
                                             miopen::conv::WrWInvokeParams>(
                ctx, problem, config.kernel_id);
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryNHWC<DeviceOpGWrwPtrs<T>,
                                          CKArgs,
                                          miopen::conv::WrWInvokeParams>(
                ctx, problem, config.kernel_id);
        });

#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
