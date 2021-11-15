#include "../include/conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp"

namespace ck {
namespace driver {
bool TunableConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::IsValid() const
{
    if(!((MPerXDL == 32 && NPerXDL == 32) || (MPerXDL == 16 && NPerXDL == 16)))
        return false;

    if(!(K1 % 4 == 0))
        return false;

    if(!(K0PerBlock % 4 == 0))
        return false;

    if(!(MPerBlock % MPerXDL == 0))
        return false;

    if(!(NPerBlock % NPerXDL == 0))
        return false;

    if(!(MPerBlock % (MPerXDL * MRepeat) == 0))
        return false;
    if(!(NPerBlock % (NPerXDL * NRepeat) == 0))
        return false;
    const int MWaves   = MPerBlock / (MPerXDL * MRepeat);
    const int NWaves   = NPerBlock / (NPerXDL * NRepeat);
    const int WaveSize = 64;
    if(!(MWaves * NWaves * WaveSize == BlockSize))
        return false;
    // A matrix copy
    {
        const auto& thread_slice_lengths = ABlockTransferThreadSliceLengths_K0_M_K1;
        const auto& cluster_lengths      = ABlockTransferThreadClusterLengths_K0_M_K1;
        const auto block_slice_lengths   = std::array<int, 3>{K0PerBlock, MPerBlock, K1};
        // check number of working thread
        const int num_work_thread = std::accumulate(
            cluster_lengths.begin(), cluster_lengths.end(), 1, std::multiplies<int>{});
        if(!(BlockSize == num_work_thread))
            return false;

        // check block slice lengths vs thread slice lengths vs cluster lengths
        for(int i = 0; i < thread_slice_lengths.size(); ++i)
        {
            if(!(cluster_lengths[i] * thread_slice_lengths[i] == block_slice_lengths[i]))
                return false;
        }

        if(ABlockTransferSrcVectorDim >= ABlockTransferThreadClusterArrangeOrder.size())
            return false;

        if(!(thread_slice_lengths[2] % ABlockTransferSrcScalarPerVector == 0))
            return false;

        if(!(thread_slice_lengths[2] % ABlockTransferDstScalarPerVector_K1 == 0))
            return false;
    }

    // B matrix copy
    {
        const auto& thread_slice_lengths = BBlockTransferThreadSliceLengths_K0_N_K1;
        const auto& cluster_lengths      = BBlockTransferThreadClusterLengths_K0_N_K1;
        const auto block_slice_lengths   = std::array<int, 3>{K0PerBlock, NPerBlock, K1};
        // check number of working thread
        const int num_work_thread = std::accumulate(
            cluster_lengths.begin(), cluster_lengths.end(), 1, std::multiplies<int>{});
        if(!(BlockSize == num_work_thread))
            return false;

        // check block slice lengths vs thread slice lengths vs cluster lengths
        for(int i = 0; i < thread_slice_lengths.size(); ++i)
        {
            if(!(cluster_lengths[i] * thread_slice_lengths[i] == block_slice_lengths[i]))
                return false;
        }

        if(ABlockTransferSrcVectorDim >= ABlockTransferThreadClusterArrangeOrder.size())
            return false;

        if(!(thread_slice_lengths[2] % BBlockTransferSrcScalarPerVector == 0))
            return false;

        if(!(thread_slice_lengths[2] % BBlockTransferDstScalarPerVector_K1 == 0))
            return false;
    }
    // C Matrix
    {
        if(CThreadTransferSrcDstAccessOrder.size() <= CThreadTransferSrcDstVectorDim)
            return false;
    }
    return true;
}

//////////////////////////////////////////
std::tuple<CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk, bool>
ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::CalculateCompileParameterBasedOnTunable(
    const ConvolutionProblemDescriptor& conv_problem_desc,
    const TunableConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& tunable)
{
    const int C = conv_problem_desc.C;
    const int Y = conv_problem_desc.Y;
    const int X = conv_problem_desc.X;

    if(!(conv_problem_desc.InDataTypeEnum == tunable.ABDataTypeEnum &&
         conv_problem_desc.WeiDataTypeEnum == tunable.ABDataTypeEnum &&
         conv_problem_desc.OutDataTypeEnum == tunable.CDataTypeEnum))
        return std::make_tuple(CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{}, false);

    const auto ABDataTypeEnum = conv_problem_desc.InDataTypeEnum;
    const auto CDataTypeEnum  = conv_problem_desc.OutDataTypeEnum;

    DataTypeEnum_t AccDataTypeEnum;

    if(ABDataTypeEnum == DataTypeEnum_t::Float || ABDataTypeEnum == DataTypeEnum_t::Half)
    {
        AccDataTypeEnum = DataTypeEnum_t::Float;
    }
    else if(ABDataTypeEnum == DataTypeEnum_t::Int8)
    {
        AccDataTypeEnum = DataTypeEnum_t::Int32;
    }
    else
    {
        return std::make_tuple(CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{}, false);
    }

    const int BlockSize = tunable.BlockSize;

    const int MPerBlock  = tunable.MPerBlock;
    const int NPerBlock  = tunable.NPerBlock;
    const int K0PerBlock = tunable.K0PerBlock;

    const int MPerXDL = tunable.MPerXDL;
    const int NPerXDL = tunable.NPerXDL;
    const int K1      = tunable.K1;

    const int MRepeat = tunable.MRepeat;
    const int NRepeat = tunable.NRepeat;

    const auto ABlockTransferThreadSliceLengths_K0_M_K1 =
        tunable.ABlockTransferThreadSliceLengths_K0_M_K1;
    const auto ABlockTransferThreadClusterLengths_K0_M_K1 =
        tunable.ABlockTransferThreadClusterLengths_K0_M_K1;
    const auto ABlockTransferThreadClusterArrangeOrder =
        tunable.ABlockTransferThreadClusterArrangeOrder;
    const auto ABlockTransferSrcAccessOrder = tunable.ABlockTransferSrcAccessOrder;

    const int ABlockTransferSrcVectorDim          = tunable.ABlockTransferSrcVectorDim;
    const int ABlockTransferSrcScalarPerVector    = tunable.ABlockTransferSrcScalarPerVector;
    const int ABlockTransferDstScalarPerVector_K1 = tunable.ABlockTransferDstScalarPerVector_K1;
    const bool AThreadTransferSrcResetCoordinateAfterRun =
        tunable.AThreadTransferSrcResetCoordinateAfterRun;

    const auto BBlockTransferThreadSliceLengths_K0_N_K1 =
        tunable.BBlockTransferThreadSliceLengths_K0_N_K1;
    const auto BBlockTransferThreadClusterLengths_K0_N_K1 =
        tunable.BBlockTransferThreadClusterLengths_K0_N_K1;
    const auto BBlockTransferThreadClusterArrangeOrder =
        tunable.BBlockTransferThreadClusterArrangeOrder;
    const auto BBlockTransferSrcAccessOrder = tunable.BBlockTransferSrcAccessOrder;

    const int BBlockTransferSrcVectorDim          = tunable.BBlockTransferSrcVectorDim;
    const int BBlockTransferSrcScalarPerVector    = tunable.BBlockTransferSrcScalarPerVector;
    const int BBlockTransferDstScalarPerVector_K1 = tunable.BBlockTransferDstScalarPerVector_K1;
    const bool BThreadTransferSrcResetCoordinateAfterRun =
        tunable.BThreadTransferSrcResetCoordinateAfterRun;
    // C threadwise copy
    const auto CThreadTransferSrcDstAccessOrder  = tunable.CThreadTransferSrcDstAccessOrder;
    const auto CThreadTransferSrcDstVectorDim    = tunable.CThreadTransferSrcDstVectorDim;
    const auto CThreadTransferDstScalarPerVector = tunable.CThreadTransferDstScalarPerVector;

    const int GK = C * Y * X;

    if(!(GK % (K0PerBlock * K1) == 0))
        return std::make_tuple(CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{}, false);

    const bool HasMainKBlockLoop = (GK > K0PerBlock * K1);

    return std::make_tuple(
        CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{
            ABDataTypeEnum,
            AccDataTypeEnum,
            CDataTypeEnum,
            BlockSize,
            MPerBlock,
            NPerBlock,
            K0PerBlock,
            MPerXDL,
            NPerXDL,
            K1,
            MRepeat,
            NRepeat,
            ABlockTransferThreadSliceLengths_K0_M_K1,
            ABlockTransferThreadClusterLengths_K0_M_K1,
            ABlockTransferThreadClusterArrangeOrder,
            ABlockTransferSrcAccessOrder,
            ABlockTransferSrcVectorDim,
            ABlockTransferSrcScalarPerVector,
            ABlockTransferDstScalarPerVector_K1,
            AThreadTransferSrcResetCoordinateAfterRun,
            BBlockTransferThreadSliceLengths_K0_N_K1,
            BBlockTransferThreadClusterLengths_K0_N_K1,
            BBlockTransferThreadClusterArrangeOrder,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BBlockTransferDstScalarPerVector_K1,
            BThreadTransferSrcResetCoordinateAfterRun,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            1, // M01
            1, // N01
            HasMainKBlockLoop},
        true);
}

auto ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetDefaultCompileParameter(
    const ConvolutionProblemDescriptor& conv_problem_desc)
{
    for(const auto& tunable : generate_tunable_list_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk())
    {
        if(!tunable.IsValid())
            continue;

        CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk compile_param{};
        bool found = false;

        std::tie(compile_param, found) =
            CalculateCompileParameterBasedOnTunable(conv_problem_desc, tunable);

        if(found && IsValidCompileParameter(conv_problem_desc, compile_param))
            return std::make_tuple(compile_param, true);
    }

    return std::make_tuple(CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk{}, false);
}

bool ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::IsApplicable(
    const ConvolutionProblemDescriptor& conv_problem_desc)
{
    bool found = false;

    std::tie(std::ignore, found) = GetDefaultCompileParameter(conv_problem_desc);

    return found;
}
bool ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::IsValidCompileParameter(
    const ConvolutionProblemDescriptor& conv_problem_desc,
    const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param)
{
    const int GemmK1         = compile_param.K1;
    const int GemmMPerBlock  = compile_param.MPerBlock;
    const int GemmNPerBlock  = compile_param.NPerBlock;
    const int GemmK0PerBlock = compile_param.K0PerBlock;

    const int N  = conv_problem_desc.N;
    const int K  = conv_problem_desc.K;
    const int C  = conv_problem_desc.C;
    const int Y  = conv_problem_desc.Y;
    const int X  = conv_problem_desc.X;
    const int Ho = conv_problem_desc.Ho;
    const int Wo = conv_problem_desc.Wo;

    const auto GemmM = N * Ho * Wo;
    const auto GemmN = K;
    const auto GemmK = Y * X * C;

    if(!(GemmM % GemmMPerBlock == 0))
        return false;
    if(!(GemmN % GemmNPerBlock == 0))
        return false;

    if(!(GemmK % GemmK1 == 0))
        return false;

    if(!(GemmK % (GemmK0PerBlock * GemmK1) == 0))
        return false;

    if(!compile_param.HasMainKBlockLoop)
    {
        if(!(GemmK == (GemmK0PerBlock * GemmK1)))
            return false;
    }

    if(!(C % compile_param.ABlockTransferSrcScalarPerVector == 0))
        return false;

    return true;
}

int ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk::GetGridSize(
    const ConvolutionProblemDescriptor& conv_problem_desc,
    const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param)
{
    const int N  = conv_problem_desc.N;
    const int K  = conv_problem_desc.K;
    const int Ho = conv_problem_desc.Ho;
    const int Wo = conv_problem_desc.Wo;

    const auto GemmM = N * Ho * Wo;
    const auto GemmN = K;

    const int GemmMPerBlock = compile_param.MPerBlock;
    const int GemmNPerBlock = compile_param.NPerBlock;

    return GemmM * GemmN / (GemmMPerBlock * GemmNPerBlock);
}

} // namespace driver
} // namespace ck
