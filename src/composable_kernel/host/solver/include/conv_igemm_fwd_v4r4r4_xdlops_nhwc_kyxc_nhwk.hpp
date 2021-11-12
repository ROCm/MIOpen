#ifndef CONV_IGEMM_FWD_V4R4R4_XDLOPS_NHWC_KYXC_NHWK_HPP
#define CONV_IGEMM_FWD_V4R4R4_XDLOPS_NHWC_KYXC_NHWK_HPP

#include <numeric>
#include <sstream>

namespace ck {
namespace driver {

struct CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk
{
    ck::DataTypeEnum_t ABDataTypeEnum  = ck::DataTypeEnum_t::Unknown;
    ck::DataTypeEnum_t AccDataTypeEnum = ck::DataTypeEnum_t::Unknown;
    ck::DataTypeEnum_t CDataTypeEnum   = ck::DataTypeEnum_t::Unknown;

    int BlockSize = -1;

    int MPerBlock  = -1;
    int NPerBlock  = -1;
    int K0PerBlock = -1;

    int MPerXDL = -1;
    int NPerXDL = -1;
    int K1      = -1;

    int MRepeat = -1;
    int NRepeat = -1;

    std::array<int, 3> ABlockTransferThreadSliceLengths_K0_M_K1   = {-1, -1, -1};
    std::array<int, 3> ABlockTransferThreadClusterLengths_K0_M_K1 = {-1, -1, -1};
    std::array<int, 3> ABlockTransferThreadClusterArrangeOrder    = {-1, -1, -1};
    std::array<int, 3> ABlockTransferSrcAccessOrder               = {-1, -1, -1};
    int ABlockTransferSrcVectorDim                                = -1;
    int ABlockTransferSrcScalarPerVector                          = -1;
    int ABlockTransferDstScalarPerVector_K1                       = -1;
    bool AThreadTransferSrcResetCoordinateAfterRun                = false;

    std::array<int, 3> BBlockTransferThreadSliceLengths_K0_N_K1   = {-1, -1, -1};
    std::array<int, 3> BBlockTransferThreadClusterLengths_K0_N_K1 = {-1, -1, -1};
    std::array<int, 3> BBlockTransferThreadClusterArrangeOrder    = {-1, -1, -1};
    std::array<int, 3> BBlockTransferSrcAccessOrder               = {-1, -1, -1};
    int BBlockTransferSrcVectorDim                                = -1;
    int BBlockTransferSrcScalarPerVector                          = -1;
    int BBlockTransferDstScalarPerVector_K1                       = -1;
    bool BThreadTransferSrcResetCoordinateAfterRun                = false;

    std::array<int, 8> CThreadTransferSrcDstAccessOrder = {-1, -1, -1, -1, -1, -1, -1, -1};
    int CThreadTransferSrcDstVectorDim                  = -1;
    int CThreadTransferDstScalarPerVector               = -1;
    int M01                                             = -1;
    int N01                                             = -1;
    bool HasMainKBlockLoop                              = true;
};

struct TunableConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk
{
    ck::DataTypeEnum_t ABDataTypeEnum;
    ck::DataTypeEnum_t CDataTypeEnum;

    int BlockSize;

    int MPerBlock;
    int NPerBlock;
    int K0PerBlock;

    int MPerXDL;
    int NPerXDL;
    int K1;

    int MRepeat;
    int NRepeat;

    std::array<int, 3> ABlockTransferThreadSliceLengths_K0_M_K1;
    std::array<int, 3> ABlockTransferThreadClusterLengths_K0_M_K1;
    std::array<int, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> ABlockTransferSrcAccessOrder;
    int ABlockTransferSrcVectorDim;
    int ABlockTransferSrcScalarPerVector;
    int ABlockTransferDstScalarPerVector_K1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 3> BBlockTransferThreadSliceLengths_K0_N_K1;
    std::array<int, 3> BBlockTransferThreadClusterLengths_K0_N_K1;
    std::array<int, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<int, 3> BBlockTransferSrcAccessOrder;
    int BBlockTransferSrcVectorDim;
    int BBlockTransferSrcScalarPerVector;
    int BBlockTransferDstScalarPerVector_K1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<int, 8> CThreadTransferSrcDstAccessOrder;
    int CThreadTransferSrcDstVectorDim;
    int CThreadTransferDstScalarPerVector;

    int M01;
    int N01;

    bool IsValid() const
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
};

inline static auto generate_tunable_list_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk()
{
    constexpr auto f32 = ck::DataTypeEnum_t::Float;
    constexpr auto f16 = ck::DataTypeEnum_t::Half;

    return std::vector<TunableConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk>{
        // clang-format off
        // fp32
        {f32, f32, 256, 256, 128, 4, 32, 32, 4, 4, 2, {1, 4, 4}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false, 
            {1, 2, 4}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f32, f32, 256, 128, 128, 4, 32, 32, 4, 2, 2, {1, 2, 4}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false, 
            {1, 2, 4}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},

        // fp16
        {f16, f16, 256, 256, 256, 4, 32, 32, 8, 4, 4, {1, 4, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 4, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f16, f16, 256, 256, 128, 4, 32, 32, 8, 4, 2, {1, 4, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 2, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f16, f16, 256, 128, 256, 4, 32, 32, 8, 2, 4, {1, 2, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 4, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f16, f16, 256, 128, 128, 4, 32, 32, 8, 2, 2, {1, 2, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 2, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f16, f16, 256, 128, 64, 4, 32, 32, 8, 2, 2, {1, 4, 8}, {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 2, 8}, {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},
        {f16, f16, 256, 128, 64, 4, 32, 32, 8, 2, 1, {1, 2, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, 
            {1, 1, 8}, {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 8, 8, false, {2, 3, 0, 1, 7, 5, 4, 6}, 7, 1, 1, 1},

        // i8

        // clang-format on
    };
}

// TODO make this common interface and write specs for it
struct ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk
{
    static auto CalculateCompileParameterBasedOnTunable(
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

    static auto GetDefaultCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc)
    {
        for(const auto& tunable :
            generate_tunable_list_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk())
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

    static bool IsApplicable(const ConvolutionProblemDescriptor& conv_problem_desc)
    {
        bool found = false;

        std::tie(std::ignore, found) = GetDefaultCompileParameter(conv_problem_desc);

        return found;
    }

    static bool IsValidCompileParameter(
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
    };

    static int
    GetBlockSize(const ConvolutionProblemDescriptor&,
                 const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param)
    {
        return compile_param.BlockSize;
    }

    static int
    GetGridSize(const ConvolutionProblemDescriptor& conv_problem_desc,
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

    static std::size_t GetWorkSpaceSize(const ConvolutionProblemDescriptor&,
                                        const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk&)
    {
        // workspace is used for save transformed tensor descritpors created by prepare kernel
        return 4096L;
    }

    static std::size_t GetMaxWorkSpaceSize(const ConvolutionProblemDescriptor&) { return 4096L; }

    static auto GetTunableList()
    {
        return generate_tunable_list_conv_igemm_fwd_v4r4r4_xdlops_nhwc_kyxc_nhwk();
    }
};

} // namespace driver
} // namespace ck
#endif
