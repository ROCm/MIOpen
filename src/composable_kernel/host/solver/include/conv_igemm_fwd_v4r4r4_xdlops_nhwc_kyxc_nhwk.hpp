#ifndef CONV_IGEMM_FWD_V4R4R4_XDLOPS_NHWC_KYXC_NHWK_HPP
#define CONV_IGEMM_FWD_V4R4R4_XDLOPS_NHWC_KYXC_NHWK_HPP

#include <numeric>
#include <array>
#include <vector>
#include <tuple>
#include "convolution_problem_descriptor.hpp"

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

    bool IsValid() const;
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

        // clang-format on
    };
}

// TODO make this common interface and write specs for it
struct ConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk
{
    static std::tuple<CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk, bool>
    CalculateCompileParameterBasedOnTunable(
        const ConvolutionProblemDescriptor& conv_problem_desc,
        const TunableConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& tunable);

    static auto GetDefaultCompileParameter(const ConvolutionProblemDescriptor& conv_problem_desc);

    static bool IsApplicable(const ConvolutionProblemDescriptor& conv_problem_desc);

    static bool IsValidCompileParameter(
        const ConvolutionProblemDescriptor& conv_problem_desc,
        const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param);

    static int
    GetBlockSize(const ConvolutionProblemDescriptor&,
                 const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param)
    {
        return compile_param.BlockSize;
    }

    static int
    GetGridSize(const ConvolutionProblemDescriptor& conv_problem_desc,
                const CompileParameterConvIgemmFwdV4r4r4XdlopsNhwcKyxcNhwk& compile_param);

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
