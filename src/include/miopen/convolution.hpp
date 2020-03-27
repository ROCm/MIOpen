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
#ifndef GUARD_MIOPEN_CONVOLUTION_HPP_
#define GUARD_MIOPEN_CONVOLUTION_HPP_

#include <miopen/common.hpp>
#include <miopen/kernel.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/names.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace miopen {

namespace solver {
struct ConvSolution;
} // namespace solver

struct ConvolutionContext;
struct Handle;
struct TensorDescriptor;
struct ConvolutionUserBuffers;
struct ProblemDescription;

using ExtraKernelArgs = std::tuple<int /*N*/,
                                   int /*C*/,
                                   int /*H*/,
                                   int /*W*/,
                                   int /*K*/,
                                   int /*n_groups*/,
                                   int /*out_H*/,
                                   int /*out_W*/>;

struct ConvFwdTensors;
struct ConvWrwTensors;

struct ConvolutionDescriptor : miopenConvolutionDescriptor
{
    ConvolutionDescriptor(std::size_t spatial_dim,
                          miopenConvolutionMode_t c_mode,
                          miopenPaddingMode_t p_mode,
                          const std::vector<int>& p_pads              = {0, 0},
                          const std::vector<int>& p_strides           = {1, 1},
                          const std::vector<int>& p_dilations         = {1, 1},
                          const std::vector<int>& p_trans_output_pads = {0, 0},
                          int p_group_count                           = 1,
                          float p_lowp_quant                          = float(1));

    ConvolutionDescriptor(const std::vector<int>& p_pads              = {0, 0},
                          const std::vector<int>& p_strides           = {1, 1},
                          const std::vector<int>& p_dilations         = {1, 1},
                          const std::vector<int>& p_trans_output_pads = {0, 0},
                          int p_group_count                           = 1,
                          float p_lowp_quant                          = float(1));

    std::size_t GetSpatialDimension() const;

    const std::vector<int>& GetConvPads() const;

    const std::vector<int>& GetConvStrides() const;

    const std::vector<int>& GetConvDilations() const;

    const std::vector<int>& GetTransposeConvPads() const;

    int GetGroupCount() const;

    TensorDescriptor GetForwardOutputTensor(const TensorDescriptor& xDesc,
                                            const TensorDescriptor& wDesc,
                                            miopenDataType_t yType = miopenFloat) const;

    std::size_t ForwardGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                            const TensorDescriptor& yDesc) const;

    std::size_t ForwardGetWorkSpaceSizeGEMMTranspose(const TensorDescriptor& xDesc,
                                                     const TensorDescriptor& yDesc) const;

    std::size_t ForwardGetWorkSpaceSizeGEMMStridedBatched(Handle& handle,
                                                          const TensorDescriptor& xDesc,
                                                          const TensorDescriptor& wDesc,
                                                          const TensorDescriptor& yDesc) const;

    std::size_t
    ForwardBackwardDataGetWorkSpaceSizeDirect(const miopen::ConvolutionContext& ctx) const;

    std::size_t
    ForwardBackwardGetWorkSpaceSizeImplicitGemm(const miopen::ConvolutionContext& ctx) const;
    std::size_t
    ForwardBackwardDataGetWorkSpaceSizeSCGemm(Handle& handle,
                                              const miopen::ConvolutionContext& ctx) const;

    std::size_t ForwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& yDesc) const;

    bool IsWinograd3x3SupportedAndFast(miopen::ConvolutionContext& ctx) const;

    std::size_t ForwardGetValidWorkSpaceSizeGemm(Handle& handle,
                                                 const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& xDesc,
                                                 const TensorDescriptor& yDesc) const;

    std::size_t BackwardGetValidWorkSpaceSizeGemm(const TensorDescriptor& dyDesc,
                                                  const TensorDescriptor& wDesc,
                                                  const TensorDescriptor& dxDesc) const;

    std::size_t WrwGetValidWorkSpaceSizeGemm(const TensorDescriptor& dyDesc,
                                             const TensorDescriptor& xDesc,
                                             const TensorDescriptor& dwDesc) const;

    std::size_t ForwardGetWorkSpaceSize(Handle& handle,
                                        const TensorDescriptor& wDesc,
                                        const TensorDescriptor& xDesc,
                                        const TensorDescriptor& yDesc) const;

    void FindConvFwdAlgorithm(Handle& handle,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              int requestAlgoCount,
                              int* returnedAlgoCount,
                              miopenConvAlgoPerf_t* perfResults,
                              Data_t workSpace,
                              std::size_t workSpaceSize,
                              bool exhaustiveSearch) const;

    int FindFwdFFTKernel(Handle& handle,
                         const TensorDescriptor& xDesc,
                         const TensorDescriptor& wDesc,
                         const TensorDescriptor& yDesc,
                         std::size_t workSpaceSize,
                         std::vector<KernelInvoke>& kernels,
                         const NetworkConfig& kcache_key) const;

    float ExecuteFwdFFTKernel(Handle& handle,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              Data_t workSpace,
                              std::size_t workSpaceSize,
                              const NetworkConfig& kcache_key,
                              bool timed = false) const;

    int FindBwdFFTKernel(Handle& handle,
                         const TensorDescriptor& dyDesc,
                         const TensorDescriptor& wDesc,
                         const TensorDescriptor& dxDesc,
                         std::size_t workSpaceSize,
                         std::vector<KernelInvoke>& kernels,
                         const NetworkConfig& kcache_key) const;

    float ExecuteBwdFFTKernel(Handle& handle,
                              const TensorDescriptor& dyDesc,
                              ConstData_t dy,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const TensorDescriptor& dxDesc,
                              Data_t dx,
                              Data_t workSpace,
                              std::size_t workSpaceSize,
                              const NetworkConfig& kcache_key,
                              bool timed = false) const;

    std::vector<miopen::solver::ConvSolution>
    FindDataDirectSolutions(Handle& handle,
                            const TensorDescriptor& xDesc,
                            const TensorDescriptor& wDesc,
                            const TensorDescriptor& yDesc,
                            bool exhaustiveSearch,
                            bool isForward,
                            const ConvolutionUserBuffers& bufs) const;

    std::vector<miopen::solver::ConvSolution>
    FindWinogradSolutions(const ConvolutionContext& ctx) const;

    std::vector<miopen::solver::ConvSolution>
    FindDataImplicitGemmSolutions(Handle& handle,
                                  const TensorDescriptor& xDesc,
                                  const TensorDescriptor& wDesc,
                                  const TensorDescriptor& yDesc,
                                  bool exhaustiveSearch,
                                  bool isForward,
                                  const ConvolutionUserBuffers& bufs) const;

    std::vector<miopen::solver::ConvSolution>
    FindSCGemmSolutions(Handle& handle,
                        const TensorDescriptor& xDesc,
                        const TensorDescriptor& wDesc,
                        const TensorDescriptor& yDesc,
                        bool exhaustiveSearch,
                        bool isForward,
                        const ConvolutionUserBuffers& bufs) const;

    void ConvolutionForward(Handle& handle,
                            const void* alpha,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            miopenConvFwdAlgorithm_t algo,
                            const void* beta,
                            const TensorDescriptor& yDesc,
                            Data_t y,
                            Data_t workSpace,
                            std::size_t workSpaceSize) const;

    std::size_t GetForwardSolutionCount(Handle& handle,
                                        const TensorDescriptor& wDesc,
                                        const TensorDescriptor& xDesc,
                                        const TensorDescriptor& yDesc) const;

    void GetForwardSolutions(Handle& handle,
                             const TensorDescriptor& wDesc,
                             const TensorDescriptor& xDesc,
                             const TensorDescriptor& yDesc,
                             size_t maxSolutionCount,
                             size_t* solutionCount,
                             miopenConvSolution_t* solutions) const;

    void CompileForwardSolution(Handle& handle,
                                const TensorDescriptor& wDesc,
                                const TensorDescriptor& xDesc,
                                const TensorDescriptor& yDesc,
                                solver::Id solver_id) const;

    std::size_t GetForwardSolutionWorkspaceSize(Handle& handle,
                                                const TensorDescriptor& wDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& yDesc,
                                                solver::Id solver_id) const;

    void ConvolutionForwardImmediate(Handle& handle,
                                     const TensorDescriptor& wDesc,
                                     ConstData_t w,
                                     const TensorDescriptor& xDesc,
                                     ConstData_t x,
                                     const TensorDescriptor& yDesc,
                                     Data_t y,
                                     Data_t workSpace,
                                     std::size_t workSpaceSize,
                                     solver::Id solver_id) const;

    std::size_t BackwardDataGetWorkSpaceSizeGEMM(const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& dyDesc) const;

    std::size_t BackwardDataGetWorkSpaceSizeGEMMTranspose(const TensorDescriptor& dyDesc,
                                                          const TensorDescriptor& dxDesc) const;

    std::size_t BackwardGetWorkSpaceSizeFFT(const TensorDescriptor& wDesc,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& dxDesc) const;

    std::size_t BackwardDataGetWorkSpaceSize(Handle& handle,
                                             const TensorDescriptor& wDesc,
                                             const TensorDescriptor& dyDesc,
                                             const TensorDescriptor& dxDesc) const;

    void FindConvBwdDataAlgorithm(Handle& handle,
                                  const TensorDescriptor& dyDesc,
                                  ConstData_t dy,
                                  const TensorDescriptor& wDesc,
                                  ConstData_t w,
                                  const TensorDescriptor& dxDesc,
                                  Data_t dx,
                                  int requestAlgoCount,
                                  int* returnedAlgoCount,
                                  miopenConvAlgoPerf_t* perfResults,
                                  Data_t workSpace,
                                  std::size_t workSpaceSize,
                                  bool exhaustiveSearch) const;

    void ConvolutionBackwardData(Handle& handle,
                                 const void* alpha,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& wDesc,
                                 ConstData_t w,
                                 miopenConvBwdDataAlgorithm_t algo,
                                 const void* beta,
                                 const TensorDescriptor& dxDesc,
                                 Data_t dx,
                                 Data_t workSpace,
                                 std::size_t workSpaceSize) const;

    std::size_t GetBackwardSolutionCount(Handle& handle,
                                         const TensorDescriptor& dyDesc,
                                         const TensorDescriptor& wDesc,
                                         const TensorDescriptor& dxDesc) const;

    void GetBackwardSolutions(Handle& handle,
                              const TensorDescriptor& dyDesc,
                              const TensorDescriptor& wDesc,
                              const TensorDescriptor& dxDesc,
                              size_t maxSolutionCount,
                              size_t* solutionCount,
                              miopenConvSolution_t* solutions) const;

    void CompileBackwardSolution(Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 const TensorDescriptor& wDesc,
                                 const TensorDescriptor& dxDesc,
                                 solver::Id solver_id) const;

    std::size_t GetBackwardSolutionWorkspaceSize(Handle& handle,
                                                 const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& wDesc,
                                                 const TensorDescriptor& dxDesc,
                                                 solver::Id solver_id) const;

    void ConvolutionBackwardImmediate(Handle& handle,
                                      const TensorDescriptor& dyDesc,
                                      ConstData_t dy,
                                      const TensorDescriptor& wDesc,
                                      ConstData_t w,
                                      const TensorDescriptor& dxDesc,
                                      Data_t dx,
                                      Data_t workSpace,
                                      std::size_t workSpaceSize,
                                      solver::Id solver_id) const;

    std::size_t GetWrwSolutionCount(Handle& handle,
                                    const TensorDescriptor& dyDesc,
                                    const TensorDescriptor& xDesc,
                                    const TensorDescriptor& dwDesc) const;

    void GetWrwSolutions(Handle& handle,
                         const TensorDescriptor& dyDesc,
                         const TensorDescriptor& xDesc,
                         const TensorDescriptor& dwDesc,
                         size_t maxSolutionCount,
                         size_t* solutionCount,
                         miopenConvSolution_t* solutions) const;

    void CompileWrwSolution(Handle& handle,
                            const TensorDescriptor& dyDesc,
                            const TensorDescriptor& xDesc,
                            const TensorDescriptor& dwDesc,
                            solver::Id solver_id) const;

    std::size_t GetWrwSolutionWorkspaceSize(Handle& handle,
                                            const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& dwDesc,
                                            solver::Id solver_id) const;

    void ConvolutionWrwImmediate(Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 ConstData_t dy,
                                 const TensorDescriptor& xDesc,
                                 ConstData_t x,
                                 const TensorDescriptor& dwDesc,
                                 Data_t dw,
                                 Data_t workSpace,
                                 std::size_t workSpaceSize,
                                 solver::Id solver_id) const;

    std::size_t BackwardWeightsGetWorkSpaceSize(Handle& handle,
                                                const TensorDescriptor& dyDesc,
                                                const TensorDescriptor& xDesc,
                                                const TensorDescriptor& dwDesc) const;

    std::size_t BackwardWeightsGetWorkSpaceSizeGEMM(const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& dwDesc) const;

    std::size_t BackwardWeightsGetWorkSpaceSizeDirect(const miopen::ConvolutionContext& ctx) const;
    std::size_t
    BackwardWeightsGetWorkSpaceSizeWinograd(const miopen::ConvolutionContext& ctx) const;
    std::size_t
    BackwardWeightsGetWorkSpaceSizeImplicitGemm(const miopen::ConvolutionContext& ctx) const;

    void FindConvBwdWeightsAlgorithm(Handle& handle,
                                     const TensorDescriptor& dyDesc,
                                     ConstData_t dy,
                                     const TensorDescriptor& xDesc,
                                     ConstData_t x,
                                     const TensorDescriptor& dwDesc,
                                     Data_t dw,
                                     int requestAlgoCount,
                                     int* returnedAlgoCount,
                                     miopenConvAlgoPerf_t* perfResults,
                                     Data_t workSpace,
                                     std::size_t workSpaceSize,
                                     bool exhaustiveSearch) const;

    void ConvolutionBackwardWeights(Handle& handle,
                                    const void* alpha,
                                    const TensorDescriptor& dyDesc,
                                    ConstData_t dy,
                                    const TensorDescriptor& xDesc,
                                    ConstData_t x,
                                    miopenConvBwdWeightsAlgorithm_t algo,
                                    const void* beta,
                                    const TensorDescriptor& dwDesc,
                                    Data_t dw,
                                    Data_t workSpace,
                                    std::size_t workSpaceSize) const;

    std::size_t spatialDim;
    miopenConvolutionMode_t mode;
    miopenPaddingMode_t paddingMode;
    std::vector<int> pads;
    std::vector<int> strides;
    std::vector<int> dilations;
    std::vector<int> trans_output_pads;
    int group_count;
    float lowp_quant; // quantization factor for low precision

    private:
    void ConvFwdGemm(Handle& handle,
                     const ConvFwdTensors& tensors,
                     Data_t workSpace,
                     std::size_t workSpaceSize) const;
    void ConvFwdFFT(Handle& handle,
                    const ConvFwdTensors& tensors,
                    Data_t workSpace,
                    std::size_t workSpaceSize,
                    const NetworkConfig& kcache_key) const;

    void ConvBwdGemm(Handle& handle,
                     const struct ConvBwdTensors& tensors,
                     Data_t workSpace,
                     std::size_t workSpaceSize) const;

    void ConvBwdFFT(Handle& handle,
                    const ConvBwdTensors& tensors,
                    Data_t workSpace,
                    size_t workSpaceSize,
                    const NetworkConfig& kcache_key) const;

    ProblemDescription MakeWrwProblem(const TensorDescriptor& dyDesc,
                                      const TensorDescriptor& xDesc,
                                      const TensorDescriptor& dwDesc) const;

    void BackwardWeightsGemm(Handle& handle,
                             const ConvWrwTensors& tensors,
                             Data_t workSpace,
                             std::size_t workSpaceSize) const;

    template <class TKernels>
    void BackwardWeightsDirect(Handle& handle,
                               const ConvolutionContext& ctx,
                               const ConvWrwTensors& tensors,
                               Data_t workSpace,
                               const TKernels& kernels) const;
    template <class TKernels>
    void BackwardWeightsWinograd(Handle& handle,
                                 const ConvolutionContext& ctx,
                                 const ConvWrwTensors& tensors,
                                 Data_t workSpace,
                                 const TKernels& kernels) const;

    std::size_t GetFwdSolutionWorkspaceSizeFallback(Handle& handle,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& yDesc,
                                                    solver::Id solver_id) const;

    std::size_t GetBwdSolutionWorkspaceSizeFallback(const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& wDesc,
                                                    const TensorDescriptor& dxDesc,
                                                    solver::Id solver_id) const;

    void GetForwardSolutionsFallback(Handle& handle,
                                     const TensorDescriptor& wDesc,
                                     const TensorDescriptor& xDesc,
                                     const TensorDescriptor& yDesc,
                                     size_t maxSolutionCount,
                                     size_t* solutionCount,
                                     miopenConvSolution_t* solutions) const;

    void GetBwdSolutionsFallback(Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 const TensorDescriptor& wDesc,
                                 const TensorDescriptor& dxDesc,
                                 size_t maxSolutionCount,
                                 size_t* solutionCount,
                                 miopenConvSolution_t* solutions) const;

    bool IsGemmApplicableFwd(const TensorDescriptor& wDesc,
                             const TensorDescriptor& xDesc,
                             const TensorDescriptor& yDesc) const;

    bool IsGemmApplicableBwd(const TensorDescriptor& dyDesc,
                             const TensorDescriptor& wDesc,
                             const TensorDescriptor& dxDesc) const;

    bool IsGemmApplicableWrw(const TensorDescriptor& dyDesc,
                             const TensorDescriptor& xDesc,
                             const TensorDescriptor& dwDesc) const;

    std::size_t GetFwdSolutionCountFallback(const TensorDescriptor& wDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& yDesc) const;

    std::size_t GetBwdSolutionCountFallback(const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& wDesc,
                                            const TensorDescriptor& dxDesc) const;

    std::size_t GetWrwSolutionCountFallback(const TensorDescriptor& dyDesc,
                                            const TensorDescriptor& xDesc,
                                            const TensorDescriptor& dwDesc) const;

    void GetWrwSolutionsFallback(Handle& handle,
                                 const TensorDescriptor& dyDesc,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& dwDesc,
                                 size_t maxSolutionCount,
                                 size_t* solutionCount,
                                 miopenConvSolution_t* solutions) const;

    std::size_t GetWrwSolutionWorkspaceSizeFallback(Handle& handle,
                                                    const TensorDescriptor& dyDesc,
                                                    const TensorDescriptor& xDesc,
                                                    const TensorDescriptor& dwDesc,
                                                    solver::Id solver_id) const;
};

void ConvolutionBackwardBias(Handle& handle,
                             const void* alpha,
                             const TensorDescriptor& dyDesc,
                             ConstData_t dy,
                             const void* beta,
                             const TensorDescriptor& dbDesc,
                             Data_t db);

std::ostream& operator<<(std::ostream& stream, const ConvolutionDescriptor& c);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenConvolutionDescriptor, miopen::ConvolutionDescriptor);

#endif // GUARD_MIOPEN_CONVOLUTION_HPP_
