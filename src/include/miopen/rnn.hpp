/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_RNN_HPP_
#define GUARD_MIOPEN_RNN_HPP_

#include <miopen/common.hpp>
#include <miopen/dropout.hpp>
#include <miopen/errors.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/tensor.hpp>
#include <miopen/seq_tensor.hpp>
#include <miopen/tensor_ops.hpp>

#include <cstddef>
#include <iosfwd>
#include <type_traits>
#include <vector>

namespace miopen {

struct Handle;
struct TensorDescriptor;

template <class T>
struct c_array_view
{
    T* data;
    size_t n;

    using value_type =
        typename std::remove_cv<typename std::decay<decltype(deref(*data))>::type>::type;

    size_t size() const { return n; }

    const value_type& operator[](size_t i) const { return deref(data[i]); }

    value_type& operator[](size_t i) { return deref(data[i]); }
};

void profileRNNkernels(const Handle& handle, unsigned char select, float& ctime);

struct MIOPEN_INTERNALS_EXPORT RNNDescriptor : miopenRNNDescriptor
{

    RNNDescriptor();
    RNNDescriptor(int hsz,
                  int layers,
                  miopenRNNMode_t rmode,
                  miopenRNNInputMode_t inMode,
                  miopenRNNDirectionMode_t bidir,
                  miopenRNNBiasMode_t bmode,
                  miopenRNNAlgo_t amode,
                  miopenDataType_t dType);

    RNNDescriptor(int hsz,
                  int layers,
                  miopenRNNMode_t rmode,
                  miopenRNNInputMode_t inMode,
                  miopenRNNDirectionMode_t bidir,
                  miopenRNNBiasMode_t bmode,
                  miopenRNNAlgo_t amode,
                  miopenDataType_t dType,
                  miopenDropoutDescriptor_t dropDesc);

    size_t hsize;   // DLOWELL: is this uniform over all layers?
    size_t nLayers; // This may be twice the number of actually wDesc layers since the layout for
                    // wDesc is 2-D?

    size_t nHiddenTensorsPerLayer; // TODO dlowell: set via constructor, or "set" functions
    size_t workspaceScale;

    miopenRNNMode_t rnnMode;
    miopenRNNDirectionMode_t dirMode;
    miopenRNNAlgo_t algoMode;
    miopenRNNInputMode_t inputMode;
    miopenRNNBiasMode_t biasMode;
    miopenDataType_t dataType;
    miopenRNNPaddingMode_t paddingMode = miopenRNNIONotPadded;

    std::size_t typeSize;
    miopenDropoutDescriptor_t dropoutDesc{};

    size_t biasOffsetCalculation(const TensorDescriptor& xDesc, int layer, int biasID) const;

    size_t paramsOffsetCalculation(const TensorDescriptor& xDesc, int layer, int paramID) const;

    std::vector<int>
    pTensorLengthsCalculation(const TensorDescriptor& xDesc, int layer, int paramID) const;

    static SeqTensorDescriptor makeSeqTensorDescriptor(miopenDataType_t t,
                                                       miopenRNNBaseLayout_t layout,
                                                       int maxSeqLength,
                                                       int batchSize,
                                                       int vectorSize,
                                                       const int* lensPerSeq,
                                                       const void* padding_marker_ptr);

    static SeqTensorDescriptor makeSeqTensorDescriptor(
        c_array_view<const miopenTensorDescriptor_t> descs,
        size_t seq_len,
        miopenRNNBaseLayout_t layout = miopenRNNBaseLayout_t::miopenRNNDataSeqMajorNotPadded);

    static void SeqTensorToTensorDescArray(const SeqTensorDescriptor& desc,
                                           std::vector<miopen::TensorDescriptor>& td,
                                           std::vector<miopenTensorDescriptor_t>& ptd);

    static miopenRNNBaseLayout_t getBaseLayoutFromDataTensor(const SeqTensorDescriptor& desc);
    static std::tuple<std::vector<unsigned int>, bool>
    convertRNNBaseLayout(miopenRNNBaseLayout_t layout);

    size_t GetMainSolWorkspaceSize(size_t batchLenSum,
                                   miopenRNNFWDMode_t fwdMode,
                                   miopenRNNBaseLayout_t ioLayout) const;

    size_t GetWorkspaceSize(Handle& handle,
                            int seqLength,
                            c_array_view<const miopenTensorDescriptor_t> xDesc) const;
    size_t GetWorkspaceSize(Handle& handle,
                            const SeqTensorDescriptor& xDesc,
                            miopenRNNFWDMode_t fwdMode) const;

    size_t GetReserveSize(size_t batchLenSum) const;
    size_t GetReserveSize(Handle& handle,
                          int seqLength,
                          c_array_view<const miopenTensorDescriptor_t> xDesc) const;

    size_t GetMaxWorkspaceSize(Handle& handle,
                               const SeqTensorDescriptor& xDesc,
                               miopenRNNFWDMode_t fwdMode) const;
    size_t GetMaxReserveSize(Handle& handle, const SeqTensorDescriptor& xDesc) const;

    size_t
    GetParamsSize(Handle& handle, const TensorDescriptor& xDesc, miopenDataType_t dtype) const;
    size_t GetParamsSize(size_t inputVector) const;

    void GetParamsDescriptor(Handle& handle,
                             const TensorDescriptor& xDesc,
                             TensorDescriptor& wDesc,
                             miopenDataType_t dtype) const;

    std::size_t
    GetLayerParamSize(Handle& handle, int layer, const TensorDescriptor& xDesc, int paramID) const;

    std::size_t GetLayerBiasSize(Handle& handle, int layer, int biasID) const;

    void GetLayerParam(const Handle& handle,
                       int layer,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       int paramID,
                       TensorDescriptor& paramDesc,
                       Data_t param) const;

    void GetLayerBias(const Handle& handle,
                      int layer,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      ConstData_t w,
                      int biasID,
                      TensorDescriptor& biasDesc,
                      Data_t bias) const;

    void SetLayerParam(const Handle& handle,
                       int layer,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       Data_t w,
                       int paramID,
                       const TensorDescriptor& paramDesc,
                       ConstData_t param) const;

    void SetLayerBias(const Handle& handle,
                      int layer,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      Data_t w,
                      int biasID,
                      const TensorDescriptor& biasDesc,
                      ConstData_t bias) const;

    void SetPaddingmode(miopenRNNPaddingMode_t padding);

    void GetLayerParamOffset(int layer,
                             const TensorDescriptor& xDesc,
                             int paramID,
                             TensorDescriptor& paramDesc,
                             size_t* paramOffset) const;

    void GetLayerBiasOffset(int layer,
                            const TensorDescriptor& xDesc,
                            int biasID,
                            TensorDescriptor& biasDesc,
                            size_t* biasOffset) const;

    size_t GetRNNInputSuperTensorSize(Handle& handle,
                                      int seqLength,
                                      c_array_view<miopenTensorDescriptor_t> xDesc) const;

    size_t GetRNNHiddenSuperTensorSize(Handle& handle,
                                       c_array_view<miopenTensorDescriptor_t> xDesc) const;

    void RNNForward(Handle& handle,
                    miopenRNNFWDMode_t fwdMode,
                    const SeqTensorDescriptor& xDesc,
                    ConstData_t x,
                    const TensorDescriptor& hDesc,
                    ConstData_t hx,
                    Data_t hy,
                    const TensorDescriptor& cDesc,
                    ConstData_t cx,
                    Data_t cy,
                    const SeqTensorDescriptor& yDesc,
                    Data_t y,
                    ConstData_t w,
                    size_t weightSpaceSize,
                    Data_t workSpace,
                    size_t workSpaceSize,
                    Data_t reserveSpace,
                    size_t reserveSpaceSize) const;

    void RNNForwardTraining(Handle& handle,
                            int seqLen,
                            c_array_view<const miopenTensorDescriptor_t> xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                            const TensorDescriptor& cxDesc,
                            ConstData_t cx,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            c_array_view<const miopenTensorDescriptor_t> yDesc,
                            Data_t y,
                            const TensorDescriptor& hyDesc,
                            Data_t hy,
                            const TensorDescriptor& cyDesc,
                            Data_t cy,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            Data_t reserveSpace,
                            size_t reserveSpaceSize) const;

    void RNNForwardInference(Handle& handle,
                             int seqLen,
                             c_array_view<const miopenTensorDescriptor_t> xDesc,
                             ConstData_t x,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& cxDesc,
                             ConstData_t cx,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             c_array_view<const miopenTensorDescriptor_t> yDesc,
                             Data_t y,
                             const TensorDescriptor& hyDesc,
                             Data_t hy,
                             const TensorDescriptor& cyDesc,
                             Data_t cy,
                             Data_t workSpace,
                             size_t workSpaceSize) const;

    void RNNBackwardData(Handle& handle,
                         const SeqTensorDescriptor& yDesc,
                         ConstData_t y,
                         ConstData_t dy,
                         const TensorDescriptor& hDesc,
                         ConstData_t hx,
                         ConstData_t dhy,
                         Data_t dhx,
                         const TensorDescriptor& cDesc,
                         ConstData_t cx,
                         ConstData_t dcy,
                         Data_t dcx,
                         const SeqTensorDescriptor& xDesc,
                         Data_t dx,
                         ConstData_t w,
                         size_t weightSpaceSize,
                         Data_t workSpace,
                         size_t workSpaceSize,
                         Data_t reserveSpace,
                         size_t reserveSpaceSize) const;

    void RNNBackwardData(Handle& handle,
                         int seqLen,
                         c_array_view<const miopenTensorDescriptor_t> yDesc,
                         ConstData_t y,
                         c_array_view<const miopenTensorDescriptor_t> dyDesc,
                         ConstData_t dy,
                         const TensorDescriptor& dhyDesc,
                         ConstData_t dhy,
                         const TensorDescriptor& dcyDesc,
                         ConstData_t dcy,
                         const TensorDescriptor& wDesc,
                         ConstData_t w,
                         const TensorDescriptor& hxDesc,
                         ConstData_t hx,
                         const TensorDescriptor& cxDesc,
                         ConstData_t cx,
                         c_array_view<const miopenTensorDescriptor_t> dxDesc,
                         Data_t dx,
                         const TensorDescriptor& dhxDesc,
                         Data_t dhx,
                         const TensorDescriptor& dcxDesc,
                         Data_t dcx,
                         Data_t workSpace,
                         size_t workSpaceSize,
                         Data_t reserveSpace,
                         size_t reserveSpaceSize) const;

    void RNNBackwardWeights(Handle& handle,
                            const SeqTensorDescriptor& xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hDesc,
                            ConstData_t hx,
                            const SeqTensorDescriptor& yDesc,
                            ConstData_t y,
                            Data_t dw,
                            size_t weightSpaceSize,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            ConstData_t reserveSpace,
                            size_t reserveSpaceSize) const;

    void RNNBackwardWeights(Handle& handle,
                            int seqLen,
                            c_array_view<const miopenTensorDescriptor_t> xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                            c_array_view<const miopenTensorDescriptor_t> dyDesc,
                            ConstData_t dy,
                            const TensorDescriptor& dwDesc,
                            Data_t dw,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            ConstData_t reserveSpace,
                            size_t reserveSpaceSize) const;

    inline bool isNotRNNskip() const { return inputMode != miopenRNNskip; }
    inline bool isRNNskip() const { return inputMode == miopenRNNskip; }

private:
    size_t RNNTransformerWorkspaceSize(const SeqTensorDescriptor& xDesc,
                                       miopenRNNFWDMode_t fwdMode) const;

    // TODO rename

    void ModularForward(Handle& handle,
                        miopenRNNFWDMode_t fwdMode,
                        ConstData_t w,
                        const SeqTensorDescriptor& xDesc,
                        ConstData_t x,
                        const TensorDescriptor& hDesc,
                        ConstData_t hx,
                        Data_t hy,
                        const TensorDescriptor& cDesc,
                        ConstData_t cx,
                        Data_t cy,
                        const SeqTensorDescriptor& yDesc,
                        Data_t y,
                        Data_t workSpace,
                        size_t workSpaceSize,
                        Data_t reserveSpace,
                        size_t reserveSpaceSize) const;

    void ModularBackward(Handle& handle,
                         const SeqTensorDescriptor& yDesc,
                         ConstData_t dy,
                         const TensorDescriptor& hDesc,
                         ConstData_t hx,
                         ConstData_t dhy,
                         Data_t dhx,
                         const TensorDescriptor& cDesc,
                         ConstData_t cx,
                         ConstData_t dcy,
                         Data_t dcx,
                         const SeqTensorDescriptor& xDesc,
                         Data_t dx,
                         ConstData_t w,
                         Data_t workSpace,
                         size_t workSpaceSize,
                         Data_t reserveSpace,
                         size_t reserveSpaceSize) const;

    void ModularBackwardWeights(Handle& handle,
                                const SeqTensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& hDesc,
                                ConstData_t hx,
                                const SeqTensorDescriptor& yDesc,
                                Data_t w,
                                Data_t workSpace,
                                size_t workSpaceSize,
                                ConstData_t reserveSpace,
                                size_t /*reserveSpaceSize*/) const;

    void RNNTransformerForward(Handle& handle,
                               miopenRNNFWDMode_t fwdMode,
                               ConstData_t w,
                               const SeqTensorDescriptor& xDesc,
                               ConstData_t x,
                               const TensorDescriptor& hDesc,
                               ConstData_t hx,
                               Data_t hy,
                               const TensorDescriptor& cDesc,
                               ConstData_t cx,
                               Data_t cy,
                               const SeqTensorDescriptor& yDesc,
                               Data_t y,
                               Data_t workSpace,
                               size_t workSpaceSize,
                               Data_t reserveSpace,
                               size_t reserveSpaceSize) const;

    void RNNTransformerBackwardData(Handle& handle,
                                    const SeqTensorDescriptor& yDesc,
                                    ConstData_t dy,
                                    const TensorDescriptor& hDesc,
                                    ConstData_t hx,
                                    ConstData_t dhy,
                                    Data_t dhx,
                                    const TensorDescriptor& cDesc,
                                    ConstData_t cx,
                                    ConstData_t dcy,
                                    Data_t dcx,
                                    const SeqTensorDescriptor& xDesc,
                                    Data_t dx,
                                    ConstData_t w,
                                    Data_t workSpace,
                                    size_t workSpaceSize,
                                    Data_t reserveSpace,
                                    size_t reserveSpaceSize) const;

    void RNNTransformerBackwardWeights(Handle& handle,
                                       const SeqTensorDescriptor& xDesc,
                                       ConstData_t x,
                                       const TensorDescriptor& hDesc,
                                       ConstData_t hx,
                                       const SeqTensorDescriptor& yDesc,
                                       Data_t dw,
                                       Data_t workSpace,
                                       size_t workSpaceSize,
                                       ConstData_t reserveSpace,
                                       size_t reserveSpaceSize) const;

    void RNNVanillaForward(Handle& handle,
                           miopenRNNFWDMode_t fwdMode,
                           ConstData_t w,
                           const SeqTensorDescriptor& xDesc,
                           ConstData_t x,
                           const TensorDescriptor& hDesc,
                           ConstData_t hx,
                           Data_t hy,
                           const TensorDescriptor& cDesc,
                           ConstData_t cx,
                           Data_t cy,
                           const SeqTensorDescriptor& yDesc,
                           Data_t y,
                           Data_t workSpace,
                           size_t workSpaceSize,
                           Data_t reserveSpace,
                           size_t reserveSpaceSize) const;

    void RNNVanillaBackwardData(Handle& handle,
                                const SeqTensorDescriptor& yDesc,
                                ConstData_t dy,
                                const TensorDescriptor& hDesc,
                                ConstData_t hx,
                                ConstData_t dhy,
                                Data_t dhx,
                                const TensorDescriptor& cDesc,
                                ConstData_t cx,
                                ConstData_t dcy,
                                Data_t dcx,
                                const SeqTensorDescriptor& xDesc,
                                Data_t dx,
                                ConstData_t w,
                                Data_t workSpace,
                                size_t workSpaceSize,
                                Data_t reserveSpace,
                                size_t reserveSpaceSize) const;

    void RNNVanillaBackwardWeights(Handle& handle,
                                   const SeqTensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& hDesc,
                                   ConstData_t hx,
                                   const SeqTensorDescriptor& yDesc,
                                   Data_t dw,
                                   Data_t workSpace,
                                   size_t workSpaceSize,
                                   ConstData_t reserveSpace,
                                   size_t reserveSpaceSize) const;

    void RNNForwardTrainingPackedTensors(Handle& handle,
                                         int seqLen,
                                         c_array_view<const miopenTensorDescriptor_t> xDesc,
                                         ConstData_t x,
                                         const TensorDescriptor& hxDesc,
                                         ConstData_t hx,
                                         const TensorDescriptor& cxDesc,
                                         ConstData_t cx,
                                         const TensorDescriptor& wDesc,
                                         ConstData_t w,
                                         c_array_view<const miopenTensorDescriptor_t> yDesc,
                                         Data_t y,
                                         const TensorDescriptor& hyDesc,
                                         Data_t hy,
                                         const TensorDescriptor& cyDesc,
                                         Data_t cy,
                                         Data_t reserveSpace,
                                         size_t reserveSpaceSize) const;

    void RNNForwardMS(Handle& handle,
                      std::vector<int>& seq_array,
                      const TensorDescriptor& xDesc,
                      ConstData_t x,
                      const TensorDescriptor& hxDesc,
                      ConstData_t hx,
                      ConstData_t cx,
                      const TensorDescriptor& wDesc,
                      ConstData_t w,
                      const TensorDescriptor& yDesc,
                      Data_t y,
                      Data_t hy,
                      Data_t cy,
                      Data_t extra_space,
                      size_t extra_space_size,
                      miopenRNNFWDMode_t fwd_mode) const;

    void RNNForwardInferencePacked(Handle& handle,
                                   int seqLen,
                                   c_array_view<const miopenTensorDescriptor_t> xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& hxDesc,
                                   ConstData_t hx,
                                   const TensorDescriptor& cxDesc,
                                   ConstData_t cx,
                                   const TensorDescriptor& wDesc,
                                   ConstData_t w,
                                   c_array_view<const miopenTensorDescriptor_t> yDesc,
                                   Data_t y,
                                   const TensorDescriptor& hyDesc,
                                   Data_t hy,
                                   const TensorDescriptor& cyDesc,
                                   Data_t cy,
                                   Data_t workSpace,
                                   size_t workSpaceSize) const;

    void RNNBackwardDataPackedTensors(Handle& handle,
                                      int seqLen,
                                      c_array_view<const miopenTensorDescriptor_t> dyDesc,
                                      ConstData_t dy,
                                      ConstData_t dhy,
                                      ConstData_t dcy,
                                      ConstData_t w,
                                      ConstData_t hx,
                                      ConstData_t cx,
                                      c_array_view<const miopenTensorDescriptor_t> dxDesc,
                                      Data_t dx,
                                      const TensorDescriptor& dhxDesc,
                                      Data_t dhx,
                                      const TensorDescriptor& dcxDesc,
                                      Data_t dcx,
                                      Data_t workSpace,
                                      size_t workSpaceSize,
                                      Data_t reserveSpace,
                                      size_t reserveSpaceSize) const;

    void RNNBackwardWeightsPackedTensors(Handle& handle,
                                         int seqLen,
                                         c_array_view<const miopenTensorDescriptor_t> xDesc,
                                         ConstData_t x,
                                         const TensorDescriptor& hxDesc,
                                         ConstData_t hx,
                                         c_array_view<const miopenTensorDescriptor_t> dyDesc,
                                         const TensorDescriptor& dwDesc,
                                         Data_t dw,
                                         Data_t workSpace,
                                         size_t workSpaceSize,
                                         ConstData_t reserveSpace,
                                         size_t reserveSpaceSize) const;
};

MIOPEN_INTERNALS_EXPORT std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& r);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
