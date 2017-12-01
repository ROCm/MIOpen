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

#ifndef GUARD_MIOPEN_RNN_HPP_
#define GUARD_MIOPEN_RNN_HPP_

#include <miopen/miopen.h>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/common.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/mlo_internal.hpp>
#include <functional>
#include <numeric>
#include <map>

namespace miopen {

struct PerfField
{
    std::string name;
    float time;
    std::size_t workspace;

    bool operator<(const PerfField& p) const { return (time < p.time); }
};

template <class T>
struct c_array_view
{
    T* data;
    size_t n;

    using value_type =
        typename std::remove_cv<typename std::decay<decltype(deref(*data))>::type>::type;

    size_t size() const { return size; }

    const value_type& operator[](size_t i) const { return deref(data[i]); }

    value_type& operator[](size_t i) { return deref(data[i]); }
};

void profileRNNkernels(Handle& handle, unsigned char select, float& ctime);

struct RNNDescriptor : miopenRNNDescriptor
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
    std::size_t typeSize;

    size_t biasOffsetCalculation(const TensorDescriptor& xDesc, int layer, int biasID);

    size_t paramsOffsetCalculation(const TensorDescriptor& xDesc, int layer, int paramID);

    std::vector<int>
    pTensorLengthsCalculation(const TensorDescriptor& xDesc, int layer, int paramID);

    size_t GetWorkspaceSize(Handle& handle,
                            int seqLength,
                            c_array_view<miopenTensorDescriptor_t> xDesc) const;

    size_t GetReserveSize(Handle& handle,
                          int seqLength,
                          c_array_view<miopenTensorDescriptor_t> xDesc) const;

    size_t GetParamsSize(Handle& handle, const TensorDescriptor& xDesc, miopenDataType_t dtype);

    void GetParamsDescriptor(Handle& handle,
                             const TensorDescriptor& xDesc,
                             TensorDescriptor& wDesc,
                             miopenDataType_t dtype);

    std::size_t
    GetLayerParamSize(Handle& handle, int layer, const TensorDescriptor& xDesc, int paramID);

    std::size_t GetLayerBiasSize(Handle& handle, int layer, int biasID);

    void GetLayerParam(Handle& handle,
                       int layer,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       ConstData_t w,
                       int paramID,
                       TensorDescriptor& paramDesc,
                       Data_t param);

    void GetLayerBias(Handle& handle,
                      int layer,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      ConstData_t w,
                      int biasID,
                      TensorDescriptor& biasDesc,
                      Data_t bias);

    void SetLayerParam(Handle& handle,
                       int layer,
                       const TensorDescriptor& xDesc,
                       const TensorDescriptor& wDesc,
                       Data_t w,
                       int paramID,
                       const TensorDescriptor& paramDesc,
                       ConstData_t param);

    void SetLayerBias(Handle& handle,
                      int layer,
                      const TensorDescriptor& xDesc,
                      const TensorDescriptor& wDesc,
                      Data_t w,
                      int biasID,
                      const TensorDescriptor& biasDesc,
                      ConstData_t bias);

    size_t GetRNNInputSuperTensorSize(Handle& handle,
                                      int seqLength,
                                      c_array_view<miopenTensorDescriptor_t> xDesc);

    size_t GetRNNHiddenSuperTensorSize(Handle& handle,
                                       c_array_view<miopenTensorDescriptor_t> xDesc);

    void RNNForwardTraining(Handle& handle,
                            int seqLen,
                            c_array_view<miopenTensorDescriptor_t> xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                            const TensorDescriptor& cxDesc,
                            ConstData_t cx,
                            const TensorDescriptor& wDesc,
                            ConstData_t w,
                            c_array_view<miopenTensorDescriptor_t> yDesc,
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
                             c_array_view<miopenTensorDescriptor_t> xDesc,
                             ConstData_t x,
                             const TensorDescriptor& hxDesc,
                             ConstData_t hx,
                             const TensorDescriptor& cxDesc,
                             ConstData_t cx,
                             const TensorDescriptor& wDesc,
                             ConstData_t w,
                             c_array_view<miopenTensorDescriptor_t> yDesc,
                             Data_t y,
                             const TensorDescriptor& hyDesc,
                             Data_t hy,
                             const TensorDescriptor& cyDesc,
                             Data_t cy,
                             Data_t workSpace,
                             size_t workSpaceSize) const;

    void RNNBackwardData(Handle& handle,
                         int seqLen,
                         c_array_view<miopenTensorDescriptor_t> yDesc,
                         ConstData_t y,
                         c_array_view<miopenTensorDescriptor_t> dyDesc,
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
                         c_array_view<miopenTensorDescriptor_t> dxDesc,
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
                            int seqLen,
                            c_array_view<miopenTensorDescriptor_t> xDesc,
                            ConstData_t x,
                            const TensorDescriptor& hxDesc,
                            ConstData_t hx,
                            c_array_view<miopenTensorDescriptor_t> dyDesc,
                            ConstData_t dy,
                            const TensorDescriptor& dwDesc,
                            Data_t dw,
                            Data_t workSpace,
                            size_t workSpaceSize,
                            ConstData_t reserveSpace,
                            size_t reserveSpaceSize) const;

    inline bool isNotRNNskip() const { return inputMode != miopenRNNskip; }
    inline bool isRNNskip() const { return inputMode == miopenRNNskip; }
};

std::ostream& operator<<(std::ostream& stream, const RNNDescriptor& r);

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenRNNDescriptor, miopen::RNNDescriptor);

#endif // GUARD_MIOPEN_RNN_HPP_
