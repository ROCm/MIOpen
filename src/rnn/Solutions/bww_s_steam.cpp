/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/rnn/solvers.hpp>

namespace miopen {

namespace rnn_base {

void RNNModularSingleStreamBWWeights::Compute(const Handle& handle,
                                              ConstData_t x,
                                              ConstData_t hx,
                                              Data_t dw,
                                              Data_t workSpace,
                                              size_t workSpaceSize,
                                              ConstData_t reserveSpace,
                                              size_t /*reserveSpaceSize*/) const
{

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    const ConstData_t back_data_space = workSpace;
    const auto back_data_byte_size =
        rnnAlgoModules.workspaceInfo.getBufferSizeImpl() * GetTypeSize(rnnDesc.dataType);

    const Data_t free_ws    = moveDataPtrByte(workSpace, back_data_byte_size);
    const auto free_ws_size = workSpaceSize - back_data_byte_size;

    rnnAlgoModules.PrepareWriteBuffers(handle, dw);

    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
    {
        if(layer_i == 0)
            rnnAlgoModules.PhisXInputWeights(handle, dw, back_data_space, x);
        else
            rnnAlgoModules.HiddenXInputWeights(handle, dw, back_data_space, reserveSpace, layer_i);

        rnnAlgoModules.BiasUpdate(handle, dw, back_data_space, free_ws, layer_i, free_ws_size);

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            rnnAlgoModules.PhisHStateWeights(
                handle, dw, back_data_space, hx, layer_i, max_seq_len, seq_dir);

            rnnAlgoModules.HiddenHStateWeights(
                handle, dw, back_data_space, reserveSpace, layer_i, max_seq_len, seq_dir);
        }
    }
}

void RNNDynamicModularSingleStreamBWWeights::Compute(const Handle& handle,
                                                     ConstData_t x,
                                                     ConstData_t hx,
                                                     Data_t dw,
                                                     Data_t workSpace,
                                                     size_t workSpaceSize,
                                                     ConstData_t reserveSpace,
                                                     size_t reserveSpaceSize) const
{

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    auto args_ext = rnnAlgoModules.createRuntimeArgsExt(createRuntimeArgsBase(
        handle, x, hx, dw, workSpace, workSpaceSize, reserveSpace, reserveSpaceSize));

    const auto back_data_space      = args_ext.backData;
    const auto free_work_space      = args_ext.freeWorkSpace;
    const auto free_work_space_size = args_ext.freeWorkSpaceSize;

    rnnAlgoModules.PrepareWriteBuffers(handle, dw);

    rnnAlgoModules.realXProp(handle, args_ext);

    auto real_seq_len = rnnAlgoModules.getRealTimeSeqSize();

    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
    {
        if(layer_i == 0)
            rnnAlgoModules.PhisXInputWeights(handle, dw, back_data_space, args_ext.tempX);
        else
            rnnAlgoModules.HiddenXInputWeights(handle, dw, back_data_space, reserveSpace, layer_i);

        rnnAlgoModules.BiasUpdate(
            handle, dw, back_data_space, free_work_space, layer_i, free_work_space_size);

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            rnnAlgoModules.PhisHStateWeights(
                handle, dw, back_data_space, hx, layer_i, real_seq_len, seq_dir);

            rnnAlgoModules.HiddenHStateWeights(
                handle, dw, back_data_space, reserveSpace, layer_i, real_seq_len, seq_dir);
        }
    }
}

} // namespace rnn_base
} // namespace miopen
