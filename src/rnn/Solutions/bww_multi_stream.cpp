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

#include <miopen/rnn/multi_stream_utils.hpp>

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_RNN_MS_STREAM_CNT)

namespace miopen {

namespace rnn_base {

void RNNModularMultiStreamBWWeights::PrologueDispatch(const runtimeArgsBww& args) const
{
    rnnAlgoModules.PrepareWriteBuffers(*args.handle, args.dw);
}

void RNNModularMultiStreamBWWeights::Compute(const Handle& handle,
                                             ConstData_t x,
                                             ConstData_t hx,
                                             Data_t dw,
                                             Data_t workSpace,
                                             size_t workSpaceSize,
                                             ConstData_t reserveSpace,
                                             [[maybe_unused]] size_t reserveSpaceSize) const
{

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    const runtimeArgsBww args{&handle, x, hx, dw, workSpace, reserveSpace};

    MultiStreamController ms_controller{handle, env::value_or(MIOPEN_RNN_MS_STREAM_CNT, 4)};

    PrologueDispatch(args);

    ms_controller.AllStreamsWaitRoot();

    ////////////////////////////////////////

    const auto [bias_stream, first_stream, stream_round] =
        [](const MultiStreamController& controller) {
            auto size = static_cast<int>(controller.size());

            const int first     = size > 1 ? 1 : 0;
            const int tmp_round = size > 1 ? size - first : 1;
            const int bias      = first + tmp_round - 1;
            const int round     = bias != first ? tmp_round - 1 : tmp_round;
            return std::make_tuple(bias, first, round);
        }(ms_controller);

    ms_controller.ChangeActiveStream(bias_stream);
    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
        rnnAlgoModules.BiasUpdate(handle, dw, workSpace, layer_i, workSpaceSize);

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    for(int layer_i = 0; layer_i < rnnDesc.nLayers; layer_i++)
    {
        const auto dispatch_stream_id = first_stream + (layer_i % stream_round);
        ms_controller.ChangeActiveStream(dispatch_stream_id);

        if(layer_i == 0)
            rnnAlgoModules.PhisXInputWeights(handle, dw, workSpace, x);
        else
            rnnAlgoModules.HiddenXInputWeights(handle, dw, workSpace, reserveSpace, layer_i);

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            rnnAlgoModules.PhisHStateWeights(
                handle, dw, workSpace, hx, layer_i, max_seq_len, seq_dir);

            rnnAlgoModules.HiddenHStateWeights(
                handle, dw, workSpace, reserveSpace, layer_i, max_seq_len, seq_dir);
        }
    }

    ms_controller.RootWaitToAllStreams();
}

} // namespace rnn_base
} // namespace miopen
