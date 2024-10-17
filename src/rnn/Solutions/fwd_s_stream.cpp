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

void RNNModularSingleStreamFWD::ComputeFWD(Handle& handle, const runtimeArgsFwd& runtimeArgs) const
{

    if(rnnDesc.nLayers == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    rnnAlgoModules.PrepareWriteBuffers(handle, runtimeArgs);

    // skip or linear
    // copy or gemm
    rnnAlgoModules.PropX(handle, runtimeArgs);

    rnnAlgoModules.AddBias(handle, runtimeArgs);

    for(auto layer_i = 0; layer_i < rnnDesc.nLayers; ++layer_i)
    {

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            if(layer_i != 0)
                rnnAlgoModules.PropHiddenY(handle, runtimeArgs, layer_i, seq_dir);

            for(int ti = 0; ti < max_seq_len; ti++)
            {
                const rnn_base::SequenceIterator cur_seq(ti, seq_dir, max_seq_len, true);

                if(ti == 0)
                    rnnAlgoModules.PropHxCx(handle, runtimeArgs, layer_i, cur_seq, seq_dir);
                else
                    rnnAlgoModules.PropHiddenHt(handle, runtimeArgs, layer_i, cur_seq, seq_dir);

                rnnAlgoModules.UpdateHStatePerTimeSeq(
                    handle, runtimeArgs, layer_i, cur_seq, seq_dir);

                rnnAlgoModules.PropHyCy(handle, runtimeArgs, layer_i, cur_seq, seq_dir);
            }
        }
    }

    rnnAlgoModules.PropY(handle, runtimeArgs);
}

} // namespace rnn_base
} // namespace miopen
