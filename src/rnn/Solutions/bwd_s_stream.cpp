#include <miopen/rnn/solvers.hpp>

namespace miopen {

namespace rnn_base {

void RNNModularSingleStreamBWD::ComputeBWD(Handle& handle,
                                           ConstData_t dy,
                                           ConstData_t dhy,
                                           Data_t dhx,
                                           ConstData_t cx,
                                           ConstData_t dcy,
                                           Data_t dcx,
                                           Data_t dx,
                                           ConstData_t w,
                                           Data_t workSpace,
                                           Data_t reserveSpace) const
{
    auto layer_i = rnnDesc.nLayers;

    if(layer_i == 0 || max_seq_len == 0)
        return;

    auto sequence_directions =
        rnnDesc.dirMode == miopenRNNDirectionMode_t::miopenRNNbidirection ? 2 : 1;

    rnnAlgoModules.PrepareWriteBuffers(handle, dhx, dcx, workSpace);

    rnnAlgoModules.PropDy(handle, dy, workSpace);

#if true
    do
    {
        layer_i--;

        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;

            auto ti = max_seq_len;
            do
            {
                const rnn_base::SequenceIterator cur_seq(--ti, seq_dir, max_seq_len, false);

                rnnAlgoModules.PropDhy(handle, dhy, workSpace, layer_i, cur_seq, seq_dir);

                rnnAlgoModules.UpdateHStatePerTimeSeq(
                    handle, dcy, cx, dcx, workSpace, reserveSpace, layer_i, cur_seq, seq_dir);

                // GEMM
                if(ti != 0)
                    rnnAlgoModules.PropHiddenDht(handle, w, workSpace, layer_i, cur_seq, seq_dir);
                else
                    rnnAlgoModules.PropDhxDcx(
                        handle, w, dhx, dcx, workSpace, reserveSpace, layer_i, cur_seq, seq_dir);

            } while(ti != 0);

            if(layer_i != 0)
                rnnAlgoModules.PropHiddenDy(handle, w, workSpace, reserveSpace, layer_i, seq_dir);
            else
                rnnAlgoModules.PropDx(handle, w, workSpace, dx, seq_dir);
        }

    } while(layer_i != 0);
#else
    // for debugging only, it uses the old order of functions
    // and allows logs comparison with the old code in aples to aples form
    do
    {
        layer_i--;
        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;
            auto ti            = max_seq_len;
            do
            {
                const rnn_base::SequenceIterator cur_seq(--ti, seq_dir, max_seq_len, false);
                if(ti == max_seq_len - 1)
                    rnnAlgoModules.PropDhy(handle, dhy, workSpace, layer_i, cur_seq, seq_dir);
                rnnAlgoModules.UpdateHStatePerTimeSeq(
                    handle, dcy, cx, dcx, workSpace, reserveSpace, layer_i, cur_seq, seq_dir);

                if(ti != 0)
                    rnnAlgoModules.PropDhy(
                        handle, dhy, workSpace, layer_i, cur_seq.getNext(), seq_dir);
                if(ti != 0)
                    rnnAlgoModules.PropHiddenDht(handle, w, workSpace, layer_i, cur_seq, seq_dir);
            } while(ti != 0);
        }
        for(int dir = 0; dir < sequence_directions; dir++)
        {
            const auto seq_dir = dir == 0 ? rnn_base::SequenceDirection::Forward
                                          : rnn_base::SequenceDirection::Reverse;
            for(int ti = 0; ti < max_seq_len; ti++)
            {
                const rnn_base::SequenceIterator cur_seq(ti, seq_dir, max_seq_len, false);
                rnnAlgoModules.PropDhxDcx(
                    handle, w, dhx, dcx, workSpace, reserveSpace, layer_i, cur_seq, seq_dir);
            }
            if(layer_i != 0)
                rnnAlgoModules.PropHiddenDy(handle, w, workSpace, reserveSpace, layer_i, seq_dir);
            else
                rnnAlgoModules.PropDx(handle, w, workSpace, dx, seq_dir);
        }
    } while(layer_i != 0);
#endif
}

} // namespace rnn_base
} // namespace miopen
