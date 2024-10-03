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

namespace {

class SpiralDispatch
{

    static std::vector<std::vector<miopen::HipEventPtr>> chunk_event_init(size_t layers_cnt,
                                                                          size_t chunks_cnt)
    {
        std::vector<std::vector<miopen::HipEventPtr>> chunk_end_events;
        chunk_end_events.resize(layers_cnt);
        for(int layer_id = 0; layer_id < layers_cnt; ++layer_id)
        {
            chunk_end_events[layer_id].resize(chunks_cnt);
            for(int chunk_id = 0; chunk_id < chunks_cnt; ++chunk_id)
                chunk_end_events[layer_id][chunk_id] = make_hip_fast_event();
        }
        return chunk_end_events;
    }

public:
    SpiralDispatch(const MultiStreamController& stream_controller,
                   size_t layers,
                   size_t seq_len,
                   size_t chunk_size,
                   size_t chunk_cnt)
        : msController(stream_controller),
          layerChunkEndEvent(chunk_event_init(layers, chunk_cnt)),
          layerTimeState(layers, 0),
          layerNewTimeState(layers, 0),
          maxSeqLen(seq_len),
          timeChunkSz(chunk_size),
          layersCnt(layers)
    {
    }

    const MultiStreamController& msController;
    const std::vector<std::vector<miopen::HipEventPtr>> layerChunkEndEvent;
    std::vector<size_t> layerTimeState;
    std::vector<size_t> layerNewTimeState;
    const size_t maxSeqLen;
    const size_t timeChunkSz;
    const size_t layersCnt;

    inline void PreDispatchSync(size_t layer_id, size_t chunk_id, int stream_id) const
    {
        if(chunk_id > 0)
        {
            msController.SetWaitEvent(layerChunkEndEvent[layer_id][chunk_id - 1].get(), stream_id);
        }
        if(layer_id > 0)
        {
            msController.SetWaitEvent(layerChunkEndEvent[layer_id - 1][chunk_id].get(), stream_id);
        }
    }

    inline void PostDispatchUpdate(size_t layer_id, size_t chunk_id, int stream_id)
    {
        layerNewTimeState[layer_id] += timeChunkSz;
        msController.RecordEvent(layerChunkEndEvent[layer_id][chunk_id].get(), stream_id);
    }

    inline bool IsDispatchable(size_t layer_id, size_t dispatch_chunk_time) const
    {
        return maxSeqLen <= dispatch_chunk_time ? false
               : layer_id == 0                  ? true
                                                : layerTimeState[layer_id - 1] >=
                                     std::min(dispatch_chunk_time + timeChunkSz, maxSeqLen);
    };

    inline void ApplyStateUpdate()
    {
        std::copy(layerNewTimeState.begin(), layerNewTimeState.end(), layerTimeState.begin());
    }

    template <typename Invoker>
    inline void
    DispatchNextChunk(Invoker& chunk_dispatcher, size_t layer_id, size_t chunk_id, int stream_id)
    {
        PreDispatchSync(layer_id, chunk_id, stream_id);

        msController.ChangeActiveStream(stream_id);
        chunk_dispatcher(timeChunkSz, chunk_id * timeChunkSz, layer_id);

        PostDispatchUpdate(layer_id, chunk_id, stream_id);
        msController.ChangeActiveStream(miopen::MultiStreamController::rootStreamId);
    }

    template <typename Invoker>
    inline bool TryDispatchNextChunk(Invoker& chunk_dispatcher, size_t layer_id, int stream_id)
    {
        auto chunk_id = layerNewTimeState[layer_id] / timeChunkSz;

        if(!IsDispatchable(layer_id, chunk_id * timeChunkSz))
            return false;

        DispatchNextChunk(chunk_dispatcher, layer_id, chunk_id, stream_id);

        return true;
    }

    template <typename Invoker>
    void Start(Invoker& chunk_dispatcher)
    {
        const auto [first_stream, stream_round] = [](const MultiStreamController& controller) {
            auto size       = static_cast<int>(controller.size());
            const int first = size > 1 ? 1 : 0;
            const int round = size > 1 ? size - first : 1;
            return std::make_tuple(first, round);
        }(msController);

        bool nothing_to_dispatch = false;

        while(!nothing_to_dispatch)
        {
            ApplyStateUpdate();
            nothing_to_dispatch = true;
            int stream_it       = 0;

            for(size_t disp_layer_id = 0; disp_layer_id < layersCnt; ++disp_layer_id)
            {
                const auto dispatch_stream_id = first_stream + stream_it;
                if(TryDispatchNextChunk(chunk_dispatcher, disp_layer_id, dispatch_stream_id))
                {
                    stream_it           = (stream_it + 1) % stream_round;
                    nothing_to_dispatch = false;
                }
            }
        }
    }
};

} // namespace

bool RNNModularMultiStreamBWD::ChunkDispatch(const runtimeArgsBwd& args,
                                             size_t chunk_size,
                                             size_t chunk_time_offset,
                                             size_t chunk_layer_offset) const
{
    constexpr auto seq_dir = rnn_base::SequenceDirection::Forward;
    const Handle& handle   = *args.handle;

    if(chunk_time_offset >= max_seq_len)
        return false;

    auto ti       = max_seq_len - chunk_time_offset;
    auto layer_id = rnnDesc.nLayers - chunk_layer_offset - 1;

    for(size_t i = 0, loop_size = chunk_size; i < loop_size && ti != 0; ++i)
    {
        const rnn_base::SequenceIterator cur_seq(--ti, seq_dir, max_seq_len, false);

        if(!cur_seq.isFirst())
        {
            rnnAlgoModules.PropHiddenDht(
                handle, args.w, args.workSpace, layer_id, cur_seq.getPrev(), seq_dir);
        }

        rnnAlgoModules.PropDhy(handle, args.dhy, args.workSpace, layer_id, cur_seq, seq_dir);

        rnnAlgoModules.UpdateHStatePerTimeSeq(handle,
                                              args.dcy,
                                              args.cx,
                                              args.dcx,
                                              args.workSpace,
                                              args.reserveSpace,
                                              layer_id,
                                              cur_seq,
                                              seq_dir);
        // GEMM

        if(cur_seq.isLast())
        {
            rnnAlgoModules.PropDhxDcx(handle,
                                      args.w,
                                      args.dhx,
                                      args.dcx,
                                      args.workSpace,
                                      args.reserveSpace,
                                      layer_id,
                                      cur_seq,
                                      seq_dir);
        }
    }

    ti = max_seq_len - chunk_time_offset;
    if(ti != 0)
    {
        auto first_val = ti - 1;
        const rnn_base::SequenceIterator start_seq(first_val, seq_dir, max_seq_len, false);

        auto last_val = chunk_size > (first_val) ? 0 : first_val - (chunk_size - 1);
        const rnn_base::SequenceIterator end_seq(last_val, seq_dir, max_seq_len, false);

        if(layer_id != 0)
        {
            rnnAlgoModules.PropHiddenDy(handle,
                                        args.w,
                                        args.workSpace,
                                        args.reserveSpace,
                                        layer_id,
                                        seq_dir,
                                        start_seq,
                                        end_seq);
        }
        else
        {
            rnnAlgoModules.PropDx(
                handle, args.w, args.workSpace, args.dx, seq_dir, start_seq, end_seq);
        }
    }

    return true;
}

void RNNModularMultiStreamBWD::PrologueDispatch(const runtimeArgsBwd& args) const
{
    rnnAlgoModules.PrepareWriteBuffers(*args.handle, args.dhx, args.dcx, args.workSpace);

    rnnAlgoModules.PropDy(*args.handle, args.dy, args.workSpace);
}

void RNNModularMultiStreamBWD::ComputeBWD(Handle& handle,
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
    const auto layers_cnt = rnnDesc.nLayers;

    if(layers_cnt == 0 || max_seq_len == 0)
        return;

    const runtimeArgsBwd args{&handle, dy, dhy, dhx, cx, dcy, dcx, dx, w, workSpace, reserveSpace};

    MultiStreamController ms_controller{handle, env::value_or(MIOPEN_RNN_MS_STREAM_CNT, 2)};

    constexpr size_t try_chunks_cnt = 16;
    const auto time_chunk_sz        = ((max_seq_len + try_chunks_cnt - 1) / try_chunks_cnt);
    const auto chunks_cnt           = (max_seq_len + time_chunk_sz - 1) / time_chunk_sz;

    SpiralDispatch dispatcher{ms_controller, layers_cnt, max_seq_len, time_chunk_sz, chunks_cnt};

    auto single_chunk_disputch =
        [&](size_t chunk_size, size_t chunk_time_offset, size_t chunk_layer_offset) {
            return ChunkDispatch(args, chunk_size, chunk_time_offset, chunk_layer_offset);
        };

    PrologueDispatch(args);

    ms_controller.AllStreamsWaitRoot();

    dispatcher.Start(single_chunk_disputch);

    ms_controller.RootWaitToAllStreams();
}

} // namespace rnn_base
} // namespace miopen
