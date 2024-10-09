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

#pragma once

#include <miopen/rnn.hpp>
#include <miopen/gemm_v2.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_MS_WA_FIX)

namespace miopen {

class MultiStreamController
{
public:
    static constexpr int rootStreamId = 0;

    MultiStreamController(const Handle& handle, int extra_stream_cnt)
        : streamPoolIdsMapping{init_stream_pool_ids(handle, extra_stream_cnt)},
          streamPoolCache{init_stream_pool(handle, streamPoolIdsMapping)},
          activeHandle{handle}
    {
    }

    hipStream_t getStream(int stream_id) const { return streamPoolCache[stream_id]; }

    void ChangeActiveStream(int stream_id) const
    {
        activeHandle.SetStreamFromPool(streamPoolIdsMapping[stream_id]);
    }

    hipError_t RecordEvent(const hipEvent_t event, int stream_id) const
    {
        return hipEventRecord(event, getStream(stream_id));
    }

    hipError_t SetWaitEvent(const hipEvent_t event, int stream_id) const
    {
        return hipStreamWaitEvent(getStream(stream_id), event, 0);
    }

    void RootWaitToAllStreams() const
    {
        for(size_t i = 0, stream_cnt = size(); i < stream_cnt; i++)
        {
            if(i != rootStreamId)
            {
                const miopen::HipEventPtr sync_event = make_hip_fast_event();
                RecordEvent(sync_event.get(), i);
                SetWaitEvent(sync_event.get(), rootStreamId);
            }
        }
    }

    void AllStreamsWaitRoot() const
    {
        const miopen::HipEventPtr main_event = make_hip_fast_event();

        RecordEvent(main_event.get(), rootStreamId);

        for(size_t i = 0, stream_cnt = size(); i < stream_cnt; i++)
        {
            if(i != rootStreamId)
                SetWaitEvent(main_event.get(), i);
        }
    };

    size_t size() const { return streamPoolIdsMapping.size(); }

    ~MultiStreamController() { ChangeActiveStream(rootStreamId); }

private:
    static std::vector<int> init_stream_pool_ids(const Handle& handle, int extra_stream_cnt)
    {
        std::vector<int> ids;
        ids.reserve(extra_stream_cnt + 1);

        bool ms_wa_fix_active = extra_stream_cnt > 2 && !env::disabled(MIOPEN_MS_WA_FIX);
        handle.SetStreamFromPool(0);
        int wa_steams = ms_wa_fix_active ? (handle.GetStream() == nullptr ? 3 : 2) : 0;

        handle.ReserveExtraStreamsInPool(extra_stream_cnt + wa_steams);

        for(int i = 0; i <= extra_stream_cnt + wa_steams; i++)
            if(!(ms_wa_fix_active && (i > 0 && i <= wa_steams)))
                ids.push_back(i);

        return ids;
    }

    static std::vector<hipStream_t> init_stream_pool(const Handle& handle,
                                                     const std::vector<int>& pool_ids)
    {
        std::vector<hipStream_t> pool;
        pool.reserve(pool_ids.size());

        for(auto id : pool_ids)
        {
            handle.SetStreamFromPool(id);
            pool.push_back(handle.GetStream());
        }
        handle.SetStreamFromPool(0);

        return pool;
    }

    const std::vector<int> streamPoolIdsMapping;
    const std::vector<hipStream_t> streamPoolCache;
    const Handle& activeHandle;
};

} // namespace miopen
