/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <memory>
#include <string>
#include <unordered_map>
#include <filesystem>

struct CacheData
{
    size_t hashcode;
    size_t kernelhash;
    int    split_k;
};

enum SolutionType
{
    ST_QUN_WRW = 0,
    ST_QUN_FWD = 1,
    ST_JIN_BWD = 2,
    ST_COUNT   = 3,

};

struct DirectCkMgr
{
    static DirectCkMgr* GetInst()
    {
        static std::unique_ptr<DirectCkMgr> instance = std::make_unique<DirectCkMgr>();
        return instance.get();
    }

    static std::unordered_map<size_t, CacheData> s_qun_wrw;
    std::filesystem::path path_qun_wrw;

    static std::unordered_map<size_t, CacheData> s_qun_fwd;
    std::filesystem::path path_qun_fwd;

    static std::unordered_map<size_t, CacheData> s_jin_bwd;
    std::filesystem::path path_jin_bwd;

    DirectCkMgr();
    size_t GetStringHash(std::string str);
    void FlushToCacheFile(std::unordered_map<size_t, CacheData>& input, const char* pPath);
    void Init();

    void ReadCacheFile(std::unordered_map<size_t, CacheData>& outputData, std::filesystem::path filePath);
    void AppendToCache(std::unordered_map<size_t, CacheData>& outputData, CacheData cd);

    bool FindCacheData(std::unordered_map<size_t, CacheData>& inputData, size_t hashcode, CacheData& cd);
    uint32_t launchCount[ST_COUNT] = {};
    uint32_t hitCacheCount[ST_COUNT] = {};
    ~DirectCkMgr();
    bool enableConvCache;
};
