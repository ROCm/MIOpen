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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace miopen {

namespace graphapi {

class VariantPack
{
public:
    VariantPack() noexcept              = default;
    VariantPack(const VariantPack&)     = default;
    VariantPack(VariantPack&&) noexcept = default;
    VariantPack& operator=(const VariantPack&) = default;
    VariantPack& operator=(VariantPack&&) noexcept = default;
    VariantPack(const std::vector<int64_t>& tensorIds,
                const std::vector<void*>& dataPointers,
                void* workspace)
    {
    }
    VariantPack(std::vector<int64_t>&& tensorIds,
                std::vector<void*>&& dataPointers,
                void* workspace)
    {
    }

    void* getDataPointer(int64_t tensorId) const noexcept { return nullptr; }
    void* getWorkspace() const noexcept { return nullptr; }

private:
};

class VariantPackBuilder
{
public:
    VariantPackBuilder& setTensorIds(const std::vector<int64_t>& tensorIds) & { return *this; }
    VariantPackBuilder& setTensorIds(std::vector<int64_t>&& tensorIds) & { return *this; }
    VariantPackBuilder& setDataPointers(const std::vector<void*>& dataPointers) & { return *this; }
    VariantPackBuilder& setDataPointers(std::vector<void*>&& dataPointers) & { return *this; }
    VariantPackBuilder& setWorkspace(void* workspace) & { return *this; }

    VariantPackBuilder&& setTensorIds(const std::vector<int64_t>& tensorIds) &&
    {
        return std::move(*this);
    }
    VariantPackBuilder&& setTensorIds(std::vector<int64_t>&& tensorIds) &&
    {
        return std::move(*this);
    }
    VariantPackBuilder&& setDataPointers(const std::vector<void*>& dataPointers) &&
    {
        return std::move(*this);
    }
    VariantPackBuilder&& setDataPointers(std::vector<void*>&& dataPointers) &&
    {
        return std::move(*this);
    }
    VariantPackBuilder&& setWorkspace(void* workspace) && { return std::move(*this); }

    VariantPack build() const& { return {}; }
    VariantPack build() && { return {}; }

private:
};

} // namespace graphapi

} // namespace miopen
