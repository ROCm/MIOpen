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

#include <miopen/algorithm.hpp>
#include <miopen/errors.hpp>
#include <miopen/graphapi/graphapi.hpp>

#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

namespace miopen {

namespace graphapi {

namespace detail {

template <typename Range>
bool noRepetitions(const Range& r)
{
    std::set<std::remove_cv_t<std::remove_reference_t<decltype(*r.begin())>>> uniqueSet;
    bool isUnique = true;
    for(auto it = r.begin(), end = r.end(); isUnique && it != end; ++it)
    {
        std::tie(std::ignore, isUnique) = uniqueSet.insert(*it);
    }
    return isUnique;
}

} // namespace detail

class VariantPack
{
private:
    std::vector<int64_t> mTensorIds;
    std::vector<void*> mDataPointers;
    void* mWorkspace = nullptr;

public:
    VariantPack() noexcept              = default;
    VariantPack(const VariantPack&)     = default;
    VariantPack(VariantPack&&) noexcept = default;
    VariantPack& operator=(const VariantPack&) = default;
    VariantPack& operator=(VariantPack&&) noexcept = default;
    VariantPack(const std::vector<int64_t>& tensorIds,
                const std::vector<void*>& dataPointers,
                void* workspace)
        : mTensorIds(tensorIds), mDataPointers(dataPointers), mWorkspace(workspace)
    {
    }
    VariantPack(std::vector<int64_t>&& tensorIds,
                std::vector<void*>&& dataPointers,
                void* workspace)
        : mTensorIds(std::move(tensorIds)),
          mDataPointers(std::move(dataPointers)),
          mWorkspace(workspace)
    {
    }

    const auto& getTensorIds() const noexcept { return mTensorIds; }
    const auto& getDataPtrs() const noexcept { return mDataPointers; }

    void* getDataPointer(int64_t tensorId) const
    {
        assert(mTensorIds.size() == mDataPointers.size());
        auto iter = std::find(mTensorIds.cbegin(), mTensorIds.cend(), tensorId);
        MIOPEN_THROW_IF(iter == mTensorIds.cend(), "No such tensor id in VariantPack");
        return *(mDataPointers.cbegin() + (iter - mTensorIds.cbegin()));
    }
    void* getWorkspace() const noexcept { return mWorkspace; }

private:
    friend class VariantPackBuilder;
    friend class BackendVariantPackDescriptor;
};

class VariantPackBuilder
{
private:
    VariantPack mVariantPack;
    bool mTensorIdsSet    = false;
    bool mDataPointersSet = false;
    bool mWorkspaceSet    = false;

public:
    VariantPackBuilder& setTensorIds(const std::vector<int64_t>& tensorIds) &
    {
        if(!detail::noRepetitions(tensorIds))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        mVariantPack.mTensorIds = tensorIds;
        mTensorIdsSet           = true;
        return *this;
    }
    VariantPackBuilder& setTensorIds(std::vector<int64_t>&& tensorIds) &
    {
        if(!detail::noRepetitions(tensorIds))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        mVariantPack.mTensorIds = std::move(tensorIds);
        mTensorIdsSet           = true;
        return *this;
    }
    VariantPackBuilder& setDataPointers(const std::vector<void*>& dataPointers) &
    {
        if(miopen::any_of(dataPointers, [](const auto& v) { return v == nullptr; }) ||
           !detail::noRepetitions(dataPointers))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        mVariantPack.mDataPointers = dataPointers;
        mDataPointersSet           = true;
        return *this;
    }
    VariantPackBuilder& setDataPointers(std::vector<void*>&& dataPointers) &
    {
        if(miopen::any_of(dataPointers, [](const auto& v) { return v == nullptr; }) ||
           !detail::noRepetitions(dataPointers))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        mVariantPack.mDataPointers = std::move(dataPointers);
        mDataPointersSet           = true;
        return *this;
    }
    VariantPackBuilder& setWorkspace(void* workspace) &
    {
        mVariantPack.mWorkspace = workspace;
        mWorkspaceSet           = true;
        return *this;
    }

    VariantPackBuilder&& setTensorIds(const std::vector<int64_t>& tensorIds) &&
    {
        return std::move(setTensorIds(tensorIds));
    }
    VariantPackBuilder&& setTensorIds(std::vector<int64_t>&& tensorIds) &&
    {
        return std::move(setTensorIds(std::move(tensorIds)));
    }
    VariantPackBuilder&& setDataPointers(const std::vector<void*>& dataPointers) &&
    {
        return std::move(setDataPointers(dataPointers));
    }
    VariantPackBuilder&& setDataPointers(std::vector<void*>&& dataPointers) &&
    {
        return std::move(setDataPointers(std::move(dataPointers)));
    }
    VariantPackBuilder&& setWorkspace(void* workspace) &&
    {
        return std::move(setWorkspace(workspace));
    }

    VariantPack build() const&
    {
        if(!validate())
            MIOPEN_THROW(miopenStatusBadParm);
        return mVariantPack;
    }
    VariantPack build() &&
    {
        if(!validate())
            MIOPEN_THROW(miopenStatusBadParm);
        return std::move(mVariantPack);
    }

private:
    bool validate() const
    {
        return mTensorIdsSet && mDataPointersSet && mWorkspaceSet &&
               mVariantPack.mTensorIds.size() == mVariantPack.mDataPointers.size() &&
               std::find(mVariantPack.mDataPointers.cbegin(),
                         mVariantPack.mDataPointers.cend(),
                         mVariantPack.mWorkspace) == mVariantPack.mDataPointers.cend();
    }
};

class BackendVariantPackDescriptor : public BackendDescriptor
{
private:
    VariantPackBuilder mBuilder;
    VariantPack mVariantPack;

public:
    void virtual setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    void virtual finalize() override;
    void virtual getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;

    /// \todo return const ref and ref --amberhassaan May, 2024
    const VariantPack* getVariantPack() const { return &mVariantPack; }
    VariantPack* getVariantPack() { return &mVariantPack; }
};

} // namespace graphapi

} // namespace miopen
