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

#include <miopen/graphapi/tensor.hpp>
#include <miopen/graphapi/opgraph.hpp>

#include <functional>
#include <numeric>
#include <string_view>
#include <unordered_map>

namespace miopen {
namespace graphapi {

inline std::string tensorIdAsStr(const Tensor* tens_ptr)
{

    int64_t id = tens_ptr->getId();
    char* b    = reinterpret_cast<char*>(&id);

    return {b, sizeof(id)};
}

template <bool isVirtual, typename Vec>
Tensor makeTensor(std::string_view name, miopenDataType_t dt, const Vec& dims, const Vec& strides)
{
    int64_t id = 0;
    MIOPEN_THROW_IF(name.size() > sizeof(id), "tensor name exceeds 8 chars");
    std::copy_n(name.begin(), std::min(sizeof(id), name.size()), reinterpret_cast<char*>(&id));

    return TensorBuilder{}
        .setDataType(dt)
        .setDim(dims)
        .setStride(strides)
        .setId(id)
        .setVirtual(isVirtual)
        .build();
}

template <bool isVirtual, typename Vec>
Tensor makeTensor(std::string_view name, miopenDataType_t dt, const Vec& dims)
{
    TensorDescriptor desc{dt, dims};
    return makeTensor<isVirtual>(name, dt, desc.GetLengths(), desc.GetStrides());
}

/// An RAII style class that captures a pointer to an object on heap and frees it
/// upon destruction. It's different from std::unique_ptr in that it allows
/// capturing multiple types of pointers
struct HeapPtrDeleter
{
    using Fn = std::function<void()>;
    Fn mFn   = {};

    template <typename T>
    explicit HeapPtrDeleter(T* ptr)
        : mFn([ptr]() { delete ptr; }) // NOLINT (cppcoreguidelines-owning-memory)
    {
    }

    HeapPtrDeleter(const HeapPtrDeleter&) = delete;
    HeapPtrDeleter& operator=(const HeapPtrDeleter&) = delete;

    friend void swap(HeapPtrDeleter& left, HeapPtrDeleter& right) noexcept
    {
        std::swap(left.mFn, right.mFn);
    }

    HeapPtrDeleter(HeapPtrDeleter&& that) noexcept : mFn(std::move(that.mFn)) { that.mFn = {}; }

    HeapPtrDeleter& operator=(HeapPtrDeleter&& that) noexcept
    {
        if(this != &that)
        {
            HeapPtrDeleter tmp{std::move(that)};
            swap(*this, tmp);
        }
        return *this;
    }

    ~HeapPtrDeleter()
    {
        // default initialized std::function cannot be invoked
        if(mFn)
            mFn();
    }
};

/// an automatically deleting allocator that frees the allocated objects upon
/// destruction
struct AutoDeleteAllocator
{
    std::vector<HeapPtrDeleter> mPtrsToFree;

    AutoDeleteAllocator()                           = default;
    AutoDeleteAllocator(const AutoDeleteAllocator&) = delete;
    AutoDeleteAllocator& operator=(const AutoDeleteAllocator&) = delete;

    AutoDeleteAllocator(AutoDeleteAllocator&&) = default;
    AutoDeleteAllocator& operator=(AutoDeleteAllocator&&) = default;
    ~AutoDeleteAllocator()                                = default;

    template <typename T>
    T* allocate(T&& val)
    {
        T* ret = new T(std::forward<T>(val)); // NOLINT (cppcoreguidelines-owning-memory)
        mPtrsToFree.emplace_back(ret);
        return ret;
    }
};

struct PatternGraphGenerator
{

    struct DummyNode : public OpNode
    {
        std::string mName;
        std::vector<Tensor*> mInTensors;
        std::vector<Tensor*> mOutTensors;

        DummyNode(const std::string& name,
                  const std::vector<Tensor*>& ins,
                  const std::vector<Tensor*>& outs)
            : mName(name), mInTensors(ins), mOutTensors(outs)
        {
        }

        const std::string& signName() const final { return mName; }

        std::vector<Tensor*> getInTensors() const final { return mInTensors; }

        std::vector<Tensor*> getOutTensors() const final { return mOutTensors; }
    };

    struct DummyNodeGenSpec
    {
        std::string mName;
        std::vector<std::string> mInTensors;
        std::vector<std::string> mOutTensors;
    };

    inline Tensor* makeDummyTensor(std::string_view name)
    {

        return mAlloc.allocate(makeTensor<true>(name, miopenFloat, std::vector<size_t>({1})));
    }

private:
    AutoDeleteAllocator mAlloc{};
    OpGraph mGraph{};

    PatternGraphGenerator(const std::vector<DummyNodeGenSpec>& node_specs)
    {

        std::unordered_map<std::string, Tensor*> tensor_map;
        OpGraphBuilder builder;

        for(const auto& ns : node_specs)
        {
            std::vector<Tensor*> in_tensors;

            for(const auto& ti : ns.mInTensors)
            {
                auto [it, flag] = tensor_map.try_emplace(ti, makeDummyTensor(ti));
                in_tensors.emplace_back(it->second);
            }

            std::vector<Tensor*> out_tensors;
            for(const auto& to : ns.mOutTensors)
            {
                auto [it, flag] = tensor_map.try_emplace(to, makeDummyTensor(to));
                out_tensors.emplace_back(it->second);
            }

            builder.addNode(mAlloc.allocate(DummyNode{ns.mName, in_tensors, out_tensors}));
        }

        mGraph = std::move(builder).build();
    }

public:
    PatternGraphGenerator()                             = default;
    PatternGraphGenerator(const PatternGraphGenerator&) = delete;
    PatternGraphGenerator& operator=(const PatternGraphGenerator&) = delete;
    PatternGraphGenerator(PatternGraphGenerator&&)                 = default;
    PatternGraphGenerator& operator=(PatternGraphGenerator&&) = default;
    ~PatternGraphGenerator()                                  = default;

    static std::unique_ptr<PatternGraphGenerator>
    Make(const std::vector<DummyNodeGenSpec>& node_specs)
    {
        return std::unique_ptr<PatternGraphGenerator>(new PatternGraphGenerator(node_specs));
    }

    const auto& graph() const { return mGraph; }
};

/// \todo move this function out so that other find 2.0 code can use it
/// --amberhassaan May, 2024
inline std::string_view tensorEnumIdToStr(miopenTensorArgumentId_t id)
{

#define ENUM_CASE(k) \
    case k: return #k;

    switch(id)
    {
        ENUM_CASE(miopenTensorMhaK)
        ENUM_CASE(miopenTensorMhaQ)
        ENUM_CASE(miopenTensorMhaV)
        ENUM_CASE(miopenTensorMhaDescaleK)
        ENUM_CASE(miopenTensorMhaDescaleQ)
        ENUM_CASE(miopenTensorMhaDescaleV)
        ENUM_CASE(miopenTensorMhaDescaleS)
        ENUM_CASE(miopenTensorMhaScaleS)
        ENUM_CASE(miopenTensorMhaScaleO)
        ENUM_CASE(miopenTensorMhaDropoutProbability)
        ENUM_CASE(miopenTensorMhaDropoutSeed)
        ENUM_CASE(miopenTensorMhaDropoutOffset)
        ENUM_CASE(miopenTensorMhaO)
        ENUM_CASE(miopenTensorMhaAmaxO)
        ENUM_CASE(miopenTensorMhaAmaxS)
        ENUM_CASE(miopenTensorMhaM)
        ENUM_CASE(miopenTensorMhaZInv)
        ENUM_CASE(miopenTensorMhaDO)
        ENUM_CASE(miopenTensorMhaDescaleO)
        ENUM_CASE(miopenTensorMhaDescaleDO)
        ENUM_CASE(miopenTensorMhaDescaleDS)
        ENUM_CASE(miopenTensorMhaScaleDS)
        ENUM_CASE(miopenTensorMhaScaleDQ)
        ENUM_CASE(miopenTensorMhaScaleDK)
        ENUM_CASE(miopenTensorMhaScaleDV)
        ENUM_CASE(miopenTensorMhaDQ)
        ENUM_CASE(miopenTensorMhaDK)
        ENUM_CASE(miopenTensorMhaDV)
        ENUM_CASE(miopenTensorMhaAmaxDQ)
        ENUM_CASE(miopenTensorMhaAmaxDK)
        ENUM_CASE(miopenTensorMhaAmaxDV)
        ENUM_CASE(miopenTensorMhaAmaxDS)
    default: MIOPEN_THROW(miopenStatusInternalError, "unknown tensor enum id");
    }
#undef ENUM_CASE
}

} // end namespace graphapi
} // end namespace miopen
