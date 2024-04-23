/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/graphapi/opgraph.hpp>

#include <unordered_map>

namespace graphapi_opgraph_tests {

namespace gr = miopen::graphapi;

inline gr::Tensor makeDummyTensor(std::string_view name)
{
    int64_t id = 0;
    MIOPEN_THROW_IF(name.size() > sizeof(id), "tensor name exceeds 8 chars");
    std::copy_n(name.begin(), std::min(sizeof(id), name.size()), reinterpret_cast<char*>(&id));

    return gr::TensorBuilder{}
        .setDataType(miopenFloat)
        .setDim({1})
        .setStride({1})
        .setId(id)
        .setVirtual(true)
        .build();
}

struct DummyNode : public gr::OpNode
{
    std::string mName;
    std::vector<gr::Tensor*> mInTensors;
    std::vector<gr::Tensor*> mOutTensors;

    DummyNode(const std::string& name,
              const std::vector<gr::Tensor*>& ins,
              const std::vector<gr::Tensor*>& outs)
        : mName(name), mInTensors(ins), mOutTensors(outs)
    {
    }

    const std::string& signName() const final { return mName; }

    std::vector<gr::Tensor*> getInTensors() const final { return mInTensors; }

    std::vector<gr::Tensor*> getOutTensors() const final { return mOutTensors; }
};

struct DummyOpGraphGenerator
{

    struct DummyNodeGenSpec
    {
        std::string mName;
        std::vector<std::string> mInTensors;
        std::vector<std::string> mOutTensors;
    };

private:
    std::unordered_map<std::string, gr::Tensor> mTensorMap;
    std::vector<DummyNode> mNodes;
    gr::OpGraph mGraph;

    DummyOpGraphGenerator(const std::vector<DummyNodeGenSpec>& node_specs)
    {

        for(const auto& ns : node_specs)
        {
            std::vector<gr::Tensor*> in_tensors;

            for(const auto& ti : ns.mInTensors)
            {
                auto [it, flag] = mTensorMap.try_emplace(ti, makeDummyTensor(ti));
                in_tensors.emplace_back(&(it->second));
            }

            std::vector<gr::Tensor*> out_tensors;
            for(const auto& to : ns.mOutTensors)
            {
                auto [it, flag] = mTensorMap.try_emplace(to, makeDummyTensor(to));
                out_tensors.emplace_back(&(it->second));
            }

            mNodes.emplace_back(ns.mName, in_tensors, out_tensors);
        }

        gr::OpGraphBuilder builder;

        for(auto& n : mNodes)
        {
            builder.addNode(&n);
        }

        mGraph = std::move(builder).build();
    }

public:
    DummyOpGraphGenerator(const DummyOpGraphGenerator&) = delete;
    DummyOpGraphGenerator(DummyOpGraphGenerator&&)      = delete;
    DummyOpGraphGenerator& operator=(const DummyOpGraphGenerator&) = delete;
    DummyOpGraphGenerator& operator=(DummyOpGraphGenerator&&) = delete;
    ~DummyOpGraphGenerator()                                  = default;

    static std::unique_ptr<DummyOpGraphGenerator>
    Make(const std::vector<DummyNodeGenSpec>& node_specs)
    {
        return std::unique_ptr<DummyOpGraphGenerator>(new DummyOpGraphGenerator(node_specs));
    }

    const auto& graph() const { return mGraph; }
};

inline std::unique_ptr<DummyOpGraphGenerator> makeDiamondGraph()
{
    /*
     *       |
     *       | t_in
     *       v
     *      Top
     * t_a /   \ t_b
     *    /     \
     *   v       v
     *  Left    Right
     *    \      /
     * t_c \    / t_d
     *      v  v
     *     Bottom
     *       |
     *       |t_out
     *       v
     */

    return DummyOpGraphGenerator::Make({{"top", {"t_in"}, {"t_a", "t_b"}},
                                        {"left", {"t_a"}, {"t_c"}},
                                        {"right", {"t_b"}, {"t_d"}},
                                        {"bottom", {"t_c", "t_d"}, {"t_out"}}});
}

} // end namespace graphapi_opgraph_tests
