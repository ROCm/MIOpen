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

#include <cassert>
#include <unordered_map>

#include <miopen/graphapi/opgraph.hpp>

namespace miopen {
namespace graphapi {

class TensorInfo {
  miopenTensorArgumentId_t mEnumId = miopenTensorArgumentIdInvalid;
  Tensor* mGraphTensor = nullptr;
  TensorDescriptor mTensDesc{};
  Data_t mDevBuf = nullptr;

public:
  TensorInfo(miopenTensorArgumentId_t enum_id, Tensor* tens_ptr):
    mEnumId(enum_id),
    mGraphTensor(tens_ptr),
    mTensDesc(static_cast<TensorDescriptor>(*tens_ptr))
  {
    assert(tens_ptr);
    assert(mEnumId != miopenTensorArgumentIdInvalid;
  }

  void setDevBuf(Data_t ptr) {
    assert(ptr);
    mDevBuf = ptr;
  }

  const TensorDescriptor* tensDescPtr() const { returm mTensDesc; }

};


// int64_t is the graph tensor id
using TensorInfoMap = std::unordered_map<int64_t, TensorInfo>;

class Engine
{
    OpGraph* mGraph;
    miopenSolution_t mSolution;
    std::shared_ptr<TensorInfoMap> mTensorInfoMap;

    friend class EngineBuilder;

public:
    const OpGraph* getOpGraph() const { return mGraph; }
    OpGraph* getOpGraph() { return mGraph; }

    miopenSolution_t getSolution() const { return mSolution; }

    const TensorInfoMap* tensorInfoMap() const { return mTensorInfoMap.get(); }
};

class EngineBuilder
{

  Engine mEngine;
  bool mGraphSet = false;
  bool mSolutionSet = false;
  bool mTensorInfoMapSet = false;

public:

  EngineBuilder& setGraph(OpGraph* g) {
    assert(g);
    mEngine.mGraph = g;
    mGraphSet = true;
    return *this;
  }

  EngineBuilder& setSolution(miopenSolution_t s) {
    assert(s);
    mEngine.mSolution = s;
    mSolutionSet = true;
    return *this;
  }

  EngineBuilder& setTensorInfoMap(const std::shared_ptr<TensorInfoMap>& map) {
    mEngine.mTensorInfoMap = map;
    mTensorInfoMapSet = true;
    return *this;
  }

  Engine build() const {
    MIOPEN_THROW_IF(!mGraphSet || !mSolutionSet || !mTensorInfoMapSet, "must set graph and solution attributes");
    return mEngine;
  }
};

// Pattern is a family of solvers for the same graph shape
class GraphPattern
{

public:
    virtual bool matches(const OpGraph& graph)  const                       = 0;
    virtual std::vector<Engine> getEngines(const OpGraph& graph) const = 0;

    virtual ~GraphPattern();
};


std::vector<Engine> findEngines(const OpGraph&);

} // end namespace graphapi
} // end namespace miopen


