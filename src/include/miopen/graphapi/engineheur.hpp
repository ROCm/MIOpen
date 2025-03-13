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

#include <miopen/graphapi/enginecfg.hpp>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/opgraph.hpp>

#include <cstdint>
#include <vector>

namespace miopen {

namespace graphapi {

class EngineHeur
{
private:
    OpGraph* mOpGraph;
    std::vector<EngineCfg> mResults;
    int32_t mSmCount              = 0;
    miopenBackendHeurMode_t mMode = miopenBackendHeurMode_t(0);

    friend class EngineHeurBuilder;

public:
    EngineHeur() noexcept             = default;
    EngineHeur(const EngineHeur&)     = default;
    EngineHeur(EngineHeur&&) noexcept = default;
    EngineHeur& operator=(const EngineHeur&) = default;
    EngineHeur& operator=(EngineHeur&&) noexcept = default;

    OpGraph* getOpgraph() const noexcept { return mOpGraph; }
    miopenBackendHeurMode_t getMode() const noexcept { return mMode; }
    const std::vector<EngineCfg>& getResults() const noexcept { return mResults; }
    std::vector<EngineCfg>& getResults() noexcept { return mResults; }
    int32_t getSmCount() const noexcept { return mSmCount; }
};

class MIOPEN_INTERNALS_EXPORT EngineHeurBuilder
{
private:
    EngineHeur mEngineHeur;
    bool mModeSet = false;

public:
    EngineHeurBuilder& setOpGraph(OpGraph* opGraph);
    EngineHeurBuilder& setMode(miopenBackendHeurMode_t mode);
    EngineHeurBuilder& setSmCount(int32_t smCount);
    EngineHeur build();
};

class BackendEngineHeurDescriptor : public BackendDescriptor
{
private:
    EngineHeurBuilder mBuilder;
    EngineHeur mEngineHeur;

    miopenBackendDescriptor_t mOpGraphDescriptor = nullptr;

    class OwnedEngineCfgDescriptor : public BackendEngineCfgDescriptor
    {
        using Base = BackendEngineCfgDescriptor;
        BackendEngineDescriptor mOwnedEngineDescriptorInstance;

    public:
        OwnedEngineCfgDescriptor(const EngineCfg& engineCfg,
                                 miopenBackendDescriptor_t opGraphDescriptor)
            : Base(engineCfg, &mOwnedEngineDescriptorInstance),
              mOwnedEngineDescriptorInstance(Base::getEngineCfg().getEngine(), opGraphDescriptor)
        {
        }

        OwnedEngineCfgDescriptor(const OwnedEngineCfgDescriptor& other)     = default;
        OwnedEngineCfgDescriptor(OwnedEngineCfgDescriptor&& other) noexcept = default;
        ;
        OwnedEngineCfgDescriptor& operator=(const OwnedEngineCfgDescriptor& other) = default;
        OwnedEngineCfgDescriptor& operator=(OwnedEngineCfgDescriptor&& other) noexcept = default;
    };

    std::vector<OwnedEngineCfgDescriptor> mResults;

public:
    void setAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements) override;
    void finalize() override;
    void getAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) override;
};

} // namespace graphapi

} // namespace miopen
