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
#include <miopen/graphapi/variant_pack.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace miopen {

namespace graphapi {

class MIOPEN_INTERNALS_EXPORT ExecutionPlan
{
private:
    /* we don't use a pointer for mEngineCfg
     * because we need to support serialization
     * and deserialization
     */
    EngineCfg mEngineCfg;
    miopenHandle_t mHandle = nullptr;
    std::vector<int64_t> mIntermediateIds;

    friend class ExecutionPlanBuilder;

public:
    ExecutionPlan()                     = default;
    ExecutionPlan(const ExecutionPlan&) = default;
    ExecutionPlan(ExecutionPlan&&)      = default;
    ExecutionPlan& operator=(const ExecutionPlan&) = default;
    ExecutionPlan& operator=(ExecutionPlan&&) = default;

    miopenHandle_t getHandle() const noexcept { return mHandle; }
    const EngineCfg& getEngineCfg() const noexcept { return mEngineCfg; }
    EngineCfg& getEngineCfg() noexcept { return mEngineCfg; }
    const std::vector<int64_t>& getIntermediateIds() const noexcept { return mIntermediateIds; }
    std::string getJsonRepresentation() const;

    void execute(miopenHandle_t handle, const VariantPack& variantPack)
    {
        checkPtr(handle);
        mEngineCfg.getEngine().getExecutor()->execute(handle, variantPack);
    }

    size_t getWorkspaceSize() const
    {
        return mEngineCfg.getEngine().getExecutor()->getWorkspaceSize();
    }
};

class MIOPEN_INTERNALS_EXPORT ExecutionPlanBuilder
{
private:
    ExecutionPlan mExecutionPlan;
    bool mEngineCfgSet = false;

public:
    ExecutionPlanBuilder& setHandle(miopenHandle_t handle) &;
    ExecutionPlanBuilder& setEngineCfg(const EngineCfg& engineCfg) &;
    ExecutionPlanBuilder& setEngineCfg(EngineCfg&& engineCfg) &;
    ExecutionPlanBuilder& setIntermediateIds(const std::vector<int64_t>& ids) &;
    ExecutionPlanBuilder& setIntermediateIds(std::vector<int64_t>&& ids) &;
    ExecutionPlanBuilder& setJsonRepresentation(const std::string_view& s) &;

    ExecutionPlanBuilder&& setHandle(miopenHandle_t handle) &&
    {
        return std::move(setHandle(handle));
    }
    ExecutionPlanBuilder&& setEngineCfg(const EngineCfg& engineCfg) &&
    {
        return std::move(setEngineCfg(engineCfg));
    }
    ExecutionPlanBuilder&& setEngineCfg(EngineCfg&& engineCfg) &&
    {
        return std::move(setEngineCfg(std::move(engineCfg)));
    }
    ExecutionPlanBuilder&& setIntermediateIds(const std::vector<int64_t>& ids) &&
    {
        return std::move(setIntermediateIds(ids));
    }
    ExecutionPlanBuilder&& setIntermediateIds(std::vector<int64_t>&& ids) &&
    {
        return std::move(setIntermediateIds(std::move(ids)));
    }
    ExecutionPlanBuilder&& setJsonRepresentation(const std::string_view& s) &&
    {
        return std::move(setJsonRepresentation(s));
    }

    ExecutionPlan build() &;
    ExecutionPlan build() &&;
};

class MIOPEN_INTERNALS_EXPORT BackendExecutionPlanDescriptor : public BackendDescriptor
{
private:
    ExecutionPlanBuilder mBuilder;
    ExecutionPlan mExecutionPlan;

    miopenBackendDescriptor_t mEngineCfgDescriptor = nullptr;

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
    void execute(miopenHandle_t handle, miopenBackendDescriptor_t variantPack) override;
};

} // namespace graphapi

} // namespace miopen
