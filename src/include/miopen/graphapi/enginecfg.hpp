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

#include <miopen/config.hpp>
#include <miopen/graphapi/engine.hpp>
#include <nlohmann/json.hpp>

namespace miopen {

namespace graphapi {

class EngineCfg
{
private:
    /* we don't use a pointer here to allow a user
     * to have several configs for an Engine. Each
     * config might modify its Engine in future so
     * their instance of Engine shouldn't be shared
     */
    Engine mEngine;

    friend class EngineCfgBuilder;

public:
    EngineCfg()                 = default;
    EngineCfg(const EngineCfg&) = default;
    EngineCfg(EngineCfg&&)      = default;
    EngineCfg& operator=(const EngineCfg&) = default;
    EngineCfg& operator=(EngineCfg&&) = default;

    EngineCfg(const Engine& engine) : mEngine(engine) {}
    EngineCfg(Engine&& engine) : mEngine(std::move(engine)) {}

    const Engine& getEngine() const noexcept { return mEngine; }
    Engine& getEngine() noexcept { return mEngine; }

    friend void to_json(nlohmann::json& json, const EngineCfg& engineCfg)
    {
        json = engineCfg.mEngine;
    }

    friend void from_json(const nlohmann::json& json, EngineCfg& engineCfg)
    {
        json.get_to(engineCfg.mEngine);
    }
};

/* For now we don't support tuning and a builder is not needed,
 * but in future it will be needed.
 */
class MIOPEN_INTERNALS_EXPORT EngineCfgBuilder
{
    EngineCfg mEngineCfg;
    bool mEngineSet = false;

public:
    EngineCfgBuilder& setEngine(const Engine& engine) &
    {
        mEngineCfg.mEngine = engine;
        mEngineSet         = true;
        return *this;
    }
    EngineCfgBuilder& setEngine(Engine&& engine) &
    {
        mEngineCfg.mEngine = std::move(engine);
        mEngineSet         = true;
        return *this;
    }
    EngineCfgBuilder&& setEngine(const Engine& engine) && { return std::move(setEngine(engine)); }
    EngineCfgBuilder&& setEngine(Engine&& engine) &&
    {
        return std::move(setEngine(std::move(engine)));
    }
    EngineCfg build() &;
    EngineCfg build() &&;
};

class MIOPEN_INTERNALS_EXPORT BackendEngineCfgDescriptor : public BackendDescriptor
{
protected:
    EngineCfgBuilder mBuilder;
    EngineCfg mEngineCfg;

    miopenBackendDescriptor_t mEngineDescriptor = nullptr;

    BackendEngineCfgDescriptor(const EngineCfg& engineCfg,
                               miopenBackendDescriptor_t engineDescriptor)
        : mEngineCfg(engineCfg), mEngineDescriptor(engineDescriptor)
    {
        mFinalized = true;
    }
    BackendEngineCfgDescriptor(EngineCfg&& engineCfg, miopenBackendDescriptor_t engineDescriptor)
        : mEngineCfg(std::move(engineCfg)), mEngineDescriptor(engineDescriptor)
    {
        mFinalized = true;
    }

public:
    BackendEngineCfgDescriptor() = default;
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

    const EngineCfg& getEngineCfg() const { return mEngineCfg; }
    EngineCfg& getEngineCfg() { return mEngineCfg; }
};

} // namespace graphapi

} // namespace miopen
