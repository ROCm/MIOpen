/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <env.hpp>
#include <handle.hpp>
#include <stringutils.hpp>
#include <target_properties.hpp>
#include <map>
#include <string>

OLC_DECLARE_ENV_VAR(OLC_DEBUG_ENFORCE_DEVICE)

namespace online_compile {

static std::string GetDeviceNameFromMap(const std::string& in)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::map<std::string, std::string> device_name_map = {
        {"Ellesmere", "gfx803"},
        {"Baffin", "gfx803"},
        {"RacerX", "gfx803"},
        {"Polaris10", "gfx803"},
        {"Polaris11", "gfx803"},
        {"Tonga", "gfx803"},
        {"Fiji", "gfx803"},
        {"gfx800", "gfx803"},
        {"gfx802", "gfx803"},
        {"gfx804", "gfx803"},
        {"Vega10", "gfx900"},
        {"gfx901", "gfx900"},
        {"10.3.0 Sienna_Cichlid 18", "gfx1030"},
    };

    const char* const p_asciz = online_compile::GetStringEnv(OLC_DEBUG_ENFORCE_DEVICE{});
    if(p_asciz != nullptr && strlen(p_asciz) > 0)
        return {p_asciz};

    const auto name = in.substr(0, in.find(':')); // str.substr(0, npos) returns str.

    auto match = device_name_map.find(name);
    if(match != device_name_map.end())
        return match->second;
    return name; // NOLINT (performance-no-automatic-move)
}

void TargetProperties::Init(const Handle* const handle)
{
    const auto rawName = [&]() -> std::string { return handle->GetDeviceNameImpl(); }();
    name               = GetDeviceNameFromMap(rawName);
    // DKMS driver older than 5.9 may report incorrect state of SRAMECC feature.
    // Therefore we compute default SRAMECC and rely on it for now.
    sramecc = [&]() -> boost::optional<bool> {
        if(name == "gfx906" || name == "gfx908")
            return {true};
        return {};
    }();
    // However we need to store the reported state, even if it is incorrect,
    // to use together with COMGR.
    sramecc_reported = [&]() -> boost::optional<bool> {
        if(rawName.find(":sramecc+") != std::string::npos)
            return true;
        if(rawName.find(":sramecc-") != std::string::npos)
            return false;
        return sramecc; // default
    }();
    xnack = [&]() -> boost::optional<bool> {
        if(rawName.find(":xnack+") != std::string::npos)
            return true;
        if(rawName.find(":xnack-") != std::string::npos)
            return false;
        return {}; // default
    }();
    InitDbId();
}

void TargetProperties::InitDbId()
{
    dbId = name;
    if(name == "gfx906" || name == "gfx908")
    {
        // Let's stay compatible with existing gfx906/908 databases.
        // When feature equal to the default (SRAMECC ON), do not
        // append feature suffix. This is for backward compatibility
        // with legacy databases ONLY!
        if(!sramecc || !(*sramecc))
            dbId += "_nosramecc";
    }
    else
    {
        if(sramecc && *sramecc)
            dbId += "_sramecc";
    }
    if(xnack && *xnack)
        dbId += "_xnack";
}

} // namespace online_compile
