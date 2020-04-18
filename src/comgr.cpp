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

#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <amd_comgr.h>
#include <cstddef>

namespace miopen {
namespace comgr {

#define EC(expr)                                                                     \
    do                                                                               \
    {                                                                                \
        status = (expr);                                                             \
        if((failed = (status != AMD_COMGR_STATUS_SUCCESS)))                          \
        {                                                                            \
            const char* reason;                                                      \
            if(AMD_COMGR_STATUS_SUCCESS != amd_comgr_status_string(status, &reason)) \
                reason = "UNKNOWN";                                                  \
            MIOPEN_LOG_E("\'" #expr "\' :" << std::string(reason));                  \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            MIOPEN_LOG_I("Ok \'" #expr "\'");                                        \
        }                                                                            \
    } while(false)

static bool PrintVersion()
{
    std::size_t major = 0;
    std::size_t minor = 0;
    amd_comgr_get_version(&major, &minor);
    MIOPEN_LOG_I("comgr v." << major << '.' << minor);
    return true;
}

static bool once = PrintVersion(); /// FIXME remove this

static amd_comgr_status_t DatasetAddData(const std::string& name,
                                         const std::string& content,
                                         const amd_comgr_data_kind_t type,
                                         const amd_comgr_data_set_t dataset)
{
    amd_comgr_data_t handle;
    amd_comgr_status_t status;
    bool failed = false;

    EC(amd_comgr_create_data(type, &handle));
    if(failed)
        return status;

    do
    {
        EC(amd_comgr_set_data(handle, content.size(), content.data()));
        if(failed)
            break;
        EC(amd_comgr_set_data_name(handle, name.c_str()));
        if(failed)
            break;
        EC(amd_comgr_data_set_add(dataset, handle));
        if(failed)
            break;

    } while(false);

    EC(amd_comgr_release_data(handle)); // Will be destroyed with data set.
    return status;
}

void BuildOcl(const std::string& name,
              const std::string& text,
              const std::string& options,
              std::string& binary)
{
    (void)options;
    (void)binary;

    amd_comgr_status_t status;
    bool failed = false;
    amd_comgr_data_set_t inputs;
    bool inputs_created = false;
    do
    {
        // Create input data set.
        EC(amd_comgr_create_data_set(&inputs));
        if(!(inputs_created = !failed))
            break;
        // Add source text.
        EC(DatasetAddData(name, text, AMD_COMGR_DATA_KIND_SOURCE, inputs));
        if(failed)
            break;

    } while(false);

    { // If cleanup fails, then do not throw, just issue error messages.
        const auto failed_save = failed;
        if(inputs_created)
            EC(amd_comgr_destroy_data_set(inputs));
        failed = failed_save;
    }

    if(failed)
    {
        MIOPEN_THROW("comgr::BuildOcl");
    }
}

} // namespace comgr
} // namespace miopen
