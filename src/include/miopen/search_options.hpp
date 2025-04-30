/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>

#include <miopen/common.hpp>
#include <miopen/find_controls.hpp>
#include <miopen/object.hpp>

#include <limits>
#include <unordered_map>
#include <optional>

namespace miopen {

struct FindOptions : miopenFindOptions
{
    struct Workspace
    {
        Data_t buffer;
        std::size_t size;
    };

    bool exhaustive_search                 = false;
    miopenFindResultsOrder_t results_order = miopenFindResultsOrderByTime;
    std::size_t workspace_limit            = std::numeric_limits<std::size_t>::max();
    std::unordered_map<miopenTensorArgumentId_t, Data_t> preallocated_tensors;
    std::optional<Workspace> preallocated_workspace;
    std::optional<FindEnforce> find_enforce;
    bool attach_binaries = false;
};

} // namespace miopen

inline std::ostream& operator<<(std::ostream& stream, const miopen::FindOptions& options)
{
    stream << "options(";
    stream << "exhaustive search: " << (options.exhaustive_search ? " true" : "false");
    stream << ", results order: ";
    switch(options.results_order)
    {
    case miopenFindResultsOrderByTime: stream << "by time"; break;
    case miopenFindResultsOrderByWorkspaceSize: stream << "by workspace size"; break;
    }
    stream << ", workspace limit: " << options.workspace_limit;
    stream << ")";
    return stream;
}

MIOPEN_DEFINE_OBJECT(miopenFindOptions, miopen::FindOptions);
