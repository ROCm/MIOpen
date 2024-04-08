/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include "test.hpp"
#include "driver.hpp"
#include "get_handle.hpp"

#include <miopen/config.h>

#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/db.hpp>
#include <miopen/sqlite_db.hpp>
#include <miopen/find_db.hpp>

#include <miopen/filesystem.hpp>

namespace miopen {

struct EmbedSQLite : test_driver
{
    Handle handle{};
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    Allocator::ManageDataPtr x_dev;
    Allocator::ManageDataPtr w_dev;
    Allocator::ManageDataPtr y_dev;

    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {0, 0}, {2, 2}, {1, 1}};

    EmbedSQLite()
    {
        x = {128, 1024, 14, 14};
        w = {2048, 1024, 1, 1};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};
    }

    void run()
    {
        // create a context/problem decriptor
        const auto problem = miopen::conv::ProblemDescription{
            x.desc, w.desc, y.desc, filter, miopen::conv::Direction::Forward};
        miopen::ExecutionContext ctx{};
        ctx.SetStream(&handle);
        // Check PerfDb
        {
            // Get filename for the sys db
            // Check it in miopen_data()
            fs::path pdb_path(ctx.GetPerfDbPath());
            const auto& it_p =
                miopen_data().find(make_object_file_name(pdb_path.filename()).string());
            EXPECT(it_p != miopen_data().end());
            // find all the entries in perf db
            // Assert result is non-empty
            auto pdb = GetDb(ctx);
            EXPECT(pdb.FindRecord(problem));
        }
        // Check FindDb
        {
            // FindDb will throw if the file is not present
            FindDbRecord rec{handle, problem};
            EXPECT(!rec.empty());
        }
    }
};
} // namespace miopen
#endif

int main(int argc, const char* argv[])
{
#if MIOPEN_EMBED_DB
    test_drive<miopen::EmbedSQLite>(argc, argv);
#else
    (void)argc;
    (void)argv;
#endif
}
