/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define MIOPEN

#include "miopen/solver.hpp"
#include "miopen/db.hpp"
#include "miopen/mlo_utils.hpp"
#include "miopen/handle.hpp"

namespace miopen {
namespace solver {

static int mloReadDb(const std::string confreq_db_name, std::vector<std::string>& db)
{
    int ret = 0;
    mloFile f;

    ret = f.readBinaryFromFile(confreq_db_name.c_str());

    tokenize(f.source(), db, std::string("\n\r"));

    return (ret);
}

int mloReadConfigDB(Handle& stream, std::map<std::string, std::string>& conf_db)
{
    int ret = 0;
    std::string conf_file =
        miopen::GetDbPath(); // (_kernel_path == "") ? miopen::GetDbPath() : _kernel_path;

    conf_file += std::string("/") + stream.GetDeviceName() + "_" +
                 std::to_string(stream.GetMaxComputeUnits()) + "." + std::string("cd.pdb.txt");

    std::vector<std::string> db;
    mloReadDb(conf_file, db);

    // build searchable db

    std::vector<std::string>::iterator it;
    for(it = db.begin(); it != db.end(); ++it)
    {
        std::vector<std::string> v_key_val;
        tokenize((*it), v_key_val, std::string(" "));

        conf_db[v_key_val[0]] = v_key_val[1];
    }
    return (ret);
}

static int mloBuildConf_Key(const ConvolutionContext& params, std::string& conf_key)
{
    conf_key = std::to_string(static_cast<long long>(params.n_inputs)) + std::string("x") +
               std::to_string(static_cast<long long>(params.in_height)) + std::string("x") +
               std::to_string(static_cast<long long>(params.in_width)) + std::string("x") +
               std::to_string(static_cast<long long>(params.kernel_size1)) + std::string("x") +
               std::to_string(static_cast<long long>(params.kernel_size0)) + std::string("x") +
               std::to_string(static_cast<long long>(params.n_outputs)) + std::string("x") +
               std::to_string(static_cast<long long>(params.out_height)) + std::string("x") +
               std::to_string(static_cast<long long>(params.out_width)) + std::string("x") +
               std::to_string(static_cast<long long>(params.batch_sz)) + std::string("x") +
               params.in_layout + std::string("x") + params.in_data_type + std::string("x") +
               std::to_string(static_cast<long long>(params.forward));
    return (0);
}

static bool mloSearchConfigDB(std::map<std::string, std::string>& conf_db,
                              std::string& conf_key,
                              std::string& conf_val,
                              std::map<std::string, std::string>::iterator& it)
{

    bool found = false;

    it = conf_db.find(conf_key);
    if(it != conf_db.end())
    {
        found    = true;
        conf_val = (*it).second;

        //			std::cout << "Found : " << conf_val << std::endl;
    }
    return (found);
}

static bool mloSearchConfigInDB(const ConvolutionContext& params,
                                std::string& conf_key,
                                std::string& conf_val)
{
    bool known_config = false;
    // build searchable db
    std::map<std::string, std::string> conf_db;

    mloReadConfigDB(params.GetStream(), conf_db);

    mloBuildConf_Key(params, conf_key);

    std::map<std::string, std::string>::iterator m_it;
    known_config = mloSearchConfigDB(conf_db, conf_key, conf_val, m_it);

    return (known_config);
}

static int mloParseConf(const std::string& conf_val,
                        int& grp_tile1,
                        int& grp_tile0,
                        int& in_tile1,
                        int& in_tile0,
                        int& out_pix_tile1,
                        int& out_pix_tile0,
                        int& n_out_pix_tiles,
                        int& n_in_data_tiles,
                        int& n_stacks)
{
    std::vector<std::string> conf_val_vec;
    tokenize(conf_val, conf_val_vec, std::string("."));
    grp_tile1       = std::stoi(conf_val_vec[0]);
    grp_tile0       = std::stoi(conf_val_vec[1]);
    in_tile1        = std::stoi(conf_val_vec[2]);
    in_tile0        = std::stoi(conf_val_vec[3]);
    out_pix_tile1   = std::stoi(conf_val_vec[4]);
    out_pix_tile0   = std::stoi(conf_val_vec[5]);
    n_out_pix_tiles = std::stoi(conf_val_vec[6]);
    n_in_data_tiles = std::stoi(conf_val_vec[7]);
    n_stacks        = std::stoi(conf_val_vec[8]);
    return (0);
}

/*
the search db is a text file with the name defined by the device characteristics.
each line is a key/value pair, separated by a space:
32x16x16x3x3x64x16x16x100xNCHWxFP32x1 16.16.16.16.1.4.8.4.1
or
64x8x8x5x5x32x8x8x100xNCHWxFP32x0 16.16.8.8.2.4.1.1.4

key format (all values are separted by x):
n input maps
input height
input width
filter height
filter width
n output maps
output height
output width
batch size
tensors' layout
tensprs' data type
direction (1 - forward, 0 - backward)

Note:
for backward direction - input and output are reversed.

value format (all values are separated by .):
vertical group size
horizontal group size
input block vertical size
input block horizontal size
output tile vertical size
output tile horizaontal size
n of output tiles
n of input blocks
n batchs (stacks) processed by the group
*/

static int mloSetConf(const std::string& conf_val, ConvOclDirectFwdLegacyExhaustiveSearch::PerformanceConfigImpl& result)
{
    mloParseConf(conf_val,
                 result.grp_tile1,
                 result.grp_tile0,
                 result.in_tile1,
                 result.in_tile0,
                 result.out_pix_tile1,
                 result.out_pix_tile0,
                 result.n_out_pix_tiles,
                 result.n_in_data_tiles,
                 result.n_stacks);

    return (0);
}

static int mloBuildConf_Val(std::string& conf_val,
                            int grp_tile1,
                            int grp_tile0,
                            int in_tile1,
                            int in_tile0,
                            int out_pix_tile1,
                            int out_pix_tile0,
                            int n_out_pix_tiles,
                            int n_in_data_tiles,
                            int n_stacks)
{
    conf_val = std::to_string(static_cast<long long>(grp_tile1)) + std::string(".") +
               std::to_string(static_cast<long long>(grp_tile0)) + std::string(".") +
               std::to_string(static_cast<long long>(in_tile1)) + std::string(".") +
               std::to_string(static_cast<long long>(in_tile0)) + std::string(".") +
               std::to_string(static_cast<long long>(out_pix_tile1)) + std::string(".") +
               std::to_string(static_cast<long long>(out_pix_tile0)) + std::string(".") +
               std::to_string(static_cast<long long>(n_out_pix_tiles)) + std::string(".") +
               std::to_string(static_cast<long long>(n_in_data_tiles)) + std::string(".") +
               std::to_string(static_cast<long long>(n_stacks));
    return (0);
}

/*
* select defult configuration if a known configuration has not been found.
*/
static int mloSelectDefaultConfig(std::string& conf_val,
                                  const ConvolutionContext& params,
                                  ConvOclDirectFwdLegacyExhaustiveSearch::PerformanceConfigImpl& result)
{

    //
    result.in_tile0 = (params.in_width <= 8) ? 8 : (params.in_width <= 16)
                                                       ? 16
                                                       : 32; // size of input data per ALU plane
    result.in_tile1 = (params.in_height <= 8) ? 8 : (params.in_height <= 16)
                                                        ? 16
                                                        : 8; // size of input data per ALU plane

    result.grp_tile0 = (result.in_tile0 == 8) ? 8 : 16;
    result.grp_tile1 = (result.in_tile1 == 8) ? 8 : 16;

    result.out_pix_tile0 = 2; // size of ouptput tile per wk-item (ALU))
    result.out_pix_tile1 = 2; //

    result.n_out_pix_tiles = 8; // # output pixel tiles per wk-item (ALU)
    result.n_in_data_tiles = 2; // # of blocks of different inputs in LDS

    result.n_stacks = 1; // # of diff stacks (part of batch).

    if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
    {

        if((params.n_outputs / 16) * 16 == params.n_outputs &&
           (params.n_inputs / 4) * 4 == params.n_inputs)
        {
            // version
            if(params.forward && (params.n_inputs / 8) * 8 == params.n_inputs)
            {
                result.n_in_data_tiles = 128;

                result.n_out_pix_tiles = 32;
                // 0 or 1
                // CHEAT_SHADER_COMPILER =
                result.out_pix_tile0 = 0;

                // int version =
                result.out_pix_tile1 = 1;
            }
            else
            {
                int i_sz             = params.out_height * params.out_width;
                result.out_pix_tile0 = (i_sz & 1) ? 1 : 2;

                if(params.pad0 > 0 || params.kernel_stride0 > 1)
                {
                    if(params.forward)
                    {
                        result.out_pix_tile0 = (params.out_width & 1) ? 1 : 2;
                    }
                    else
                    {
                        result.out_pix_tile0 =
                            ((params.out_width & 1) || (params.in_width & 1)) ? 1 : 2;
                    }
                }

                result.n_out_pix_tiles = 16;
                result.n_in_data_tiles = 4;
                result.grp_tile0       = 64;
                // int version =
                result.out_pix_tile1 = 0;
            }
        }
        else
        {
            result.in_tile0 = 4; // size of input data per ALU plane
            result.in_tile1 = 1; // size of input data per ALU plane

            int out_len4 = (params.out_height * params.out_width + 3) / 4;

            result.grp_tile0 =
                (out_len4 > 192) ? 256 : (out_len4 > 128) ? 192 : (out_len4 > 64) ? 128 : 64;
            result.grp_tile1 = 1;

            result.out_pix_tile0 = 4; // size of ouptput tile per wk-item (ALU))
            result.out_pix_tile1 = 1; // 4; //

            result.n_out_pix_tiles = 16; // 2;  // # output pixel tiles per wk-item (ALU)
            result.n_in_data_tiles = 2;  // 4; // # of blocks of different inputs in LDS

            result.n_stacks = (params.batch_sz > 1) ? 2 : 1; // # of diff stacks (part of batch).
            result.n_stacks = (params.batch_sz > 1) ? 2 : 1; // # of diff stacks (part of batch).
        }
    }

    mloBuildConf_Val(conf_val,
                     result.grp_tile1,
                     result.grp_tile0,
                     result.in_tile1,
                     result.in_tile0,
                     result.out_pix_tile1,
                     result.out_pix_tile0,
                     result.n_out_pix_tiles,
                     result.n_in_data_tiles,
                     result.n_stacks);

    return (0);
}

static bool mloFindConfigReq(const std::string confreq_db_name,
                             const std::string& conf_key,
                             std::vector<std::string>& req_conf_db,
                             std::vector<std::string>::iterator& it)
{
    bool ret = true;

    mloReadDb(confreq_db_name, req_conf_db);

    // find req string
    ret = false;
    for(it = req_conf_db.begin(); it != req_conf_db.end(); ++it)
    {
        if(*it == conf_key)
        {
            ret = true;
            break;
        }
    }
    return (ret);
}

static int mloUpdateDb(const std::string& file_nm, const std::vector<std::string>& db)
{
    mloFile f;
    // serialize
    std::string serial;
    std::vector<std::string>::const_iterator it;
    for(it = db.begin(); it != db.end(); ++it)
    {
        serial += (*it) + "\n";
    }

    int ret = f.writeBinaryToFile(file_nm.c_str(), serial.c_str(), serial.length());

    return (ret);
}

/*
* request cofiguraion db management
* request configuration db is a text file
* each line is a key (in cofig db format) that has not been found in teh configuratio db
*/

static int mloAddConfigReq(Handle& stream, const std::string& conf_key)
{
    int ret = 0;
    std::vector<std::string> req_conf_db;
    std::string conf_file =
        miopen::GetDbPath(); // (_kernel_path == "") ? miopen::GetDbPath() : _kernel_path;

    conf_file += std::string("/") + stream.GetDeviceName() + "_" +
                 std::to_string(stream.GetMaxComputeUnits()) + "." + std::string("cd.rdb.txt");
#ifdef MIOPEN_LOG_CONVOLUTION
    printf("file %s\n", conf_file.c_str());
#endif
    std::vector<std::string>::iterator it;
    bool found = mloFindConfigReq(conf_file, conf_key, req_conf_db, it);

    if(!found)
    {
        req_conf_db.push_back(conf_key);
        ret = mloUpdateDb(conf_file, req_conf_db);
    }
    return (ret);
}

/*
* return a known or default configuration
*/
static bool mloGetConfig(const ConvolutionContext& params,
                         ConvOclDirectFwdLegacyExhaustiveSearch::PerformanceConfigImpl& result)
{
    bool known_config = false;
    std::string conf_key;
    std::string conf_val;

    // find a db and configuration in it
    known_config = mloSearchConfigInDB(params, conf_key, conf_val);

    // if found save it

    if(known_config)
    {
        mloSetConf(conf_val, result);
    }
    else
    // otherwise
    {
        // select default
        mloSelectDefaultConfig(conf_val, params, result);
        // save the unknown configuration
        // if allowed
        if(params.save_srch_req)
        {
            mloAddConfigReq(params.GetStream(), conf_key);
        }
    }

    return (known_config);
}

static int mloWriteConfigDB(Handle& stream, const std::map<std::string, std::string>& conf_db)
{

    int ret = 0;
    // serialize
    std::string conf_file =
        miopen::GetDbPath(); // (_kernel_path == "") ? miopen::GetDbPath() : _kernel_path;

    conf_file += std::string("/") + stream.GetDeviceName() + "_" +
                 std::to_string(stream.GetMaxComputeUnits()) + "." + std::string("cd.pdb.txt");

    std::vector<std::string> db;

    std::map<std::string, std::string>::const_iterator it;

    for(it = conf_db.begin(); it != conf_db.end(); ++it)
    {
        db.push_back((*it).first + std::string(" ") + (*it).second + std::string("\n"));
    }

    ret = mloUpdateDb(conf_file, db);

    return (ret);
}

static int mloRemoveConfigReq(Handle& stream, const std::string& conf_key)
{
    int ret = 0;
    std::vector<std::string> req_conf_db;

    std::vector<std::string>::iterator it;

    std::string conf_file =
        miopen::GetDbPath(); // (_kernel_path == "") ? miopen::GetDbPath() : _kernel_path;
    conf_file += std::string("/") + stream.GetDeviceName() + "_" +
                 std::to_string(stream.GetMaxComputeUnits()) + "." + std::string("cd.rdb.txt");

    bool found = mloFindConfigReq(conf_file, conf_key, req_conf_db, it);

    if(found)
    {
        req_conf_db.erase(it);
        ret = mloUpdateDb(conf_file, req_conf_db);
    }
    return (ret);
}

static int mloAddConfig(Handle& stream, std::string& conf_key, std::string& conf_val)
{
    int ret = 0;

    // build searchable db
    std::map<std::string, std::string> conf_db;

    mloReadConfigDB(stream, conf_db);
    // add config

    conf_db[conf_key] = conf_val;
    // serialize
    ret = mloWriteConfigDB(stream, conf_db);

    // remove request
    mloRemoveConfigReq(stream, conf_key);

    return (ret);
}

const std::vector<std::unique_ptr<const Solver>>&
ConvOclDirectFwdLegacyExhaustiveSearch::GetImplementationsToMeasure()
{
    static const std::vector<std::unique_ptr<const Solver>>
        implementations = [] {
            std::vector<std::unique_ptr<const Solver>> data;
            data.emplace_back(new ConvOclDirectFwd1x1);
            data.emplace_back(new ConvOclDirectFwdC);
            data.emplace_back(new ConvOclDirectFwd);
            return data;
        }();

    return implementations;
}

/*
* Measure the current configuration performance.
*/
int ConvOclDirectFwdLegacyExhaustiveSearch::MeasureLoop(
    Handle* profile_h,
    Data_t bot_ocl_buf,
    Data_t top_ocl_buf,
    Data_t wei_ocl_buf,
    Data_t bias_ocl_buf,
    double& processing_time,
    const ConvolutionContext& params) const
{
    int ret = 0;
    ConvSolution kernel_search_result;
    auto sub_search_params      = params;
    sub_search_params.do_search = false;

    for(const auto& traits : GetImplementationsToMeasure())
    {
        if(traits->IsApplicable(params))
        {
            const auto sub_search_result = Find(params);
            traits->GetSolution(kernel_search_result, params, *sub_search_result);

            if(kernel_search_result.Succeeded())
                break;
        }
    }

    if(!kernel_search_result.Succeeded())
    {
        return 1;
    }

    const auto kernel_params     = kernel_search_result.construction_params[0];
    std::string compiler_options = params.general_compile_options + kernel_params.comp_options;

    // Creating OCLKernel obj
    try
    {

        float padding_value = 0;

        double s = 0, e = 0;
        int iter = 1;

        if(profile_h)
        {
            processing_time = std::numeric_limits<float>::max();

            auto k = profile_h->GetKernel("",
                                          "",
                                          kernel_params.kernel_file,
                                          kernel_params.kernel_name,
                                          kernel_params.l_wk,
                                          kernel_params.g_wk,
                                          compiler_options);

            if(params.bias)
            {
                k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
            }
            else
            {
                k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
            }
            processing_time = profile_h->GetKernelTime();
        }
        else
        {
            iter = (params.n_timer_iter <= 0) ? 1 : params.n_timer_iter;

            auto k = params.GetStream().GetKernel("",
                                                  "",
                                                  kernel_params.kernel_file,
                                                  kernel_params.kernel_name,
                                                  kernel_params.l_wk,
                                                  kernel_params.g_wk,
                                                  compiler_options);

            if(params.bias)
            {
                k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
            }
            else
            {
                k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
            }

            params.GetStream().Finish();

            s = miopen_mach_absolute_time();

            for(int i = 0; i < iter && ret == 0; i++)
            {
                if(params.bias)
                {
                    k(bot_ocl_buf, wei_ocl_buf, bias_ocl_buf, top_ocl_buf, padding_value);
                }
                else
                {
                    k(bot_ocl_buf, wei_ocl_buf, top_ocl_buf, padding_value);
                }
            }

            params.GetStream().Finish();
            e = miopen_mach_absolute_time();

            processing_time = subtractTimes(e, s) / iter;
        }
    }
    catch(miopen::Exception&)
    {
        return -1;
    }

    return (ret);
}

std::unique_ptr<Solver::PerformanceConfig>
ConvOclDirectFwdLegacyExhaustiveSearch::Find(
    const ConvolutionContext& params) const
{
    auto result = std::make_unique<PerformanceConfigImpl>();

    // search known configurations
    bool known_config = mloGetConfig(params, *result);
    // if not known and the search is allowed - do actual search

    if(!known_config)
    {
        if(params.do_search)
        {
            SearchDirect2D(params, *result);
        }
    }

    return std::unique_ptr<Solver::PerformanceConfig>(result.release());
}

void ConvOclDirectFwdLegacyExhaustiveSearch::SearchDirect2D(
    const ConvolutionContext& params, PerformanceConfigImpl& result) const
{
    miopen::Handle profile_h;
    double processing_time;
    std::string conf_key;
    std::string conf_val;

    int min_grp_tile0 = 16;
    int min_grp_tile1 = 16;
    // tile 0
    int min_in_tile0 = 16;
    // tile 1
    int min_in_tile1 = 16;
    // out pix 0
    int min_out_pix_tile0 = 1;
    // out pix 1
    int min_out_pix_tile1   = 1;
    int min_n_out_pix_tiles = 2;
    int min_n_in_data_tiles = 3;
    int min_n_stacks        = 1;

    // enable profiling for the handle for benchmarking
    profile_h.EnableProfiling();

    size_t localMemSize = profile_h.GetLocalMemorySize();
    profile_h.EnableProfiling();

    // const auto hw_wave_sz = 64;
    const auto dev_local_mem_sz = localMemSize; // in bytes

    // if it is not known
    bool known_config = mloSearchConfigInDB(params, conf_key, conf_val);

    // proceed
    if(!known_config)
    {

        // allocate tem input/output buffers
        size_t bot_sz = params.bot_sz / sizeof(float);
        std::vector<float> bot_sys_buf(bot_sz);

        for(int i = 0; i < bot_sz; i++)
        {
            bot_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
        }

        auto bot_ocl_buf = profile_h.Write(bot_sys_buf);

        size_t top_sz = params.top_sz / sizeof(float);
        std::vector<float> top_sys_buf(top_sz);

        auto top_ocl_buf = profile_h.Write(top_sys_buf);

        std::vector<float> random_top_sys_buf(top_sz);
        for(int i = 0; i < top_sz; i++)
        {
            random_top_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
        }

        size_t weights_sz = params.weights_sz / sizeof(float);
        std::vector<float> wei_sys_buf(weights_sz);
        for(int i = 0; i < weights_sz; i++)
        {
            wei_sys_buf[i] = static_cast<float>((rand() * (1.0 / RAND_MAX) - 0.5) * 0.001);
        }

        auto wei_ocl_buf = profile_h.Write(wei_sys_buf);

        std::vector<float> bias_sys_buf;
        ManageDataPtr bias_ocl_buf = nullptr;

        if(params.bias)
        {
            size_t bias_sz = params.bias_sz / sizeof(float);
            bias_sys_buf   = std::vector<float>(bias_sz);
            for(int i = 0; i < bias_sz; i++)
            {
                bias_sys_buf[i] = static_cast<float>(rand() * (1.0 / RAND_MAX));
            }

            bias_ocl_buf = profile_h.Write(bias_sys_buf);
        }

        // search loop here
        int grp_tl_ln[4]       = {8, 16};
        int tile_sz[3]         = {8, 16, 32};
        int tile_sz1[3]        = {8, 16, 32};
        int tile_sz0[3]        = {8, 16, 32};
        int out_pix_tile_sz[3] = {1, 2, 4};
        int n_out_tiles_rg[2]  = {1, 8};
        int n_in_tiles_rg[2]   = {1, 4};
        int n_in_stacks_sz[3]  = {1, 2, 4};
        int in_tiles[4]        = {64, 128, 256, 2048};
        /*
        std::vector<int> v_tile_sz;
        std::vector<int> v_out_pix_tile_sz;
        std::vector<int> v_n_out_tiles_rg;
        std::vector<int> v_n_in_tiles_rg;
        std::vector<int> v_n_in_stacks_sz;
        */
        //

        double min_proc_time = std::numeric_limits<float>::max();

#if 1

        size_t run_counter = 0;
        int n_grp_tiles1   = 2;
        int n_grp_tiles0   = 2;

        int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
        int n_out_tls      = n_out_tiles_rg[1];
        int stack_cnt      = 2;
        int n_tile0_sz     = 3;
        int n_tile1_sz     = 3;

        n_out_tls = std::min(params.n_outputs, n_out_tls);

        if(params.in_width <= 8)
        {
            n_tile0_sz       = 1;
            n_in_tiles_rg[1] = 16;
        }
        else if(params.in_width <= 16)
        {
            n_tile0_sz       = 1;
            tile_sz0[0]      = 16;
            n_in_tiles_rg[1] = 8;
        }
        else if(params.in_width <= 32)
        {
            n_tile0_sz  = 2;
            tile_sz0[0] = 16;
            tile_sz0[1] = 32;
        }

        if(params.in_height <= 8)
        {
            n_tile1_sz       = 1;
            n_in_tiles_rg[1] = 16;
        }
        else if(params.in_height <= 16)
        {
            n_tile1_sz       = 1;
            tile_sz1[0]      = 16;
            n_in_tiles_rg[1] = 8;
        }
        else if(params.in_width <= 32)
        {
            n_tile1_sz  = 2;
            tile_sz1[0] = 16;
            tile_sz1[1] = 32;
        }

        bool unaligned = (params.out_height < 8 || params.out_width < 8 ||
                          (params.out_height > 8 && params.out_height < 16) ||
                          (params.out_width > 8 && params.out_width < 16) ||
                          (params.out_height > 16 && params.out_height < 32) ||
                          (params.out_width > 16 && params.out_width < 32));

        if(unaligned)
        {
            out_pix_tile_sz[1] = 6;
            out_pix_tl_cnt     = out_pix_tile_sz[1];
        }

        int n_grp_tiles = n_grp_tiles1 * n_grp_tiles0;

        int n_tiles_cnt = n_tile0_sz * n_tile1_sz;
        n_grp_tiles =
            (params.out_height > 16 && params.out_width > 16) ? n_grp_tiles - 1 : n_grp_tiles;
        n_tiles_cnt =
            (params.out_height > 16 && params.out_width > 16) ? n_tiles_cnt - 1 : n_tiles_cnt;
        size_t report_inteval = 100;
        //			_n_timer_iter = 250;

        long long runs_left = 0;

        if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
        {
            grp_tl_ln[0] = 64;
            grp_tl_ln[1] = 128;
            grp_tl_ln[2] = 256;
            grp_tl_ln[3] = 512;
            n_grp_tiles1 = 1;
            n_grp_tiles0 = 4;

            tile_sz1[0] = 1;
            tile_sz0[0] = 4;
            n_tile0_sz = n_tile1_sz = 1;
            n_tiles_cnt             = n_tile0_sz * n_tile1_sz;
            out_pix_tile_sz[0]      = (unaligned) ? 0 : out_pix_tile_sz[0];
            out_pix_tile_sz[1]      = 1;
            n_out_tiles_rg[1]       = 16;
            n_in_tiles_rg[1]        = 8;
            stack_cnt               = 3;
            out_pix_tl_cnt          = out_pix_tile_sz[1];
            n_out_tls               = n_out_tiles_rg[1];
            n_grp_tiles             = n_grp_tiles1 * n_grp_tiles0;

            report_inteval = 20;
        }

        if(params.kernel_size0 == 1 && params.kernel_size1 == 1 &&
           (params.n_outputs / 16) * 16 == params.n_outputs &&
           (params.n_inputs / 4) * 4 == params.n_inputs)
        {

            std::cout
                << "Searching the best solution in the 4 dim space. Please, be patient it may "
                   "take few minutes."
                << std::endl;
            result.grp_tile1 = 1;
            result.in_tile1  = 1;
            result.in_tile0  = 1;
            report_inteval   = 4;

            if(params.forward && (params.n_inputs / 8) * 8 == params.n_inputs)
            {

                // uint N_LCL_IN_MAPS = result.n_in_data_tiles;
                n_in_tiles_rg[0] = 0;
                n_in_tiles_rg[1] = 3;
                int n_in_tls     = 4;

                //					int N_LCL_OUT_MAPS = result.n_out_pix_tiles;
                n_out_tiles_rg[0] = 4;
                n_out_tiles_rg[1] = 6;
                // 0 or 1
                out_pix_tl_cnt = 3;
                //					uint CHEAT_SHADER_COMPILER =
                // result.out_pix_tile0;
                out_pix_tile_sz[0] = 0;
                out_pix_tile_sz[1] = 1;
                n_out_tls          = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
                n_grp_tiles0       = 0;

                runs_left = out_pix_tl_cnt * n_out_tls * n_in_tls * (n_grp_tiles0 + 1);

                result.out_pix_tile1 = 1;
            }
            else
            {
                int i_sz = params.in_width * params.in_height;
                if(params.kernel_stride0 == 1)
                {
                    out_pix_tl_cnt = (i_sz & 1) ? 1 : (i_sz & 0x3) ? 2 : 3;
                }
                else
                {
                    if(params.forward)
                    {
                        out_pix_tl_cnt = (params.out_width & 1) ? 1 : 2;
                    }
                    else
                    {
                        out_pix_tl_cnt = ((params.out_width & 1) || (params.in_width & 1)) ? 1 : 2;
                    }
                }
                out_pix_tile_sz[0] = 1;
                out_pix_tile_sz[1] = 2;
                out_pix_tile_sz[2] = 4;

                n_out_tiles_rg[0] = 2;
                n_out_tiles_rg[1] =
                    ((params.n_outputs / 64) * 64 == params.n_outputs)
                        ? 6
                        : ((params.n_outputs / 32) * 32 == params.n_outputs) ? 5 : 4;

                n_in_tiles_rg[0] = 2;
                n_in_tiles_rg[1] = ((params.n_inputs / 8) * 8 == params.n_inputs) ? 3 : 2;

                grp_tl_ln[0] = 64;
                grp_tl_ln[1] = 128;
                grp_tl_ln[2] = 256;
                n_grp_tiles0 = 3;
                n_grp_tiles1 = 1;

                n_grp_tiles  = n_grp_tiles1 * n_grp_tiles0;
                n_out_tls    = (n_out_tiles_rg[1] - n_out_tiles_rg[0] + 1);
                int n_in_tls = 2;
                runs_left    = n_grp_tiles * out_pix_tl_cnt * n_out_tls * n_in_tls;

                result.out_pix_tile1 = 0;
            }

            int version = result.out_pix_tile1;

            for(int g0 = 0; g0 <= n_grp_tiles0; ++g0)
            {
                result.grp_tile0 = grp_tl_ln[g0];

                // out pix 0
                for(int o_t = n_out_tiles_rg[0]; o_t <= n_out_tiles_rg[1]; ++o_t)
                {
                    result.n_out_pix_tiles = (1 << o_t);
                    for(int l = 0; l < out_pix_tl_cnt; ++l)
                    {
                        result.out_pix_tile0 = out_pix_tile_sz[l];
                        if(version == 0 &&
                           ((result.n_out_pix_tiles == 32 && result.out_pix_tile0 >= 4) ||
                            (result.n_out_pix_tiles == 64 && result.out_pix_tile0 >= 2)))
                        {
                            continue;
                        }

                        for(int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1]; ++i_t)
                        {
                            if(version == 1)
                            {
                                result.n_in_data_tiles = in_tiles[i_t];
                            }
                            else
                            {
                                result.n_in_data_tiles = (1 << i_t);
                            }
                            // randomize output
                            profile_h.WriteTo(
                                reinterpret_cast<const void*>(random_top_sys_buf.data()),
                                top_ocl_buf,
                                random_top_sys_buf.size() * sizeof(float));

                            const auto ret = MeasureLoop(&profile_h,
                                                          bot_ocl_buf.get(),
                                                          top_ocl_buf.get(),
                                                          wei_ocl_buf.get(),
                                                          bias_ocl_buf.get(),
                                                          processing_time,
                                                          params);

                            if(ret != 0)
                            {
                                std::cout << "Failed run." << std::endl;
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }

                            if(run_counter != 0 && run_counter % report_inteval == 0)
                            {
                                std::cout << "Runs left : " << runs_left << ", "
                                          << "min time so far : " << min_proc_time << ", "
                                          << "curr time : " << processing_time
#if 1
                                          << ", " << result.grp_tile1 << ", " << result.grp_tile0
                                          << ", " << result.in_tile1 << ", " << result.in_tile0
                                          << ", " << result.out_pix_tile1 << ", "
                                          << result.out_pix_tile0 << ", " << result.n_out_pix_tiles
                                          << ", " << result.n_in_data_tiles << ", "
                                          << result.n_stacks
#endif
                                          << std::endl;
                            }

                            run_counter++;
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            if(min_proc_time > processing_time)
                            {
                                min_proc_time       = processing_time;
                                min_grp_tile0       = result.grp_tile0;
                                min_grp_tile1       = result.grp_tile1;
                                min_in_tile0        = result.in_tile0;
                                min_in_tile1        = result.in_tile1;
                                min_out_pix_tile0   = result.out_pix_tile0;
                                min_out_pix_tile1   = result.out_pix_tile1;
                                min_n_out_pix_tiles = result.n_out_pix_tiles;
                                min_n_in_data_tiles = result.n_in_data_tiles;
                                min_n_stacks        = result.n_stacks;
                            }

                        } // for (int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1];
                          // ++i_t)
                    }     // if (result.out_pix_tile0 > result.in_tile0)
                }         // for (int l = 0; l < l_l; ++l)
            }             // for (int g0 = 0; g0 < 2; ++g0)
        }
        else
        {

            std::cout
                << "Searching the best solution in the 9 dim space. Please, be patient it may "
                   "take few minutes."
                << std::endl;

            runs_left = n_grp_tiles * n_tiles_cnt * out_pix_tl_cnt * out_pix_tl_cnt * n_out_tls *
                        n_in_tiles_rg[1] * stack_cnt;

            for(int g1 = 0; g1 < n_grp_tiles1; g1++)
            {
                result.grp_tile1 =
                    (params.kernel_size0 == 1 && params.kernel_size1 == 1) ? 1 : grp_tl_ln[g1];
                for(int g0 = 0; g0 < n_grp_tiles0; ++g0)
                {
                    result.grp_tile0 = grp_tl_ln[g0];

                    // tile1
                    for(int j = 0; j < n_tile1_sz; ++j)
                    {
                        result.in_tile1 = tile_sz1[j];
                        if(params.out_height * 2 <= result.in_tile1 && result.in_tile1 > tile_sz[0])
                        {
                            runs_left--;
                            runs_left = (runs_left < 0) ? 0 : runs_left;
                            continue;
                        }

                        // tile 0
                        for(int i = 0; i < n_tile0_sz; ++i)
                        {
                            result.in_tile0 = tile_sz0[i];
                            if((params.out_width * 2 <= result.in_tile0 &&
                                result.in_tile0 > tile_sz[0]))
                            {
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }
                            if(params.out_height > 16 && params.out_width > 16 &&
                               ((result.in_tile1 == 8 && result.in_tile0 == 8) ||
                                (result.grp_tile0 == 8 && result.grp_tile1 == 8)))
                            {
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }
                            if(params.out_width > 32 && result.in_tile1 > result.in_tile0)
                            {
                                runs_left--;
                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                continue;
                            }
                            // out pix 1

                            for(int k = (unaligned) ? out_pix_tile_sz[0] : 0; k < out_pix_tl_cnt;
                                ++k)
                            {
                                result.out_pix_tile1 = (unaligned) ? k : out_pix_tile_sz[k];
                                if(result.out_pix_tile1 > result.in_tile1)
                                {
                                    runs_left--;
                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                    continue;
                                }
                                // out pix 0

                                for(int l = (unaligned) ? out_pix_tile_sz[0] : 0;
                                    l < out_pix_tl_cnt;
                                    ++l)
                                {
                                    result.out_pix_tile0 =
                                        (params.kernel_size0 == 1 && params.kernel_size1 == 1)
                                            ? 4
                                            : (unaligned) ? l : out_pix_tile_sz[l];

                                    if(result.out_pix_tile0 > result.in_tile0)
                                    {
                                        runs_left--;
                                        runs_left = (runs_left < 0) ? 0 : runs_left;
                                        continue;
                                    }

                                    int o_l = n_out_tiles_rg[1];
                                    for(int o_t = n_out_tiles_rg[0]; o_t <= o_l; ++o_t)
                                    {
                                        result.n_out_pix_tiles = o_t;
                                        if(params.n_outputs < result.n_out_pix_tiles)
                                        {
                                            runs_left--;
                                            runs_left = (runs_left < 0) ? 0 : runs_left;
                                            continue;
                                        }
#if 1
                                        if(params.kernel_size0 == 1 && params.kernel_size1 == 1)
                                        {
                                            int N4S = 1;

                                            int MAP_SZ4 =
                                                (params.in_width * params.in_height + N4S * 4 - 1) /
                                                (N4S * 4);

                                            int GRP_SZ          = result.grp_tile0;
                                            int N_MAPS_PERGROUP = 1;
                                            int exchange_step;

                                            if(MAP_SZ4 <= GRP_SZ / 2)
                                            {
                                                N_MAPS_PERGROUP   = GRP_SZ / MAP_SZ4;
                                                int lcl_mem_avial = (result.grp_tile0 <= 192)
                                                                        ? (dev_local_mem_sz / 4) / 2
                                                                        : (dev_local_mem_sz / 4);

                                                exchange_step =
                                                    lcl_mem_avial / (N_MAPS_PERGROUP * MAP_SZ4 * 4);
                                                exchange_step = std::min(
                                                    std::min(exchange_step, result.n_out_pix_tiles),
                                                    N_MAPS_PERGROUP);
                                                if(exchange_step < result.n_out_pix_tiles)
                                                {
                                                    auto tmp_stp =
                                                        static_cast<int>(std::ceil(std::sqrt(
                                                            static_cast<float>(exchange_step))));
                                                    n_in_tiles_rg[0] = tmp_stp;
                                                    n_in_tiles_rg[1] = exchange_step;
                                                }
                                                else
                                                {
                                                    n_in_tiles_rg[0] = 1;
                                                    n_in_tiles_rg[1] = 1;
                                                }
                                            }
                                        }
#endif
                                        for(int i_t = n_in_tiles_rg[0]; i_t <= n_in_tiles_rg[1];
                                            ++i_t)
                                        {
                                            result.n_in_data_tiles = i_t;
                                            if(params.n_inputs < result.n_in_data_tiles)
                                            {
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                continue;
                                            }

                                            for(int s = 0; s < stack_cnt; ++s)
                                            {

                                                result.n_stacks = n_in_stacks_sz[s];
                                                if(params.kernel_size0 == 1 &&
                                                   params.kernel_size1 == 1)
                                                {
                                                }
                                                else
                                                {
                                                    int alu_tile0 = std::max(
                                                        1, result.in_tile0 / result.out_pix_tile0);
                                                    int alu_tile1 = std::max(
                                                        1, result.in_tile1 / result.out_pix_tile1);
                                                    int alu_tiles_sz = (alu_tile0 * alu_tile1);
                                                    int n_alus_total =
                                                        (result.grp_tile0 * result.grp_tile1);

                                                    if(alu_tiles_sz >
                                                       n_alus_total /* || result.n_in_data_tiles*result.n_out_pix_tiles*result.out_pix_tile1*result.out_pix_tile0 > 240*/)
                                                    {
                                                        runs_left--;
                                                        runs_left = (runs_left < 0) ? 0 : runs_left;
                                                        continue;
                                                    }
                                                }

                                                if(result.n_stacks > params.batch_sz)
                                                {
                                                    runs_left--;
                                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                                    continue;
                                                }
                                                const auto ret = MeasureLoop(&profile_h,
                                                                              bot_ocl_buf.get(),
                                                                              top_ocl_buf.get(),
                                                                              wei_ocl_buf.get(),
                                                                              bias_ocl_buf.get(),
                                                                              processing_time,
                                                                              params);

                                                if(ret != 0)
                                                {
                                                    std::cout << "Failed run." << std::endl;
                                                    runs_left--;
                                                    runs_left = (runs_left < 0) ? 0 : runs_left;
                                                    continue;
                                                }

                                                if(run_counter != 0 &&
                                                   run_counter % report_inteval == 0)
                                                {
                                                    std::cout
                                                        << "Runs left : " << runs_left << ", "
                                                        << "min time so far : " << min_proc_time
                                                        << ", "
                                                        << "curr time : " << processing_time
#if 1
                                                        << ", " << result.grp_tile1 << ", "
                                                        << result.grp_tile0 << ", "
                                                        << result.in_tile1 << ", "
                                                        << result.in_tile0 << ", "
                                                        << result.out_pix_tile1 << ", "
                                                        << result.out_pix_tile0 << ", "
                                                        << result.n_out_pix_tiles << ", "
                                                        << result.n_in_data_tiles << ", "
                                                        << result.n_stacks
#endif
                                                        << std::endl;
                                                }

                                                run_counter++;
                                                runs_left--;
                                                runs_left = (runs_left < 0) ? 0 : runs_left;
                                                if(min_proc_time > processing_time)
                                                {
                                                    min_proc_time       = processing_time;
                                                    min_grp_tile0       = result.grp_tile0;
                                                    min_grp_tile1       = result.grp_tile1;
                                                    min_in_tile0        = result.in_tile0;
                                                    min_in_tile1        = result.in_tile1;
                                                    min_out_pix_tile0   = result.out_pix_tile0;
                                                    min_out_pix_tile1   = result.out_pix_tile1;
                                                    min_n_out_pix_tiles = result.n_out_pix_tiles;
                                                    min_n_in_data_tiles = result.n_in_data_tiles;
                                                    min_n_stacks        = result.n_stacks;
                                                }

                                            } // for (int s = 0; s < 3; ++s)
                                        }     // for (int i_t = n_in_tiles_rg[0]; i_t <=
                                              // n_in_tiles_rg[1];
                                              // ++i_t)
                                    }         // if (result.out_pix_tile0 > result.in_tile0)
                                }             // for (int l = 0; l < l_l; ++l)
                            }                 // for (int k = 0; k < k_l; ++k)
                        }                     // for (int i = 0; i < 3; ++i)
                    }                         // for (int j = 0; j < 3; ++j)
                }                             // for (int g0 = 0; g0 < 2; ++g0)
            }                                 // for (int g1 = 0; g1 < 2; g1++)
        }

        std::cout << std::endl << "Score: " << min_proc_time << std::endl;
#endif

        mloBuildConf_Val(conf_val,
                         min_grp_tile1,
                         min_grp_tile0,
                         min_in_tile1,
                         min_in_tile0,
                         min_out_pix_tile1,
                         min_out_pix_tile0,
                         min_n_out_pix_tiles,
                         min_n_in_data_tiles,
                         min_n_stacks);

        mloAddConfig(params.GetStream(), conf_key, conf_val);
        // set the learnt data fo the current run.
        mloSetConf(conf_val, result);
    }

    profile_h.EnableProfiling(false);

    // return(ret);
}

} // namespace solver
} // namespace miopen
