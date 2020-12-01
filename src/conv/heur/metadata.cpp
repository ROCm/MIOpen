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

#include <miopen/conv/heur/metadata.hpp>

#include <cstring>

extern char fwd_gfx908_binary_param_bin_start;
extern char fwd_gfx908_binary_param_bin_end;
extern "C" miopen::MemRef2D
fwd_gfx908_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern char bwd_gfx908_binary_param_bin_start;
extern char bwd_gfx908_binary_param_bin_end;
extern "C" miopen::MemRef2D
bwd_gfx908_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern char wrw_gfx908_binary_param_bin_start;
extern char wrw_gfx908_binary_param_bin_end;
extern "C" miopen::MemRef2D
wrw_gfx908_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern char fwd_gfx906_binary_param_bin_start;
extern char fwd_gfx906_binary_param_bin_end;
extern "C" miopen::MemRef2D
fwd_gfx906_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern char bwd_gfx906_binary_param_bin_start;
extern char bwd_gfx906_binary_param_bin_end;
extern "C" miopen::MemRef2D
bwd_gfx906_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern char wrw_gfx906_binary_param_bin_start;
extern char wrw_gfx906_binary_param_bin_end;
extern "C" miopen::MemRef2D
wrw_gfx906_main_graph(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
extern "C" void* fwd_gfx908_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&fwd_gfx908_binary_param_bin_end -
                                          &fwd_gfx908_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &fwd_gfx908_binary_param_bin_start, size);
    return buffer;
}

extern "C" void* bwd_gfx908_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&bwd_gfx908_binary_param_bin_end -
                                          &bwd_gfx908_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &bwd_gfx908_binary_param_bin_start, size);
    return buffer;
}

extern "C" void* wrw_gfx908_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&wrw_gfx908_binary_param_bin_end -
                                          &wrw_gfx908_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &wrw_gfx908_binary_param_bin_start, size);
    return buffer;
}

extern "C" void* fwd_gfx906_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&fwd_gfx906_binary_param_bin_end -
                                          &fwd_gfx906_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &fwd_gfx906_binary_param_bin_start, size);
    return buffer;
}

extern "C" void* bwd_gfx906_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&bwd_gfx906_binary_param_bin_end -
                                          &bwd_gfx906_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &bwd_gfx906_binary_param_bin_start, size);
    return buffer;
}

extern "C" void* wrw_gfx906_getEmbeddedConstPool(int64_t /*_*/)
{
    auto size = static_cast<unsigned int>(&wrw_gfx906_binary_param_bin_end -
                                          &wrw_gfx906_binary_param_bin_start);
    void* buffer = malloc(size);
    if(buffer != nullptr)
        memcpy(buffer, &wrw_gfx906_binary_param_bin_start, size);
    return buffer;
}
namespace miopen {

const std::vector<std::string>& GetFeatureNames(const Handle& handle)
{
    const auto& arch = handle.GetDeviceName();

    if(arch == "gfx908")

    {
        static const std::vector<std::string> gfx908_feature_names = {"batchsize",
                                                                      "conv_stride_h",
                                                                      "conv_stride_w",
                                                                      "fil_h",
                                                                      "fil_w",
                                                                      "in_channels",
                                                                      "in_h",
                                                                      "in_w",
                                                                      "out_channels",
                                                                      "pad_h",
                                                                      "pad_w"};
        return gfx908_feature_names;
    }
    else if(arch == "gfx906")

    {
        static const std::vector<std::string> gfx906_feature_names = {"batchsize",
                                                                      "conv_stride_h",
                                                                      "conv_stride_w",
                                                                      "fil_h",
                                                                      "fil_w",
                                                                      "in_channels",
                                                                      "in_h",
                                                                      "in_w",
                                                                      "out_channels",
                                                                      "pad_h",
                                                                      "pad_w"};
        return gfx906_feature_names;
    }

    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

const std::unordered_map<int, std::string>& GetSolverMap(const Handle& handle,
                                                         const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd         = problem.direction.IsForward();
    auto bwd         = problem.direction.IsBackwardData();
    auto wrw         = problem.direction.IsBackwardWrW();

    if(arch == "gfx908" && fwd)

    {
        static const std::unordered_map<int, std::string> gfx908_solver_map = {
            {0, "ConvAsmImplicitGemmGTCDynamicFwdXdlops"},
            {1, "ConvBinWinograd3x3U"},
            {2, "ConvBinWinogradRxS"},
            {3, "ConvBinWinogradRxSf2x3"},
            {4, "ConvBinWinogradRxSf3x2"},
            {5, "ConvMPBidirectWinograd<3-3>"},
            {6, "ConvMPBidirectWinograd<4-3>"},
            {7, "ConvMPBidirectWinograd<5-3>"},
            {8, "ConvMPBidirectWinograd<6-3>"},
            {9, "gemm"}};
        return gfx908_solver_map;
    }
    else if(arch == "gfx908" && bwd)

    {
        static const std::unordered_map<int, std::string> gfx908_solver_map = {
            {0, "ConvAsmImplicitGemmGTCDynamicBwdXdlops"},
            {1, "ConvBinWinograd3x3U"},
            {2, "ConvBinWinogradRxS"},
            {3, "ConvBinWinogradRxSf2x3"},
            {4, "ConvBinWinogradRxSf3x2"},
            {5, "ConvMPBidirectWinograd<3-3>"},
            {6, "ConvMPBidirectWinograd<4-3>"},
            {7, "ConvMPBidirectWinograd<5-3>"},
            {8, "ConvMPBidirectWinograd<6-3>"},
            {9, "gemm"}};
        return gfx908_solver_map;
    }

    else if(arch == "gfx908" && wrw)

    {
        static const std::unordered_map<int, std::string> gfx908_solver_map = {
            {0, "ConvAsmImplicitGemmGTCDynamicWrwXdlops"},
            {1, "ConvBinWinogradRxS"},
            {2, "ConvBinWinogradRxSf2x3"},
            {3, "ConvWinograd3x3MultipassWrW<3-4>"},
            {4, "ConvWinograd3x3MultipassWrW<3-5>"},
            {5, "ConvWinograd3x3MultipassWrW<3-6>"},
            {6, "gemm"}};
        return gfx908_solver_map;
    }

    else if(arch == "gfx906" && fwd)

    {
        static const std::unordered_map<int, std::string> gfx906_solver_map = {
            {0, "ConvAsmImplicitGemmV4R1DynamicFwd"},
            {1, "ConvAsmImplicitGemmV4R1DynamicFwd_1x1"},
            {2, "ConvBinWinogradRxS"},
            {3, "ConvBinWinogradRxSf2x3"},
            {4, "ConvBinWinogradRxSf3x2"},
            {5, "ConvMPBidirectWinograd<3-3>"},
            {6, "ConvMPBidirectWinograd<4-3>"},
            {7, "ConvMPBidirectWinograd<5-3>"},
            {8, "ConvMPBidirectWinograd<6-3>"},
            {9, "gemm"}};
        return gfx906_solver_map;
    }

    else if(arch == "gfx906" && bwd)

    {
        static const std::unordered_map<int, std::string> gfx906_solver_map = {
            {0, "ConvAsmImplicitGemmV4R1DynamicBwd"},
            {1, "ConvBinWinograd3x3U"},
            {2, "ConvBinWinogradRxS"},
            {3, "ConvBinWinogradRxSf2x3"},
            {4, "ConvBinWinogradRxSf3x2"},
            {5, "ConvMPBidirectWinograd<3-3>"},
            {6, "ConvMPBidirectWinograd<4-3>"},
            {7, "ConvMPBidirectWinograd<5-3>"},
            {8, "ConvMPBidirectWinograd<6-3>"},
            {9, "gemm"}};
        return gfx906_solver_map;
    }

    else if(arch == "gfx906" && wrw)

    {
        static const std::unordered_map<int, std::string> gfx906_solver_map = {
            {0, "ConvAsmImplicitGemmV4R1DynamicWrw"},
            {1, "ConvBinWinogradRxS"},
            {2, "ConvBinWinogradRxSf2x3"},
            {3, "ConvWinograd3x3MultipassWrW<3-4>"},
            {4, "ConvWinograd3x3MultipassWrW<3-5>"},
            {5, "ConvWinograd3x3MultipassWrW<3-6>"},
            {6, "gemm"}};
        return gfx906_solver_map;
    }

    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

const std::vector<float>& GetMu(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd         = problem.direction.IsForward();
    auto bwd         = problem.direction.IsBackwardData();
    auto wrw         = problem.direction.IsBackwardWrW();

    if(arch == "gfx908" && fwd)

    {
        static const std::vector<float> gfx908_fwd_mu = {3.5623828968151168,
                                                         1.0490354202886407,
                                                         1.0490354202886407,
                                                         1.845688192538908,
                                                         1.845688192538908,
                                                         7.2175793137487965,
                                                         7.3473644845741015,
                                                         7.357070521225214,
                                                         6.440877100721474,
                                                         0.0692277078402212,
                                                         0.0692277078402212};
        return gfx908_fwd_mu;
    }
    else if(arch == "gfx908" && bwd)

    {
        static const std::vector<float> gfx908_bwd_mu = {3.562415623729587,
                                                         1.0490364474235443,
                                                         1.0490364474235443,
                                                         1.845664013405949,
                                                         1.845664013405949,
                                                         7.217595713136259,
                                                         7.347286957856764,
                                                         7.357026940010353,
                                                         6.440578999637922,
                                                         0.06922915793883536,
                                                         0.06922915793883536};
        return gfx908_bwd_mu;
    }

    else if(arch == "gfx908" && wrw)

    {
        static const std::vector<float> gfx908_wrw_mu = {3.560701646212823,
                                                         1.049438249704342,
                                                         1.049438249704342,
                                                         1.8454130765331982,
                                                         1.8454130765331982,
                                                         7.205665058322538,
                                                         7.340939813866135,
                                                         7.348962820069909,
                                                         6.442699442123986,
                                                         0.06945852339922284,
                                                         0.06945852339922284};
        return gfx908_wrw_mu;
    }

    else if(arch == "gfx906" && fwd)

    {
        static const std::vector<float> gfx906_fwd_mu = {3.562254229799534,
                                                         1.0490631680429223,
                                                         1.0490631680429223,
                                                         1.8450769166282432,
                                                         1.8450769166282432,
                                                         7.216793060869346,
                                                         7.346732736859883,
                                                         7.356530466828507,
                                                         6.439847943224059,
                                                         0.0692668818376158,
                                                         0.0692668818376158};
        return gfx906_fwd_mu;
    }

    else if(arch == "gfx906" && bwd)

    {
        static const std::vector<float> gfx906_bwd_mu = {3.56393444109118,
                                                         1.0490950653272655,
                                                         1.0490950653272655,
                                                         1.8446196757754336,
                                                         1.8446196757754336,
                                                         7.215053383579987,
                                                         7.3455587154344695,
                                                         7.3550561654984765,
                                                         6.439193034117408,
                                                         0.06931191409936455,
                                                         0.06931191409936455};
        return gfx906_bwd_mu;
    }

    else if(arch == "gfx906" && wrw)

    {
        static const std::vector<float> gfx906_wrw_mu = {3.5591826946882694,
                                                         1.049514583641786,
                                                         1.049514583641786,
                                                         1.8466761141310095,
                                                         1.8466761141310095,
                                                         7.211762075269513,
                                                         7.334758780962944,
                                                         7.343318259426912,
                                                         6.451273305824059,
                                                         0.0695657691575541,
                                                         0.0695657691575541};
        return gfx906_wrw_mu;
    }

    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

const std::vector<float>& GetSigma(const Handle& handle, const ProblemDescription& problem)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd         = problem.direction.IsForward();
    auto bwd         = problem.direction.IsBackwardData();
    auto wrw         = problem.direction.IsBackwardWrW();

    if(arch == "gfx908" && fwd)

    {
        static const std::vector<float> gfx908_fwd_sigma = {2.6491930432687614,
                                                            0.21594200111547812,
                                                            0.21594200111547812,
                                                            1.0550720529005788,
                                                            1.0550720529005788,
                                                            2.918484040479085,
                                                            1.9296308754313851,
                                                            1.933199941799668,
                                                            2.8483697598762787,
                                                            0.31623115824167336,
                                                            0.31623115824167336};
        return gfx908_fwd_sigma;
    }
    else if(arch == "gfx908" && bwd)

    {
        static const std::vector<float> gfx908_bwd_sigma = {2.6492111385763306,
                                                            0.21594414612955404,
                                                            0.21594414612955404,
                                                            1.055069876102947,
                                                            1.055069876102947,
                                                            2.9185627833231207,
                                                            1.9296062790559745,
                                                            1.9331999139466463,
                                                            2.8483469394234184,
                                                            0.3162343115131855,
                                                            0.3162343115131855};
        return gfx908_bwd_sigma;
    }

    else if(arch == "gfx908" && wrw)

    {
        static const std::vector<float> gfx908_wrw_sigma = {2.6423595788308702,
                                                            0.21678124727594192,
                                                            0.21678124727594192,
                                                            1.055564981168448,
                                                            1.055564981168448,
                                                            2.913888072272486,
                                                            1.9268366036241893,
                                                            1.9304613737131484,
                                                            2.849219661449283,
                                                            0.31700652161588144,
                                                            0.31700652161588144};
        return gfx908_wrw_sigma;
    }

    else if(arch == "gfx906" && fwd)

    {
        static const std::vector<float> gfx906_fwd_sigma = {2.6497136461435846,
                                                            0.21599993885303384,
                                                            0.21599993885303384,
                                                            1.0550192070577091,
                                                            1.0550192070577091,
                                                            2.9189336812233235,
                                                            1.9298884275467378,
                                                            1.933442898328871,
                                                            2.8478979717577504,
                                                            0.31631632958394595,
                                                            0.31631632958394595};
        return gfx906_fwd_sigma;
    }

    else if(arch == "gfx906" && bwd)

    {
        static const std::vector<float> gfx906_bwd_sigma = {2.6500491715364114,
                                                            0.21606651727599308,
                                                            0.21606651727599308,
                                                            1.0549941964791987,
                                                            1.0549941964791987,
                                                            2.9189599723227193,
                                                            1.9297505896906963,
                                                            1.9332938577300325,
                                                            2.848377248679382,
                                                            0.316414203533129,
                                                            0.316414203533129};
        return gfx906_bwd_sigma;
    }

    else if(arch == "gfx906" && wrw)

    {
        static const std::vector<float> gfx906_wrw_sigma = {2.6434980940465076,
                                                            0.2169398295577983,
                                                            0.2169398295577983,
                                                            1.0558501715837074,
                                                            1.0558501715837074,
                                                            2.908475872778594,
                                                            1.9275893711030234,
                                                            1.9319344303963975,
                                                            2.843535807406935,
                                                            0.31723940189553435,
                                                            0.31723940189553435};
        return gfx906_wrw_sigma;
    }

    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

miopen::MemRef2D
CallModel(const Handle& handle, const ProblemDescription& problem, miopen::Tensor2D& features)
{
    const auto& arch = handle.GetDeviceName();
    auto fwd         = problem.direction.IsForward();
    auto bwd         = problem.direction.IsBackwardData();
    auto wrw         = problem.direction.IsBackwardWrW();

    if(arch == "gfx908" && fwd)

    {
        return fwd_gfx908_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }
    else if(arch == "gfx908" && bwd)

    {
        return bwd_gfx908_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }

    else if(arch == "gfx908" && wrw)

    {
        return wrw_gfx908_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }

    else if(arch == "gfx906" && fwd)

    {
        return fwd_gfx906_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }

    else if(arch == "gfx906" && bwd)

    {
        return bwd_gfx906_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }

    else if(arch == "gfx906" && wrw)

    {
        return wrw_gfx906_main_graph(features.data(),
                                     features.data(),
                                     features.offset,
                                     features.size0,
                                     features.size1,
                                     features.stride0,
                                     features.stride1);
    }

    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}
} // namespace miopen
