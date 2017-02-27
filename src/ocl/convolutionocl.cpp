#include <mlopen/convolution.hpp>
#include <mlopen/util.hpp>
#include <mlopen/mlo_internal.hpp>

#if MLOPEN_USE_TINYGEMM
#include <mlopen/gemm.hpp>
#endif

namespace mlopen {

void ConvolutionDescriptor::FindConvFwdAlgorithm(Handle& handle,
        const TensorDescriptor&     xDesc,
        ConstData_t             x,
        const TensorDescriptor&     wDesc,
        ConstData_t             w,
        const TensorDescriptor&     yDesc,
        Data_t             y,
        const int                    /*requestAlgoCount*/,
        int                         * /*returnedAlgoCount*/,
        mlopenConvAlgoPerf_t        *perfResults,
        mlopenConvPreference_t       /*preference*/,
        Data_t                      workSpace,
        size_t                       /*workSpaceSize*/,
        bool                        exhaustiveSearch) const {

    if(x == nullptr || w == nullptr || y == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
#if 0
        if(returnedAlgoCount == nullptr || perfResults == nullptr) {
            MLOPEN_THROW(mlopenStatusBadParm);
        }
        if(requestAlgoCount < 1) {
            MLOPEN_THROW(mlopenStatusBadParm);
        }
#endif 

    // GEMM based
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

    std::string network_config;
    std::string program_name;
    std::string kernel_name;
    std::string parms;

#if MLOPEN_USE_TINYGEMM
    if(workSpace != nullptr) {
        if(wei_h != 1 && wei_w != 1) {
            size_t in_offset = 0;
            Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
        }

        GemmGeometry gg = CreateGemmGeometryConvFwd(xDesc, wDesc, yDesc, false, network_config);
        gg.FindSolution(.003, handle, workSpace, w, y, false);
    }
#else
    (void)workSpace; // Suppress warning
#endif

    // Winograd algo
    // TODO: duplicating code for now
    
    mlo_construct_winograd construct_params_wino(1);
    construct_params_wino.setStream(&handle);

    construct_params_wino.setOutputDescFromMLDesc(yDesc);
    construct_params_wino.setInputDescFromMLDesc(xDesc);
    construct_params_wino.setWeightDescFromMLDesc(wDesc);

    construct_params_wino.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

    construct_params_wino.mloConstruct();
    program_name = construct_params_wino.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
    kernel_name = construct_params_wino.getKernelName(); // "hello_world_kernel"; // kernel name
    parms = construct_params_wino.getCompilerOptions(); // kernel parameters

    construct_params_wino.mloBuildConf_Key(network_config);

    const std::vector<size_t> & vld_wino = construct_params_wino.getLocalWkSize();
    const std::vector<size_t> & vgd_wino = construct_params_wino.getGlobalWkSize();

	handle.GetKernel("mlopenConvolutionFwdAlgoWinograd",
		network_config,
		program_name,
		kernel_name,
		vld_wino,
		vgd_wino,
		parms);

    // Direct algo
    // Generate kernels if OpenCL
    // Compile, cache kernels, etc.
    // Launch all kernels and store the perf, workspace limits, etc.
    mlo_construct_direct2D construct_params_direct(1); // forward
    construct_params_direct.doSearch(exhaustiveSearch);
    construct_params_direct.saveSearchRequest(true);

    construct_params_direct.setGeneralCompOptions("");

    construct_params_direct.setStream(&handle);

    construct_params_direct.setOutputDescFromMLDesc(yDesc);
    construct_params_direct.setInputDescFromMLDesc(xDesc);
    construct_params_direct.setWeightDescFromMLDesc(wDesc);

    construct_params_direct.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

    construct_params_direct.mloConstruct();
    program_name = construct_params_direct.getKernelFile();  //"../src/Hello.cl"; // CL kernel filename
    kernel_name = construct_params_direct.getKernelName(); // "hello_world_kernel"; // kernel name
    parms = construct_params_direct.getCompilerOptions(); // kernel parameters

    construct_params_direct.mloBuildConf_Key(network_config);

    const std::vector<size_t> & vld = construct_params_direct.getLocalWkSize();
    const std::vector<size_t> & vgd = construct_params_direct.getGlobalWkSize();

	// float padding_val = 0;

	handle.GetKernel("mlopenConvolutionFwdAlgoDirect",
		network_config,
		program_name,
		kernel_name,
		vld,
		vgd,
		parms);

	// if (kernel.GetName() == "sp3AsmConv3x3F")
	// {
	// 	int flags = 0;
	// 	int reserved = 0;
	// 	int *return_addr = nullptr;
	// 	int N, C, H, W, K, n_groups;
	// 	construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
	// 	kernel(N, C, H, W, K, n_groups, flags, reserved, x, w, y, return_addr);
	// }
	// else
	// {
	// 	kernel(x, w, y, padding_val);
	// }
	
	// FIXME: MD temporary hack for hipcaffe
	// should be ideally wrapped under mlopen::deref to check 
	// for the size of perfResults == requestedAlgoCount
	perfResults->fwd_algo = mlopenConvolutionFwdAlgoDirect;
}

void ConvolutionDescriptor::ConvolutionForward(Handle& handle,
        const void                  * /*alpha*/,
        const TensorDescriptor&     xDesc,
        ConstData_t             x,
        const TensorDescriptor&     wDesc,
        ConstData_t             w,
        mlopenConvFwdAlgorithm_t    algo,
        const void                  * /*beta*/,
        const TensorDescriptor&     yDesc,
        Data_t                      y, 
        Data_t                      workSpace,
        size_t                       /*workSpaceSize*/) const {

    if(x == nullptr || w == nullptr || y == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1]) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(xDesc.GetSize() < 3) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }

    switch (algo)
    {
        case mlopenConvolutionFwdAlgoDirect:
        {
            // TODO(paul): Replicating code for now.
            mlo_construct_direct2D construct_params(1); // forward
            construct_params.setOutputDescFromMLDesc(yDesc);
            construct_params.setInputDescFromMLDesc(xDesc);
            construct_params.setWeightDescFromMLDesc(wDesc);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            std::string algorithm_name = "mlopenConvolutionFwdAlgoDirect";
            float padding_val = 0;
            auto kernel = handle.GetKernel(algorithm_name, network_config);

            kernel(x, w, y, padding_val);
        }
        break;

        case mlopenConvolutionFwdAlgoWinograd:
        {
            mlo_construct_winograd construct_params(1); // forward
            construct_params.setOutputDescFromMLDesc(yDesc);
            construct_params.setInputDescFromMLDesc(xDesc);
            construct_params.setWeightDescFromMLDesc(wDesc);

            construct_params.setStream(&handle);

            std::string network_config;
            construct_params.mloBuildConf_Key(network_config);

            std::string algorithm_name = "mlopenConvolutionFwdAlgoWinograd";
            auto kernel = handle.GetKernel(algorithm_name, network_config);

            int flags = 0;
            int reserved = 0;
            int *return_addr = nullptr;
            int N, C, H, W, K, n_groups;
            construct_params.getCompiledInParameters(&N, &C, &H, &W, &K, &n_groups);
            kernel(N, C, H, W, K, n_groups, flags, reserved, x, w, y, return_addr);
        }
        break;

        case mlopenConvolutionFwdAlgoGEMM:
        {
            if(workSpace == nullptr) {
                MLOPEN_THROW("Workspace is required");
            }

            int in_n, in_c, in_h, in_w;
            std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

            int wei_n, wei_h, wei_w;
            std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

            int out_h, out_w;
            std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

            std::string network_config;
#if MLOPEN_USE_TINYGEMM
            CreateGemmGeometryConvFwd(xDesc, wDesc, yDesc, false, network_config);
            GemmGeometry gg = GetGemmGeometry("mlopenConvolutionFwdAlgoGEMM", network_config);

            float time_0 = 0;
            float t1 = 0;
            for(int i = 0; i < in_n; i++) {
                int out_offset = i * wei_n * out_h * out_w;
                if(wei_h != 1 && wei_w != 1) {
                    size_t in_offset = i * in_c * in_h * in_w;
                    Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    gg.RunGemm(handle, workSpace, w, y, 0, 0, out_offset);

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled()) {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1+time_0);
                        else
                            handle.AccumKernelTime(t1);
                        time_0 += handle.GetKernelTime();
                    }
                }
                else if(wei_h == 1 && wei_w == 1) {
                    int in_offset = i * in_c * in_h * in_w;
                    gg.RunGemm(handle, x, w, y, in_offset, 0, out_offset);
                    if(handle.IsProfilingEnabled()) {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(time_0);
                        time_0 += handle.GetKernelTime();
                    }

                } 
            }
#else
            MLOPEN_THROW("GEMM is not supported");
#endif
        }
        break;
        case mlopenConvolutionFwdAlgoFFT:
            break;
    }
}

// FindBackwardDataAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdDataAlgorithm(Handle& handle,
        const TensorDescriptor&     dyDesc,
        ConstData_t             dy,
        const TensorDescriptor&     wDesc,
        ConstData_t             w,
        const TensorDescriptor&     dxDesc,
        ConstData_t             dx,
        const int                    /*requestAlgoCount*/,
        int                         * /*returnedAlgoCount*/,
        mlopenConvAlgoPerf_t        *perfResults,
        mlopenConvPreference_t       /*preference*/,
        void                        * /*workSpace*/,
        size_t                       /*workSpaceSize*/,
        bool                        exhaustiveSearch) const {

    if(dx == nullptr || w == nullptr || dy == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
#if 0
        if(returnedAlgoCount == nullptr || perfResults == nullptr) {
            MLOPEN_THROW(mlopenStatusBadParm);
        }
        if(requestAlgoCount < 1) {
            MLOPEN_THROW(mlopenStatusBadParm);
        }
#endif 

    // Generate kernels if OpenCL
    // Compile, cache kernels, etc.
    // Launch all kernels and store the perf, workspace limits, etc.
    mlo_construct_direct2D construct_params(0); // backward
    construct_params.doSearch(exhaustiveSearch);
    construct_params.saveSearchRequest(true);

    construct_params.setGeneralCompOptions("");

    construct_params.setStream(&handle);

    construct_params.setOutputDescFromMLDesc(dyDesc);
    construct_params.setInputDescFromMLDesc(dxDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);

    construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);

    construct_params.mloConstruct();

    std::string program_name = construct_params.getKernelFile();
    std::string kernel_name = construct_params.getKernelName(); // kernel name
    std::string parms = construct_params.getCompilerOptions(); // kernel parameters

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    const std::vector<size_t> & vld = construct_params.getLocalWkSize();
    const std::vector<size_t> & vgd = construct_params.getGlobalWkSize();

    float padding_val = 0;
    handle.GetKernel("mlopenConvolutionBwdDataAlgo_0",
            network_config,
            program_name,
            kernel_name,
            vld,
            vgd,
            parms)(dy, w, dx, padding_val);

    // FIXME: MD temporary hack for hipcaffe
    // should be ideally wrapped under mlopen::deref to check 
    // for the size of perfResults == requestedAlgoCount
    perfResults->bwd_data_algo = mlopenConvolutionBwdDataAlgo_0;
    perfResults->time = handle.GetKernelTime();

}

// BackwardDataAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardData(Handle& handle,
        const void                      * /*alpha*/,
        const TensorDescriptor&         dyDesc,
        ConstData_t                 dy,
        const TensorDescriptor&         wDesc,
        ConstData_t                 w,
        mlopenConvBwdDataAlgorithm_t    /* algo */,
        const void                      * /*beta*/,
        const TensorDescriptor&         dxDesc,
        Data_t                          dx, 
        void                            * /*workSpace*/,
        size_t                           /*workSpaceSize*/) const {

    if(dx == nullptr || w == nullptr || dy == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetSize() != dxDesc.GetSize() || dyDesc.GetSize() != wDesc.GetSize()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetType() != dxDesc.GetType() || dyDesc.GetType() != wDesc.GetType()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetLengths()[1] != wDesc.GetLengths()[0]) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetSize() < 3) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }

    // Launch all kernels and store the perf, workspace limits, etc.
    mlo_construct_direct2D construct_params(0); // backward
    {
        construct_params.setOutputDescFromMLDesc(dyDesc);
        construct_params.setInputDescFromMLDesc(dxDesc);
        construct_params.setWeightDescFromMLDesc(wDesc);
        construct_params.setStream(&handle);
    }

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    float padding_val = 0;
    handle.GetKernel("mlopenConvolutionBwdDataAlgo_0", network_config) (dy, w, dx, padding_val);
}

// ConvolutionBackwardWeightsGetWorkSpaceSize
size_t ConvolutionDescriptor::ConvolutionBackwardWeightsGetWorkSpaceSize(
    const TensorDescriptor&      dyDesc,
	const TensorDescriptor&		 xDesc,
	const TensorDescriptor&		 dwDesc) const
{
    mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
    construct_params.doSearch(false);
    construct_params.setOutputDescFromMLDesc(dyDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(dwDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);
    construct_params.mloConstruct();

    // Compute for gemm
    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = mlopen::tie4(dyDesc.GetLengths());
    int wei_c, wei_h, wei_w;
    std::tie(std::ignore, wei_c, wei_h, wei_w) = mlopen::tie4(dwDesc.GetLengths());
    auto gemm_size = wei_c*wei_h*wei_w * out_h*out_w * sizeof(dyDesc.GetType()); // FIXME: sizeof is wrong

    return std::max(construct_params.getWorkSpaceSzBytes(), gemm_size);
}

// FindBackwardWeightsAlgorithm()
//
void ConvolutionDescriptor::FindConvBwdWeightsAlgorithm(Handle& handle,
        const TensorDescriptor&     dyDesc,
        ConstData_t             dy,
        const TensorDescriptor&     xDesc,
        ConstData_t             x,
        const TensorDescriptor&     dwDesc,
        Data_t             dw,
        const int                    /*requestAlgoCount*/,
        int                         * /*returnedAlgoCount*/,
        mlopenConvAlgoPerf_t        *perfResults,
        mlopenConvPreference_t       /*preference*/,
        Data_t                      workSpace,
        size_t                      /*workSpaceSize*/,
        bool                        /*exhaustiveSearch*/) const {

    if(x == nullptr || dw == nullptr || dy == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(workSpace == nullptr) {
        MLOPEN_THROW("Workspace is requried");
    }

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

    if(wei_h != 1 && wei_w != 1) {
        size_t in_offset = 0;
        Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
    }

    std::string network_config;
#if MLOPEN_USE_TINYGEMM
    GemmGeometry gg = CreateGemmGeometryConvBwdWeights(dyDesc, xDesc, dwDesc, false, network_config);
    gg.FindSolution(.003, handle, workSpace, dy, dw, false);
#endif
// temprorary guard
//    if((u == 1 && v == 1) || (wei_w >= 7 && (u > 1 || v > 1))
    {
        mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
        construct_params.doSearch(false);
        construct_params.setStream(&handle);
        construct_params.setOutputDescFromMLDesc(dyDesc);
        construct_params.setInputDescFromMLDesc(xDesc);
        construct_params.setWeightDescFromMLDesc(dwDesc);
        construct_params.setConvDescr(pad_h, pad_w, u, v, upscalex, upscaley);
        construct_params.mloConstruct();

        construct_params.mloBuildConf_Key(network_config);

        const std::vector<mlo_kernel_info> & bwd_wrw_info = construct_params.getKernelsInfo();
        /*
         * get info for all kernels of the layer
         * std::string _kernel_name;
         * std::string _kernel_file;
         * std::string _comp_options;
         * std::vector<size_t> _g_wk;
         * std::vector<size_t> _l_wk;
         */

        //TODO: the kernels should be able to be called from Find()
        // Actually, that is requried to correctly populate the
        // PerfResults. May be we should clear the outputs in Find()
        // after the kernel finishes
        // main kernel
        if (bwd_wrw_info.size() == 1)
        {
            const mlo_kernel_info &bwd_wrw = bwd_wrw_info[0];
            //          float padding_val = 0;

            handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
                    network_config,
                    std::get<1>(bwd_wrw),
                    std::get<0>(bwd_wrw),
                    std::get<4>(bwd_wrw),
                    std::get<3>(bwd_wrw),
                    std::get<2>(bwd_wrw));
            //              (dy, x, dw, padding_val);
        }
        else
        {
            auto bwd_wrw_main = bwd_wrw_info[0];
            //          float padding_val = 0;

            handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
                    network_config,
                    std::get<1>(bwd_wrw_main),
                    std::get<0>(bwd_wrw_main),
                    std::get<4>(bwd_wrw_main),
                    std::get<3>(bwd_wrw_main),
                    std::get<2>(bwd_wrw_main));
            //                  (dy, x, workSpace, padding_val);

            float time0 = handle.GetKernelTime();
            // second kernel hash
            network_config += "x1";
            // reduction  kernel
            auto bwd_wrw_red = bwd_wrw_info[1];

            handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Red",
                    network_config,
                    std::get<1>(bwd_wrw_red),
                    std::get<0>(bwd_wrw_red),
                    std::get<4>(bwd_wrw_red),
                    std::get<3>(bwd_wrw_red),
                    std::get<2>(bwd_wrw_red));
            //                  (workSpace, dw);

            handle.AccumKernelTime(time0);

        }
    }

    // FIXME: MD temporary hack for hipcaffe
    // should be ideally wrapped under mlopen::deref to check 
    // for the size of perfResults == requestedAlgoCount
#if MLOPEN_USE_TINYGEMM
    perfResults->bwd_weights_algo = mlopenConvolutionBwdWeightsAlgoGEMM;
#else
    perfResults->bwd_weights_algo = mlopenConvolutionBwdWeightsAlgoDirect;
#endif
}

// BackwardWeightsAlgorithm()
void ConvolutionDescriptor::ConvolutionBackwardWeights(Handle& handle,
        const void                      * /*alpha*/,
        const TensorDescriptor&         dyDesc,
        ConstData_t                 dy,
        const TensorDescriptor&         xDesc,
        ConstData_t                 x,
        mlopenConvBwdWeightsAlgorithm_t algo,
        const void                      * /*beta*/,
        const TensorDescriptor&         dwDesc,
        Data_t                          dw, 
        Data_t                          workSpace,
        size_t                          /*workSpaceSize*/) const {

    if(x == nullptr || dw == nullptr || dy == nullptr) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetSize() != dwDesc.GetSize() || dyDesc.GetSize() != xDesc.GetSize()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetType() != dwDesc.GetType() || dyDesc.GetType() != xDesc.GetType()) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetLengths()[0] != xDesc.GetLengths()[0]) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }
    if(dyDesc.GetSize() < 3) {
        MLOPEN_THROW(mlopenStatusBadParm);
    }

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

    if(workSpace == nullptr) {
        MLOPEN_THROW("Workspace is requried");
    }
    switch (algo)
    {
        case mlopenConvolutionBwdWeightsAlgoGEMM:
        {
            std::string network_config;
#if MLOPEN_USE_TINYGEMM
            CreateGemmGeometryConvBwdWeights(dyDesc, xDesc, dwDesc, false, network_config);
            GemmGeometry gg = GetGemmGeometry("mlopenConvolutionBwdWeightsAlgoGEMM", network_config);

            handle.ResetKernelTime();
            float time_0 = 0;
            float t1 = 0;
            for(int i = 0; i < in_n; i++) {
                int out_offset = i * wei_n * out_h * out_w;
                if(wei_h != 1 && wei_w != 1) {
                    size_t in_offset = i * in_c * in_h * in_w;
                    Im2ColGPU(handle, x, in_offset, in_c, in_h, in_w, wei_h, wei_w, out_h, out_w, pad_h, pad_w, v, u, workSpace);
                    if(handle.IsProfilingEnabled())
                        t1 = handle.GetKernelTime();

                    gg.RunGemm(handle, workSpace, dy, dw, 0, out_offset, 0);

                    // Update times for both the kernels
                    if(handle.IsProfilingEnabled()) {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(t1+time_0);
                        else
                            handle.AccumKernelTime(t1);
                        time_0 += handle.GetKernelTime();
                    }
                }
                else if(wei_h == 1 && wei_w == 1) {
                    int in_offset = i * in_c * in_h * in_w;
                    gg.RunGemm(handle, x, dy, dw, in_offset, out_offset, 0);

                    if(handle.IsProfilingEnabled()) {
                        if(i == in_n - 1)
                            handle.AccumKernelTime(time_0);
                        time_0 += handle.GetKernelTime();
                    }
                }
            }
#else
            MLOPEN_THROW("GEMM is not supported");
#endif
        }
        break;

        case mlopenConvolutionBwdWeightsAlgoDirect:
        {

 //           if ((u == 1 && v == 1) || (wei_w >= 7 && (u > 1 || v > 1)))
            {
                mlo_construct_BwdWrW2D construct_params(0); // backward with regards to weights
                construct_params.setOutputDescFromMLDesc(dyDesc);
                construct_params.setInputDescFromMLDesc(xDesc);
                construct_params.setWeightDescFromMLDesc(dwDesc);
                construct_params.mloConstruct();

                std::string network_config;
                construct_params.mloBuildConf_Key(network_config);
                const std::vector<mlo_kernel_info> & bwd_wrw_info = construct_params.getKernelsInfo();

                handle.ResetKernelTime();

                // main kernel
                if (bwd_wrw_info.size() == 1)
                {
                    float padding_val = 0;
                    handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
                            network_config) (dy, x, dw, padding_val);
                }
                else
                {
                    float padding_val = 0;
                    handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Main",
                            network_config) (dy, x, workSpace, padding_val);

                    float time0 = handle.GetKernelTime();
                    // second kernel has
                    network_config += "x1";
                    // reduction  kernel
                    handle.GetKernel("mlopenConvolutionBwdWeightsAlgoDirect_Red",
                            network_config) (workSpace, dw);

                    handle.AccumKernelTime(time0);
                }
            }
        }
        break;
    };
}

}  // namespace mlopen
