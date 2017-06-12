#include <miopen/gemm.hpp>

namespace miopen {
	
GemmGeometry CreateGemmGeometryTranBwdData(
        const TensorDescriptor&     dyDesc,
        const TensorDescriptor&     wDesc,
        const TensorDescriptor&     dxDesc,
        bool                        isDataColMajor,
        std::string                 &network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(dxDesc.GetLengths());

    int wei_c, wei_n, wei_h, wei_w;
    std::tie(wei_c, wei_n, wei_h, wei_w) = tie4(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

    // GEMM
    int K = wei_n * wei_h * wei_w;
    int M = wei_c;
    int N = in_h * in_w;
    float alpha = 1.0;
    float beta = 0.0;
    bool tA = false;
    bool tB = false;
    bool tC = false;
    int lda = K;
    int ldb = N;
    int ldc = N;

    // bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
    TinyGemmGeometry tgg{};
    GemmGeometry gg;
#if 0   
    if (!isDataColMajor) {
        tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{N, M, K}}, 
            std::array<int, 3>{{ldb, lda, ldc}},
            "miopenTransposeBwdDataAlgoGEMM",
            alpha, beta, tgg};
    }
    else 
#endif
	{
        tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{M, N, K}},
            std::array<int, 3>{{lda, ldb, ldc}},
            "miopenTransposeBwdDataAlgoGEMM", 
            alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvBwdData(
	const TensorDescriptor&     dyDesc,
	const TensorDescriptor&     wDesc,
	const TensorDescriptor&     dxDesc,
	bool                        isDataColMajor,
	std::string                 &network_config)
{
	int in_n, in_c, in_h, in_w;
	std::tie(in_n, in_c, in_h, in_w) = tie4(dxDesc.GetLengths());

	int wei_n, wei_h, wei_w;
	std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

	int out_h, out_w;
	std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

	// GEMM
	int K = wei_n;
	int N = out_h * out_w;
	int M = in_c * wei_h * wei_w;
	float alpha = 1.0;
	float beta = 0.0;
	bool tA = true;
	bool tB = false;
	bool tC = false;
	int lda = M;
	int ldb = N;
	int ldc = N;

	// bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
	TinyGemmGeometry tgg{};
	GemmGeometry gg;
	(void)isDataColMajor;
#if 0
	if (!isDataColMajor) {
		tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);
		gg = GemmGeometry{ std::array<int, 3>{ {N, M, K}},
			std::array<int, 3>{ {ldb, lda, ldc}},
			"miopenConvolutionBwdDataAlgoGEMM",
			alpha, beta, tgg };
	}
	else
#endif 
	{
		tgg = TinyGemmGeometry(false, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);

		gg = GemmGeometry{ std::array<int, 3>{ {M, N, K}},
			std::array<int, 3>{ {lda, ldb, ldc}},
			"miopenConvolutionBwdDataAlgoGEMM",
			alpha, beta, tgg };
	}
	network_config = tgg.get_networkconfig_string();
	return gg;
}

GemmGeometry CreateGemmGeometryConvBwdWeights(
        const TensorDescriptor&     dyDesc,
        const TensorDescriptor&     xDesc,
        const TensorDescriptor&     dwDesc,
        bool                        isDataColMajor,
        std::string                 &network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(dwDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(dyDesc.GetLengths());

    // GEMM
    int N = in_c * wei_h * wei_w; 
    int M = wei_n; 
    int K = out_h * out_w;  
    bool tA = false;
    bool tB = true;
    bool tC = false;
    int lda = K;
    int ldb = K;
    int ldc = N;
    float alpha = 1.0;
    float beta = 1.0;

    // bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
    TinyGemmGeometry tgg{};
    GemmGeometry gg;
    
    if (!isDataColMajor) {
        tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{N, M, K}},
            std::array<int, 3>{{ldb, lda, ldc}},
            "miopenConvolutionBwdWeightsAlgoGEMM",
            alpha, beta, tgg};
    }
    else {
        tgg = TinyGemmGeometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{M, N, K}},
            std::array<int, 3>{{lda, ldb, ldc}},
            "miopenConvolutionBwdWeightsAlgoGEMM", 
            alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateGemmGeometryConvFwd(
        const TensorDescriptor&     xDesc,
        const TensorDescriptor&     wDesc,
        const TensorDescriptor&     yDesc,
        bool                        isDataColMajor,
        std::string                 &network_config)
{
    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = tie4(xDesc.GetLengths());

    int wei_n, wei_h, wei_w;
    std::tie(wei_n, std::ignore, wei_h, wei_w) = tie4(wDesc.GetLengths());

    int out_h, out_w;
    std::tie(std::ignore, std::ignore, out_h, out_w) = tie4(yDesc.GetLengths());

    // GEMM
    int K = in_c * wei_h * wei_w;
    int M = wei_n;
    int N = out_h * out_w;
    float alpha = 1.0;
    float beta = 0.0;
    bool tA = false;
    bool tB = false;
    bool tC = false;
    int lda = K;
    int ldb = N;
    int ldc = N;

    // bool isColMajor, bool tA, bool tB, bool tC, lda, ldb, ldc, m, n, k, a_offset, b_offset, c_offset
    TinyGemmGeometry tgg{};
    GemmGeometry gg;
    
    if (!isDataColMajor) {
        tgg = TinyGemmGeometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{N, M, K}}, 
            std::array<int, 3>{{ldb, lda, ldc}},
            "miopenConvolutionFwdAlgoGEMM",
            alpha, beta, tgg};
    }
    else {
        tgg = TinyGemmGeometry(true, tA, tB, tC, lda, ldb, ldc, M, N, K, 0, 0, 0);

        gg = GemmGeometry{std::array<int, 3>{{M, N, K}},
            std::array<int, 3>{{lda, ldb, ldc}},
            "miopenConvolutionFwdAlgoGEMM", 
            alpha, beta, tgg};
    }
    network_config = tgg.get_networkconfig_string();
    return gg;
}

GemmGeometry CreateMIOpenGemmGeometry( 
        int M, int N, int K,
        int lda, int ldb, int ldc,
        bool tA, bool tB,
        bool isDataColMajor,
        float alpha, float beta)
{
    TinyGemmGeometry tgg{};
    
    // Assuming we are using tinygemm as only col major
    // Therefore, if the user provides data in col. major
    // then no transformations are requrired and vice versa
    if(isDataColMajor) {        
        tgg = TinyGemmGeometry(true, tA, tB, false, lda, ldb, ldc, M, N, K, 0, 0, 0);

        return GemmGeometry{std::array<int, 3>{{M, N, K}},
            std::array<int, 3>{{lda, ldb, ldc}},
            "miopenGEMM",
            alpha, beta, tgg};
    }
    else {
        tgg = TinyGemmGeometry(true, tB, tA, false, ldb, lda, ldc, N, M, K, 0, 0, 0);

        return GemmGeometry{std::array<int, 3>{{N, M, K}},
            std::array<int, 3>{{ldb, lda, ldc}}, 
            "miopenGEMM",
            alpha, beta, tgg};
    }
}

GemmGeometry GetGemmGeometry(std::string algorithm_name, std::string network_config)
{
    auto gemm_iterator = gemm_geo_map().find(std::make_pair(algorithm_name, network_config));
    if (gemm_iterator != gemm_geo_map().end())
    {
        return gemm_iterator->second;
    }
    else
    {
        MIOPEN_THROW("looking for gemm kernel (does not exist): " + algorithm_name + ", " + network_config);
    }
}

} // namespace miopen
