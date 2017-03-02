#ifndef GUARD_MLOPEN_CONV_ALGO_NAME_HPP
#define GUARD_MLOPEN_CONV_ALGO_NAME_HPP

#include <string>
#include <unordered_map>

namespace mlopen {
    
static std::unordered_map<std::string, int> FwdAlgoResolver {
    {"mlopenConvolutionFwdAlgoGEMM", 0},
	{"mlopenConvolutionFwdAlgoDirect", 1},
	{"mlopenConvolutionFwdAlgoFFT", 2},
	{"mlopenConvolutionFwdAlgoWinograd", 3},
};

static std::unordered_map<std::string, int> BwdDataAlgoResolver {
    {"mlopenConvolutionBwdDataAlgo_0", 0},
};

static std::unordered_map<std::string, int> BwdWeightsAlgoResolver {
    {"mlopenConvolutionBwdWeightsAlgoGEMM", 0},
	{"mlopenConvolutionBwdWeightsAlgoDirect", 1},
};

} // namespace mlopen

#endif // GUARD_MLOPEN_CONV_ALGO_NAME_HPP
