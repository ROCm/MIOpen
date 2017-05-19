#ifndef GUARD_MIOPEN_CONV_ALGO_NAME_HPP
#define GUARD_MIOPEN_CONV_ALGO_NAME_HPP

#include <string>
#include <unordered_map>

namespace miopen {
 
inline int FwdAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"miopenConvolutionFwdAlgoGEMM", 0},
    	{"miopenConvolutionFwdAlgoDirect", 1},
    	{"miopenConvolutionFwdAlgoFFT", 2},
    	{"miopenConvolutionFwdAlgoWinograd", 3},
    };
    return data.at(s);
}

inline int BwdDataAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"miopenConvolutionBwdDataAlgoDirect", 0},
        {"miopenConvolutionBwdDataAlgoWinograd", 1},
		{"miopenConvolutionBwdDataAlgoFFT", 2},
    };
    return data.at(s);
}

inline int BwdWeightsAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"miopenConvolutionBwdWeightsAlgoGEMM", 0},
    	{"miopenConvolutionBwdWeightsAlgoDirect", 1},
    };
    return data.at(s);
}

} // namespace miopen

#endif // GUARD_MIOPEN_CONV_ALGO_NAME_HPP
