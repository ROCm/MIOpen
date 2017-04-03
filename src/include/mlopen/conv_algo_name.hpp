#ifndef GUARD_MLOPEN_CONV_ALGO_NAME_HPP
#define GUARD_MLOPEN_CONV_ALGO_NAME_HPP

#include <string>
#include <unordered_map>

namespace mlopen {
 
inline int FwdAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"mlopenConvolutionFwdAlgoGEMM", 0},
    	{"mlopenConvolutionFwdAlgoDirect", 1},
    	{"mlopenConvolutionFwdAlgoFFT", 2},
    	{"mlopenConvolutionFwdAlgoWinograd", 3},
    };
    return data.at(s);
}

inline int BwdDataAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"mlopenConvolutionBwdDataAlgoDirect", 0},
        {"mlopenConvolutionBwdDataAlgoWinograd", 1},
    };
    return data.at(s);
}

inline int BwdWeightsAlgoResolver(const std::string& s) {
    static std::unordered_map<std::string, int> data {
        {"mlopenConvolutionBwdWeightsAlgoGEMM", 0},
    	{"mlopenConvolutionBwdWeightsAlgoDirect", 1},
    };
    return data.at(s);
}

} // namespace mlopen

#endif // GUARD_MLOPEN_CONV_ALGO_NAME_HPP
