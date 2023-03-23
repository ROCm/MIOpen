#ifndef GAURD_MIOPEN_TUNING_METADATA_HPP_
#define GAURD_MIOPEN_TUNING_METADATA_HPP_

#include <unordered_map>

inline std::pair<int, int> get_num_params(std::string solver_name)
{
    if(solver_name == "ConvAsm1x1U")
    {
        std::pair<int, int> num_params = {7, 8};
        return num_params;
    }
    return std::make_pair(0, 0);
}

inline std::unordered_map<int, int> get_decodings(std::string solver_name)
{
    if(solver_name == "ConvAsm1x1U")
    {
        std::unordered_map<int, int> decodings = {
            {1, 4},   {2, 2},   {3, 1},  {4, 3},  {5, 16}, {6, 8},  {7, 1},  {8, 4},  {9, 32},
            {10, 4},  {11, 1},  {12, 2}, {13, 5}, {14, 7}, {15, 3}, {16, 6}, {17, 8}, {18, 64},
            {19, 16}, {20, 32}, {21, 4}, {22, 1}, {23, 1}, {24, 3}, {25, 2}, {26, 4}, {27, 2},
            {28, 4},  {29, 1},  {30, 2}, {31, 1}, {32, 4}, {33, 2}, {34, 4}, {35, 8}, {36, 1}};
        return decodings;
    }
    std::unordered_map<int, int> decodings;
    return decodings;
}

#endif
