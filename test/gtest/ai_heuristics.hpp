#include <miopen/miopen.h>
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/solver.hpp>
#include "get_handle.hpp"
#include <unordered_map>
#include <gtest/cba.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

struct AIModelTestCase
{
    struct ConvTestCase conv;
    miopen::conv::Direction direction;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
};
