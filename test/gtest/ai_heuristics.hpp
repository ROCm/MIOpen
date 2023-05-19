#include <miopen/miopen.h>
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <miopen/solver.hpp>
#include "get_handle.hpp"
#include <gtest/cba.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

struct AIModelTestCase
{
    struct ConvTestCase conv;
    miopen::conv::Direction direction;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
};
