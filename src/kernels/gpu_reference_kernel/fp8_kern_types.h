#pragma once

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)

#ifndef INPUT_TYPE
#define INPUT_TYPE half
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE half
#endif

#ifndef WEIGHTS_TYPE
#define WEIGHTS_TYPE half
#endif

#ifndef INPUT_CAST_TYPE
#define INPUT_CAST_TYPE float8
#endif

#ifndef WEIGHTS_CAST_TYPE
#define WEIGHTS_CAST_TYPE float8
#endif

#ifndef OUTPUT_CAST_TYPE
#define OUTPUT_CAST_TYPE float8
#endif

#ifndef ACCUMULATOR_TYPE
#define ACCUMULATOR_TYPE double
#endif

#define KERNEL_NAME_SUFFIX CAT(CAT(INPUT_TYPE, _), CAT(CAT(WEIGHTS_TYPE, _), OUTPUT_TYPE))

#define FWD_KERNEL_NAME CAT(naive_conv_fwd_nchw_, KERNEL_NAME_SUFFIX)
#define BWD_KERNEL_NAME CAT(naive_conv_bwd_nchw_, KERNEL_NAME_SUFFIX)
#define WRW_KERNEL_NAME CAT(naive_conv_wrw_nchw_, KERNEL_NAME_SUFFIX)