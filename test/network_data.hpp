/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TEST_NETWORK_DATA_HPP
#define GUARD_MIOPEN_TEST_NETWORK_DATA_HPP

#include <initializer_list>
#include <set>
#include <vector>

#ifndef MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR
#define MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR 0
#endif

inline int pick_batch_size(int x, int y)
{
    if(y == 0 || y > x)
        return 1;
    else
        return x / y;
}

// Reduce tests execution time
#define MIOPEN_TESTS_GET_INPUTS_ENABLE_HUGE_TENSORS 1

inline std::set<std::vector<int>> get_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32,  n), 1,    14,  14  },
        { pick_batch_size(100, n), 1,    8,   8   },
        { pick_batch_size(256, n), 1,    27,  27  },
#if MIOPEN_TESTS_GET_INPUTS_ENABLE_HUGE_TENSORS
        { pick_batch_size(64, n), 19,   1024,2048},
#endif
        { pick_batch_size(100, n), 3,    32,  32  },
        { pick_batch_size(100, n), 32,   16,  16  },
        { pick_batch_size(100, n), 32,   8,   8   },
        { pick_batch_size(128, n), 256,  12,  12  },
        { pick_batch_size(128, n), 3,    231, 231 },
        { pick_batch_size(128, n), 512,  12,  12  },
        { pick_batch_size(256, n), 256,  13,  13  },
        { pick_batch_size(256, n), 3,    227, 227 },
        { pick_batch_size(256, n), 384,  13,  13  },
        { pick_batch_size(256, n), 96,   27,  27  },
        { pick_batch_size(32, n),  128,  28,  28  },
        { pick_batch_size(32, n),  144,  14,  14  },
        { pick_batch_size(32, n),  192,  28,  28  },
        { pick_batch_size(32, n),  192,  7,   7   },
        { pick_batch_size(32, n),  256,  28,  28  },
        { pick_batch_size(32, n),  3,    224, 224 },
        { pick_batch_size(32, n),  32,   28,  28  },
        { pick_batch_size(32, n),  48,   7,   7   },
        { pick_batch_size(32, n),  480,  128, 256 },
        { pick_batch_size(32, n),  480,  64,  128 },
        { pick_batch_size(32, n),  512,  4,   4   },
        { pick_batch_size(32, n),  512,  64,  128 },
        { pick_batch_size(16, n),  64,   56,  56  },
        { pick_batch_size(32, n),  832,  7,   7   },
        { pick_batch_size(64, n),  128,  56,  56  },
        { pick_batch_size(64, n),  256,  28,  28  },
        { pick_batch_size(64, n),  3,    224, 224 },
        { pick_batch_size(64, n),  512,  28,  28  },
        { pick_batch_size(64, n),  64,   112, 112 },
        { pick_batch_size(32, n),  64,   14,  14  },
        { pick_batch_size(32, n),  192,  14,  14  },
        { pick_batch_size(32, n),  320,  28,  28  },
        { pick_batch_size(32, n),  576,  14,  14  },
        { pick_batch_size(32, n),  576,  4,   4   },
        { pick_batch_size(32, n),  1056, 7,   7   },
        { pick_batch_size(32, n),  2048, 11,  11  },
#if MIOPEN_TESTS_GET_INPUTS_ENABLE_HUGE_TENSORS
        { pick_batch_size(32, n),  16,   2048, 2048 },
        { pick_batch_size(32, n),  16,   3072, 3072 },
        { pick_batch_size(32, n),  16,   4096, 4096 },
#endif
        { 1,                       1,    1,   1   }
    };
    // clang-format on
}

inline std::set<std::vector<int>> get_weights(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(1024, n),1024, 3,  3  },
        { pick_batch_size(1024, n),512,  3,  3  },
        { pick_batch_size(128, n), 256,  1,  1  },
        { pick_batch_size(128, n), 528,  1,  1  },
        { pick_batch_size(128, n), 96,   3,  3  },
        { pick_batch_size(16, n),  192,  1,  1  },
        { pick_batch_size(224, n), 112,  3,  3  },
        { pick_batch_size(256, n), 96,   5,  5  },
        { pick_batch_size(288, n), 144,  3,  3  },
        { pick_batch_size(48, n),  832,  1,  1  },
        { pick_batch_size(512, n), 256,  3,  3  },
        { pick_batch_size(64, n),  1,    2,  2  },
        { pick_batch_size(64, n),  3,    3,  3  },
        { pick_batch_size(64, n),  3,    7,  7  },
        { pick_batch_size(64, n),  32,   5,  5  },
        { pick_batch_size(64, n),  480,  1,  1  },
        { pick_batch_size(64, n),  64,   1,  1  },
        { pick_batch_size(96, n),  3,    11, 11 },
        { pick_batch_size(192, n), 64,   5,  5  },
        { pick_batch_size(64, n),  64,   3,  3  },
        { pick_batch_size(224, n), 224,  3,  3  },
        { pick_batch_size(224, n), 192,  3,  3  },
        { pick_batch_size(128, n), 320,  1,  1  },
        { pick_batch_size(192, n), 576,  1,  1  },
        { pick_batch_size(128, n), 1056, 1,  1  },
        { pick_batch_size(128, n), 1024, 1,  1  },
        { pick_batch_size(512, n), 2048, 1,  1  }
    };
    // clang-format on
}

inline std::set<std::vector<int>> get_immed_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32,  n), 1,    14,  14  },
        { pick_batch_size(256, n), 1,    27,  27  },
        { pick_batch_size(128, n), 512,  12,  12  },
        { pick_batch_size(256, n), 256,  13,  13  },
        { pick_batch_size(256, n), 3,    227, 227 },
        { pick_batch_size(32, n),  64,   56,  56  },
        { pick_batch_size(32, n),  96,   14,  14  },
        { pick_batch_size(32, n),  96,   28,  28  },
        { pick_batch_size(64, n),  128,  56,  56  },
        { pick_batch_size(64, n),  3,    224, 224 },
        { pick_batch_size(64, n),  256,  14,  14  },
        { 1,                       1,    1,   1   }
    };
    // clang-format on
}

inline std::set<std::vector<int>> get_immed_weights(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(208, n), 96,   3,  3  },
        { pick_batch_size(24, n),  512,  1,  1  },
        { pick_batch_size(256, n), 128,  3,  3  },
        { pick_batch_size(256, n), 256,  3,  3  },
        { pick_batch_size(256, n), 64,   5,  5  },
        { pick_batch_size(288, n), 144,  3,  3  },
        { pick_batch_size(96, n),  3,    11, 11 },
        { pick_batch_size(32, n),  128,   5,  5  },
        { pick_batch_size(32, n),  128,  1,  1  },
        { pick_batch_size(256, n), 256,  3,  3  },
        { pick_batch_size(512, n), 512,  3,  3  },
        { pick_batch_size(160, n), 128,  3,  3  },
        { pick_batch_size(32, n),  3,    7,  7  }
    };
    // clang-format on
}

inline std::set<std::vector<int>>
get_3d_conv_input_shapes(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(128, n),   1,   1,   2,   2},
        { pick_batch_size(128, n),  64,   1,   1,   1},
        { pick_batch_size(128, n),  64,   3,   4,   4},
        { pick_batch_size(352, n),  32,   4,   9,   9},
        { pick_batch_size(192, n), 512,   3,  14,  14},
        { pick_batch_size(352, n), 512,   4,  28,  28},
        { pick_batch_size(256, n), 512,   4,  56,  56},
        { pick_batch_size(192, n),   3,   4, 227, 227},
        { pick_batch_size(128, n),   4,   4, 161, 700}
    };
    // clang-format on
}

inline std::set<std::vector<int>>
get_3d_conv_weight_shapes(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size( 128, n),   1,   1,   1,   1},
        { pick_batch_size( 352, n), 128,   1,   1,   1},
        { pick_batch_size( 256, n), 128,   1,   1,   1},
        { pick_batch_size( 352, n),  32,   3,   3,   3},
        { pick_batch_size( 352, n),   4,   3,   3,   3},
        { pick_batch_size( 160, n),   4,   3,   5,   5},
        { pick_batch_size( 128, n),  64,   5,   7,   7},
        { pick_batch_size( 192, n),   4,   3,  11,  11},
        { pick_batch_size( 128, n),   1,   3,   1,   7},
        { pick_batch_size( 128, n),   1,   3,   7,   1},
        { pick_batch_size( 128, n),   1,   3,   5,  20}
    };
    // clang-format on
}

inline std::set<std::vector<int>>
get_bn_peract_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32, n),  4,    1024,2048}, //Making this much smaller
        { pick_batch_size(100, n), 3,    32,  32  },
        { pick_batch_size(100, n), 32,   8,   8   },
        { pick_batch_size(128, n), 256,  12,  12  },
        { pick_batch_size(256, n), 3,    227, 227 },
        { pick_batch_size(64, n),  64,   112, 112 },//Batch-norm ResNet 152 after this line
        { pick_batch_size(256, n), 1024, 14,  14  },// n is from the paper @ 256
        { pick_batch_size(256, n), 2048, 7,   7   },
        { pick_batch_size(256, n), 256,  56,  56  },
        { pick_batch_size(256, n), 256,  14,  14  },
        { pick_batch_size(256, n), 512,  28,  28  },
        { pick_batch_size(256, n), 512,  7,   7   },
        { pick_batch_size(256, n), 64,   112, 112 },
        { pick_batch_size(256, n), 64,   56,  56  },//Batch-norm Inception_v3 after this
        { pick_batch_size(32, n),  1024, 1,   1   },// n is from the paper @ 32
        { pick_batch_size(32, n),  128,  14,  14  },
        { pick_batch_size(32, n),  128,  28,  28  },
        { pick_batch_size(32, n),  128,  4,   4   },
        { pick_batch_size(32, n),  128,  7,   7   },
        { pick_batch_size(32, n),  160,  7,   7   },
        { pick_batch_size(32, n),  192,  14,  14  },
        { pick_batch_size(32, n),  192,  56,  56  },
        { pick_batch_size(32, n),  192,  7,   7   },
        { pick_batch_size(32, n),  224,  14,  14  },
        { pick_batch_size(32, n),  256,  7,   7   },
        { pick_batch_size(32, n),  256,  14,  14  },
        { pick_batch_size(32, n),  352,  7,   7   },
        { pick_batch_size(32, n),  64,   112, 112 },
        { pick_batch_size(32, n),  64,   14,  14  },
        { pick_batch_size(32, n),  64,   56,  56  },
        { pick_batch_size(32, n),  96,   28,  28  },
        { pick_batch_size(32, n),  32,  256,  512 }, //Killing this config. Takes way too long on the CPU
        { pick_batch_size(32, n),  256,  28,  28  },
        { pick_batch_size(32, n),  3,    224, 224 },
        { pick_batch_size(32, n),  480,  128, 256 },
        { pick_batch_size(32, n),  528,  64,  128 }
    };
    // clang-format on
}

inline std::set<std::vector<int>>
get_bn_spatial_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32, n),  4,    1024,2048}, //Making this much smaller
        { pick_batch_size(32, n),  192,  256, 512 },
        { pick_batch_size(32, n),  480,  128, 256 },
        { pick_batch_size(256, n), 3,    227, 227 },
        { pick_batch_size(256, n), 64,   112, 112 },
        { pick_batch_size(512, n), 16,   32,  32  },
        { pick_batch_size(100, n), 32,   8,   8   },
        { pick_batch_size(128, n), 256,  12,  12  },
        { pick_batch_size(256, n), 128,  28,  28  },
        { pick_batch_size(256, n), 2048, 7,   7   },
        { pick_batch_size(256, n), 256,  56,  56  },
        { pick_batch_size(256, n), 256,  14,  14  },
        { pick_batch_size(256, n), 512,  28,  28  },
        { pick_batch_size(256, n), 512,  7,   7   },
        { pick_batch_size(256, n), 64,   56,  56  },//Batch-norm Inception_v3 after this
        { pick_batch_size(32, n),  1024, 1,   1   },// n is from the paper @ 32
        { pick_batch_size(32, n),  128,  14,  14  },
        { pick_batch_size(32, n),  128,  4,   4   },
        { pick_batch_size(32, n),  160,  7,   7   },
        { pick_batch_size(32, n),  192,  14,  14  },
        { pick_batch_size(32, n),  192,  56,  56  },
        { pick_batch_size(32, n),  192,  7,   7   },
        { pick_batch_size(32, n),  224,  14,  14  },
        { pick_batch_size(32, n),  256,  7,   7   },
        { pick_batch_size(32, n),  352,  7,   7   },
        { pick_batch_size(32, n),  64,   14,  14  },
        { pick_batch_size(32, n),  64,   28,  28  },
        { pick_batch_size(32, n),  64,   56,  56  },
        { pick_batch_size(32, n),  96,   28,  28  },
        { pick_batch_size(32, n),  192,  256, 512 },
        { pick_batch_size(32, n),  256,  28,  28  },
        { pick_batch_size(32, n),  3,    224, 224 },
        { pick_batch_size(32, n),  480,  128, 256 },
        { pick_batch_size(32, n),  528,  64,  128 },
        { pick_batch_size(770, n),  1,  8,  8 },
        { pick_batch_size(770, n),  1024,  1,  1 },
        { pick_batch_size(152, n),  128,  80,  80 },
        { pick_batch_size(152, n),  256,  20,  20 },
        { pick_batch_size(152, n),  32,  160,  160 },
        { pick_batch_size(152, n),  512,  20,  20 },
        { pick_batch_size(152, n),  64,  160,  160 },
        { pick_batch_size(152, n),  64,  80,  80 },
        { pick_batch_size(256, n),  256,  20,  20 },
        { pick_batch_size(256, n),  512,  20,  20 }
    };
    // clang-format on
}

inline std::set<std::vector<int>>
get_3d_bn_peract_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32, n),   1,   32,  32,  32  },       // 32x32x32 based on VoxNet arch
        { pick_batch_size(32, n),   1,   14,  14,  14  },
        { pick_batch_size(32, n),  32,   14,  14,  14  },
        { pick_batch_size(32, n),  32,   12,  12,  12  },
        { pick_batch_size(32, n),  32,    6,   6,   6  },
        { pick_batch_size(256, n),  1,   32,  32,  32  },      // 32x32x32 based on VoxNet arch
        { pick_batch_size(256, n), 32,   14,  14,  14  },
        { pick_batch_size(256, n), 32,   12,  12,  12  },
        { pick_batch_size(256, n), 32,    6,   6,   6  },        
        { pick_batch_size(512, n),  1,   32,  32,  32  },      // 32x32x32 based on VoxNet arch
        { pick_batch_size(512, n), 32,   14,  14,  14  },
        { pick_batch_size(512, n), 32,   12,  12,  12  },
        { pick_batch_size(512, n), 32,    6,   6,   6  },                
        { pick_batch_size(32, n),   2,   32,  57, 125  },       // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { pick_batch_size(32, n),  32,   14,  25,  59  },
        { pick_batch_size(32, n),  32,    6,  10,  27  },
        { pick_batch_size(32, n),  32,    4,   6,  11  },                        
        { pick_batch_size(32, n),  32,    2,   2,   3  },                        
        { pick_batch_size(32, n),  32,   32,  28,  62  },       // Hand-gesture recognition CVPR 2015 paper Low Res Net Path 
        { pick_batch_size(32, n),  32,   14,  12,  29  },
        { pick_batch_size(32, n),  32,    6,   4,  12  },                        
        { pick_batch_size(32, n),  32,    4,   2,   2  },                        
        { pick_batch_size(16, n),  32,    6,  50,  50  },       // Multi-view 3D convnet
        { pick_batch_size(1,  n),   3,    8, 240, 320  },      // 3D convet on video
        { pick_batch_size(1,  n),   3,   16, 240, 320  },      // 3D convet on video
        { pick_batch_size(1,  n),   3,    8, 128, 171  },      // 3D convet on video
        { pick_batch_size(1,  n),   3,   16, 128, 171  },      // 3D convet on video
        { pick_batch_size(1,  n),   3,    8, 112, 112  },      // 3D convet on video
        { pick_batch_size(1,  n),   3,   16, 112, 112  }      // 3D convet on video
    };

    // clang-format on
}

inline std::set<std::vector<int>>
get_3d_bn_spatial_inputs(int n = MIOPEN_TEST_DEFAULT_BATCH_SIZE_FACTOR)
{
    // clang-format off
    return 
    {
        { pick_batch_size(32, n),   1,   32,  32,  32  },       // 32x32x32 based on VoxNet arch
        { pick_batch_size(32, n),   1,   14,  14,  14  },
        { pick_batch_size(32, n),  32,   14,  14,  14  },
        { pick_batch_size(32, n),  32,   12,  12,  12  },
        { pick_batch_size(32, n),  32,    6,   6,   6  },
        { pick_batch_size(256, n),  1,   32,  32,  32  },      // 32x32x32 based on VoxNet arch
        { pick_batch_size(256, n), 32,   14,  14,  14  },
        { pick_batch_size(256, n), 32,   12,  12,  12  },
        { pick_batch_size(256, n), 32,    6,   6,   6  },        
        { pick_batch_size(512, n),  1,   32,  32,  32  },      // 32x32x32 based on VoxNet arch
        { pick_batch_size(512, n), 32,   14,  14,  14  },
        { pick_batch_size(512, n), 32,   12,  12,  12  },
        { pick_batch_size(512, n), 32,    6,   6,   6  },                
        { pick_batch_size(32,  n),  2,   32,  57, 125  },       // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { pick_batch_size(32,  n), 32,   14,  25,  59  },
        { pick_batch_size(32,  n), 32,    6,  10,  27  },
        { pick_batch_size(32,  n), 32,    4,   6,  11  },                        
        { pick_batch_size(32,  n), 32,    2,   2,   3  },                        
        { pick_batch_size(32,  n), 32,   32,  28,  62  },       // Hand-gesture recognition CVPR 2015 paper Low Res Net Path 
        { pick_batch_size(32,  n), 32,   14,  12,  29  },
        { pick_batch_size(32,  n), 32,    6,   4,  12  },                        
        { pick_batch_size(32,  n), 32,    4,   2,   2  },                        
        { pick_batch_size(16,  n), 32,    6,  50,  50  },       // Multi-view 3D convnet
        { pick_batch_size(1,   n), 3,     8,  240, 320 },      // 3D convet on video
        { pick_batch_size(1,   n), 3,    16,  240, 320 },      // 3D convet on video
        { pick_batch_size(1,   n), 3,     8,  128, 171 },      // 3D convet on video
        { pick_batch_size(1,   n), 3,    16,  128, 171 },      // 3D convet on video
        { pick_batch_size(1,   n), 3,     8,  112, 112 },      // 3D convet on video
        { pick_batch_size(1,   n), 3,    16,  112, 112 }      // 3D convet on video
    };
    // clang-format on
}

inline std::vector<std::vector<int>> get_sub_tensor()
{
    return {{16, 4, 8, 1, 4},
            {2, 4, 8, 8, 4},
            {16, 4, 8, 4},
            {13, 8, 4, 8},
            {3, 8, 7},
            {16, 4, 10},
            {3, 8},
            {16, 4},
            {4}};
}

inline std::vector<std::vector<int>> get_tensor_offsets()
{
    return {{0, 0}, {0, 2}, {4, 0}, {5, 7}};
}

inline std::vector<int> get_tensor_offset() { return {0, 1, 2, 3, 4, 5}; }

#endif
