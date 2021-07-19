# How to build and run

# Docker
```
docker run                                                                   \
-it                                                                          \
--rm                                                                         \
--privileged                                                                 \
--group-add sudo                                                             \
-w /root/workspace                                                           \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace                                \
rocm/tensorflow:rocm4.2-tf2.4-dev                                            \
/bin/bash
```

# Install Boost for online compilation
https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html#easy-build-and-install


# Build
Change target ID in source code, example below is gfx908
https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/composable_kernel/include/utility/config.amd.hpp.in#L16-L23

Add path of Boost
```
 export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

```
mkdir build && cd build

# need to manually set target ID, example below is gfx908
cmake                                                                                                                              \
-D CMAKE_BUILD_TYPE=Release                                                                                                        \
-D DEVICE_BACKEND=AMD                                                                                                              \
-D CMAKE_CXX_FLAGS="-O3 --amdgpu-target=gfx908 -mllvm --amdgpu-spill-vgpr-to-agpr=0 -gline-tables-only -save-temps=$CWD"           \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                                                          \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                                                     \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                                                  \
..
```

Build drivers:   \
``conv_driver_v2`` is (offline compilation) driver for forward convolution,  \
``conv_bwd_data_driver_v2`` is (offline compilation) driver for backward-data convolution  \
``conv_driver_v2_olc`` is (online compilation) driver for forward convolution
```
 make -j conv_driver_v2
 make -j conv_bwd_data_driver_v2
 make -j conv_driver_v2_olc
```

# Run
* layout: 0 = NCHW; 1 = NHWC
* algo:
   * Forward convolution: https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/driver/conv_driver_v2.cpp#L38
   * Backward data convolution: https://github.com/asroy/modular_convolution/blob/aafb5eb18781f1ac9e06a17c3e53d968dd53dcc0/driver/conv_bwd_data_driver_v2.cpp#L22
* verify: 0 = no verification; 1 = do verification
* init: 0 ~ 3. initialization method
* log: 0 = no log; 1 = do log
* repeat: number of time kernel being launched
```
########################### layout  algo  verify  init  log  repeat  N__ K___ C___ Y X Hi_ Wi__ Strides Dilations LeftPads RightPads
 ./conv_driver_v2                0     6       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
 ./conv_driver_v2                0     6       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 ./conv_driver_v2                1     9       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
 ./conv_driver_v2                1     9       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 ./conv_bwd_data_driver_v2       1     1       0     3    0       1  256  256 1024 3 3  14   14     1 1       1 1      1 1       1 1
```

# Result
Forward convoltuion, FP16, NCHW
```
./conv_driver_v2                0     6       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1

layout: 0
in: dim 4, lengths {128, 192, 71, 71}, strides {967872, 5041, 71, 1}
wei: dim 4, lengths {256, 192, 3, 3}, strides {1728, 9, 3, 1}
out: dim 4, lengths {128, 256, 36, 36}, strides {331776, 1296, 36, 1}
InLeftPads size 2, {1, 1, }
InRightPads size 2, {1, 1, }
ConvStrides size 2, {2, 2, }
ConvDilations size 2, {1, 1, }
device_dynamic_convolution_forward_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw
a_k0_m_k1_grid_desc{216, 256, 8}
b_k0_n_k1_grid_desc{216, 165888, 8}
c_m_n_grid_desc{ 256, 165888}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Average time : 1.4155 ms, 103.686 TFlop/s
```

Forward convoltuion, FP16, NCHW
```
 ./conv_driver_v2                0     6       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 
 layout: 0
in: dim 4, lengths {256, 256, 14, 14}, strides {50176, 196, 14, 1}
wei: dim 4, lengths {1024, 256, 3, 3}, strides {2304, 9, 3, 1}
out: dim 4, lengths {256, 1024, 14, 14}, strides {200704, 196, 14, 1}
InLeftPads size 2, {1, 1, }
InRightPads size 2, {1, 1, }
ConvStrides size 2, {1, 1, }
ConvDilations size 2, {1, 1, }
device_dynamic_convolution_forward_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw
a_k0_m_k1_grid_desc{288, 1024, 8}
b_k0_n_k1_grid_desc{288, 50176, 8}
c_m_n_grid_desc{ 1024, 50176}
launch_and_time_kernel: grid_dim {1568, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Average time : 2.21357 ms, 106.959 TFlop/s
 ```
 
 Forward convolution, FP16, NHWC
 ```
 ./conv_driver_v2                1     9       0     3    0       1  128  256  192 3 3  71   71     2 2       1 1      1 1       1 1
 
 layout: 1
in: dim 4, lengths {128, 71, 71, 192}, strides {967872, 13632, 192, 1}
wei: dim 4, lengths {256, 3, 3, 192}, strides {1728, 576, 192, 1}
out: dim 4, lengths {128, 36, 36, 256}, strides {331776, 9216, 256, 1}
InLeftPads size 2, {1, 1, }
InRightPads size 2, {1, 1, }
ConvStrides size 2, {2, 2, }
ConvDilations size 2, {1, 1, }
device_dynamic_convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk
a_k0_m_k1_grid_desc{216, 165888, 8}
b_k0_n_k1_grid_desc{216, 256, 8}
c_m_n_grid_desc{ 165888, 256}
launch_and_time_kernel: grid_dim {1296, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Average time : 1.12014 ms, 131.025 TFlop/s
 ```
 
 Forward convolution, FP16, NHWC
 ```
 ./conv_driver_v2                1     9       0     3    0       1  256 1024  256 3 3  14   14     1 1       1 1      1 1       1 1
 
 layout: 1
in: dim 4, lengths {256, 14, 14, 256}, strides {50176, 3584, 256, 1}
wei: dim 4, lengths {1024, 3, 3, 256}, strides {2304, 768, 256, 1}
out: dim 4, lengths {256, 14, 14, 1024}, strides {200704, 14336, 1024, 1}
InLeftPads size 2, {1, 1, }
InRightPads size 2, {1, 1, }
ConvStrides size 2, {1, 1, }
ConvDilations size 2, {1, 1, }
device_dynamic_convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk
a_k0_m_k1_grid_desc{288, 50176, 8}
b_k0_n_k1_grid_desc{288, 1024, 8}
c_m_n_grid_desc{ 50176, 1024}
launch_and_time_kernel: grid_dim {1568, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Average time : 1.86877 ms, 126.693 TFlop/s
 ```
 
 Backward data convolution, FP16, NHWC
 ```
 ./conv_bwd_data_driver_v2       1     1       0     3    0       1  256  256 1024 3 3  14   14     1 1       1 1      1 1       1 1
 
 layout: 1
in: dim 4, lengths {256, 14, 14, 1024}, strides {200704, 14336, 1024, 1}
wei: dim 4, lengths {256, 3, 3, 1024}, strides {9216, 3072, 1024, 1}
out: dim 4, lengths {256, 14, 14, 256}, strides {50176, 3584, 256, 1}
InLeftPads size 2, {1, 1, }
InRightPads size 2, {1, 1, }
ConvStrides size 2, {1, 1, }
ConvDilations size 2, {1, 1, }
device_dynamic_convolution_backward_data_implicit_gemm_v4r1r2_xdlops_nhwc_kyxc_nhwk
a_k0_m_k1_grid_desc{288, 50176, 8}
b_k0_n_k1_grid_desc{288, 1024, 8}
c_m_n_grid_desc{ 50176, 1024}
launch_and_time_kernel: grid_dim {1568, 1, 1}, block_dim {256, 1, 1}
Warm up
Start running 1 times...
Average time : 2.22461 ms, 106.428 TFlop/s
```
