/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include <vector>
#include <string>
#include <assert.h>
#include <chrono>
#include <functional>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <vector>
#include <float.h>
#include <cmath>
#include <iostream>
#include "gpu_tensor_reorder.h"
#include "sequence.hpp"


#ifndef HIP_CALL
#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("[hiperror](%d) fail to call %s,(%s)\n", (int)err, #call,     \
                   hipGetErrorString(err));                                    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

static inline int env_get_int(const char *var_name, int default_int) {
    char *v = getenv(var_name);
    int r = default_int;
    if (v)
        r = atoi(v);
    return r;
}

static int gen_rand_integer()
{
    static int inited = 0;
    if(inited == 0)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    return std::rand();
}


static inline char *env_get_str(char *var_name, char* default_str) {
    char *v = getenv(var_name);
    if (v)
        return v;
    return default_str;
}

template <typename T>
struct distribution_t{
};

template <>
struct distribution_t<int8_t>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int8_t operator()(URNG & rng){
        int value = distribution(rng);
        return *reinterpret_cast<int8_t*>(&value);
        //return 0xf;
    }
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<int>{
    distribution_t(int min, int max) : distribution(min, max) {}
    template<class URNG>
    int operator()(URNG & rng){ return distribution(rng);}
    std::uniform_int_distribution<int> distribution;
};
template <>
struct distribution_t<float>{
    distribution_t(float min, float max) : distribution(min, max) {}
    template<class URNG>
    float operator()(URNG & rng){ return distribution(rng);}
    std::uniform_real_distribution<float> distribution;
};

template <typename Dst_T, typename Src_T>
void block_wise_rand_generator(Dst_T *p, int tid, int block_size, int total_size, Src_T min, Src_T max, Src_T scale)
{
    std::mt19937 rng(std::chrono::system_clock::now()
                        .time_since_epoch()
                        .count() +
                    std::hash<std::thread::id>()(std::this_thread::get_id()));
    distribution_t<Dst_T> distribution(min,max);
    for (int i = tid; i < total_size; i += block_size) {
        p[i] = static_cast<Dst_T>(scale * distribution(rng));
    }
}

template <typename Dst_T, typename Src_T>
void gen_rand_vector(Dst_T *vec, size_t vec_size, Src_T fmin, Src_T fmax, Src_T scale = 1) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads < 4)
        num_threads = 4;
    // printf("total threads:%d\n",num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(block_wise_rand_generator<Dst_T, Src_T>,
            vec, t, num_threads, vec_size, fmin, fmax, scale));
    }
    for (auto &th : threads)
        th.join();
}

static inline bool valid_float(float p)
{
    return !(std::isnan(p) || std::isinf(p));
}
#ifndef ABS
#define ABS(b) ((b) > 0 ? (b) : -1 * (b))
#endif
static inline bool valid_vector(const float *ref, const float *pred, int n,
                                double nrms = 1.5e-6) {
    double s0 = 0.0;
    double s1 = 0.0;
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    int igemm_per_pixel_check_print = env_get_int("PER_PIXEL_CHECK_PRINT", 1);
    int pp_err = 0;

    for (int i = 0; i < n; ++i) {
        if(!(valid_float(ref[i]) && valid_float(pred[i]))){
            printf(" invalid float at %d, ref:%f, pred:%f\n", i, ref[i], pred[i]);
            return -1;
        }
        double ri = (double)ref[i];
        double pi = (double)pred[i];
        double d = ri - pi;
        double dd = d * d;
        double rr = 2.0 * ri * ri;
        s0 += dd;
        s1 += rr;
        if(igemm_per_pixel_check){
            double delta = ABS(ABS(ri - pi) / ri);
            printf("[%d] ref:%lf, pred:%lf(0x%08x) [%s]\n", i, ri, pi, ((uint32_t *)pred)[i], delta > 3e-5? "N":"Y");
            if (delta > 3e-5) {
                if(igemm_per_pixel_check_print){
                    if (pp_err < 100)
                        printf("diff at %d, ref:%lf, pred:%lf(0x%08x), d:%lf\n", i, ri,
                            pi, ((uint32_t *)pred)[i], delta);
                }
                pp_err++;
            }

        }
    }
    // printf("\nnrms:%lf, s0:%lf, s1:%lf, expected_nrms is %1f\n",sqrt(s0/s1),s0,s1,nrms);
    fflush(stdout);
    return (sqrt(s0 / s1) < nrms)
#ifdef PER_PIXEL_CHECK
           && (pp_err == 0)
#endif
        ;
}

static inline bool valid_vector_binary(int8_t *ref, int8_t *pred, size_t bytes) {
    int igemm_per_pixel_check = env_get_int("PER_PIXEL_CHECK", 0);
    size_t err = 0;
    for(int i = 0; i < bytes ; i++){
        // {
        //     uint32_t r = 0;
        //     uint32_t p = 0;
        //     memcpy(reinterpret_cast<void*>(&r), reinterpret_cast<void*>(&ref[i]), 1);
        //     memcpy(reinterpret_cast<void*>(&p), reinterpret_cast<void*>(&pred[i]), 1);
        //     printf("%7d, ref:0x%x, pred:0x%x, %s\n", i, r, p, r==p?"y":"n");
        // }
        if(ref[i] != pred[i]){
            err ++;
            if(igemm_per_pixel_check){
                uint32_t r = 0;
                uint32_t p = 0;
                memcpy(reinterpret_cast<void*>(&r), reinterpret_cast<void*>(&ref[i]), 1);
                memcpy(reinterpret_cast<void*>(&p), reinterpret_cast<void*>(&pred[i]), 1);
                printf("fail at %d, ref:0x%x, pred:0x%x\n", i, r, p);
            }
        }
    }
    return err == 0;
}

template<typename T,
         typename dst_order>
void cpu_tensor_reorder(T * dst, T * src, uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, uint64_t dim_3)
{
    constexpr auto dorder = dst_order{};
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    
    const uint64_t src_stride[4]   ={src_dim[1] * src_dim[2] * src_dim[3], 
                                     src_dim[2] * src_dim[3], 
                                     src_dim[3],
                                     1 };
    const uint64_t dst_stride[4]  = {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                     dst_dim[2] * dst_dim[3], 
                                     dst_dim[3],
                                     1 };

    uint64_t itr_src_dim[4]  = {0, 0, 0, 0};
    uint64_t itr_dst_dim[4] = {0, 0, 0, 0};

    for(itr_src_dim[0] = 0; itr_src_dim[0] < src_dim[0]; itr_src_dim[0]++){
        for(itr_src_dim[1] = 0; itr_src_dim[1] < src_dim[1]; itr_src_dim[1]++){
            for(itr_src_dim[2] = 0; itr_src_dim[2] < src_dim[2]; itr_src_dim[2]++){
                for(itr_src_dim[3] = 0; itr_src_dim[3] < src_dim[3]; itr_src_dim[3]++){
                    itr_dst_dim[0] = itr_src_dim[dorder.at(0)];
                    itr_dst_dim[1] = itr_src_dim[dorder.at(1)];
                    itr_dst_dim[2] = itr_src_dim[dorder.at(2)];
                    itr_dst_dim[3] = itr_src_dim[dorder.at(3)];

                    uint64_t idx_src =   itr_src_dim[0] * src_stride[0] +
                                         itr_src_dim[1] * src_stride[1] +
                                         itr_src_dim[2] * src_stride[2] +
                                         itr_src_dim[3] * src_stride[3] ;
                    uint64_t idx_dst =   itr_dst_dim[0] * dst_stride[0] + 
                                         itr_dst_dim[1] * dst_stride[1] +
                                         itr_dst_dim[2] * dst_stride[2] +
                                         itr_dst_dim[3] * dst_stride[3] ;
                    
                    dst[idx_dst] = src[idx_src]; 
                }
            }
        }
    }
}

//compile time for_loop
namespace detail {

    template<class T, T... inds, class F>
    constexpr void loop(std::integer_sequence<T, inds...>, F&& f) {
        (f(std::integral_constant<T, inds>{}), ...);// C++17 fold expression
    }

}

template<class T, T count, class F>
constexpr void loop(F&& f) {
    detail::loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

#define WARMUP 3
#define REPEAT 7
#define BATCHED_TRANSPOSE_HSACO "out/batched_transpose.hsaco"
#define GENERAL_TENSOR_REORDER_HSACO    "out/general_tensor_reorder.hsaco"

int main(int argc, char ** argv){
    if(argc < 5){
        printf("%s Please input tensor size in order of： DIM0, DIM1, DIM2, DIM3\n", argv[0]);
        return -1;
    }
    if(argc > 5){
        printf("Too many argument\n");
        return -1;
    }
    int warmup = env_get_int("IGEMM_WARMUP", WARMUP);
    int repeat = env_get_int("IGEMM_REPEAT", REPEAT);
    const uint64_t dim_0 = std::stoull(std::string(argv[1]));
    const uint64_t dim_1 = std::stoull(std::string(argv[2]));
    const uint64_t dim_2 = std::stoull(std::string(argv[3]));
    const uint64_t dim_3 = std::stoull(std::string(argv[4]));
    
    size_t size_byte = 4;
    const char* fp = env_get_str("FP", "32");
    std::string fp_str(fp);
    if(fp_str == "32")
        size_byte = 4;
    else if(fp_str == "16")
        size_byte = 2;
    else if(fp_str == "8")
        size_byte = 1;
    else{
        printf("error FP:%s\n", fp);
        return -1;
    }

    bool batched = false;
    bool is_kernel_valid = false;
    const char* hsaco;
    void * src_cpu = malloc(dim_0*dim_1*dim_2*dim_3*size_byte);
    void * dst_cpu = malloc(dim_0*dim_1*dim_2*dim_3*size_byte);
    void * dst_gpu_valid = malloc(dim_0*dim_1*dim_2*dim_3*size_byte);

    void * src_gpu;
    void * dst_gpu;
    
    HIP_CALL(hipMalloc(&src_gpu, dim_0*dim_1*dim_2*dim_3*size_byte));
    HIP_CALL(hipMalloc(&dst_gpu, dim_0*dim_1*dim_2*dim_3*size_byte));

    gen_rand_vector<int8_t>(reinterpret_cast<int8_t*>(src_cpu), dim_0*dim_1*dim_2*dim_3*size_byte, -116, 121);
    HIP_CALL(hipMemcpy(src_gpu, src_cpu, dim_0*dim_1*dim_2*dim_3*size_byte, hipMemcpyHostToDevice));

loop<int, 23>([&](auto i) {
    constexpr int all_possible_sequence[23][4] = {
    {0, 1, 3, 2}, {2, 3, 0, 1}, {3, 0, 1, 2}, {0, 2, 3, 1}, {0, 3, 1, 2}, //BATCHED TRANSPOSE
    {0, 2, 1, 3}, {0, 3, 2, 1},
    {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 1, 0},
    {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0} };
    using dst_order = sequence<all_possible_sequence[i][0], all_possible_sequence[i][1], all_possible_sequence[i][2], all_possible_sequence[i][3]>;
    std::cout <<" Tensor reorder to ("<< dst_order::at(0)<<","<< dst_order::at(1)<<","<< dst_order::at(2)<<","<< dst_order::at(3)<<")" << std::endl;
    
    //TODO: an API with more privacy
    auto launch_gpu_init = [&](){
        if((dst_order::at(0)==0 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==2) || 
           (dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1) || 
           (dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2) ||
           (dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==2) ||
           (dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==1)
           ){
            printf("choose batched transpose kernel\n");
            batched = true;
            //batched transpose. NCHW <----> NHWC, (NC)cHW <----> (NC)HWc
            hsaco = env_get_str("BATCHED_TRANSPOSE", BATCHED_TRANSPOSE_HSACO);
            gpu_nhwc_nchw_transpose_init(hsaco);
        }
        else {
            printf("choose general tensor reorder kernel\n");
            hsaco = env_get_str("GENERAL_TENSOR_REORDER_HSACO", GENERAL_TENSOR_REORDER_HSACO);
            gpu_tensor_reorder_init(hsaco);
        }
    };

    auto launch_gpu_tensor_reorder = [&](const transpose_kernel_param_t * kparam){
        if(fp_str == "32")
            gpu_tensor_reorder<float,  dst_order>(reinterpret_cast<float*> (dst_gpu), reinterpret_cast<float*> (src_gpu), dim_0, dim_1, dim_2, dim_3, kparam);
        else if(fp_str == "16")
            gpu_tensor_reorder<ushort, dst_order>(reinterpret_cast<ushort*>(dst_gpu), reinterpret_cast<ushort*>(src_gpu), dim_0, dim_1, dim_2, dim_3, kparam);
        else if(fp_str == "8")
            gpu_tensor_reorder<int8_t, dst_order>(reinterpret_cast<int8_t*>(dst_gpu), reinterpret_cast<int8_t*>(src_gpu), dim_0, dim_1, dim_2, dim_3, kparam);
    };

    auto launch_cpu_tensor_reorder = [&](){
        if(fp_str == "32")
            cpu_tensor_reorder<float,  dst_order>(reinterpret_cast<float*> (dst_cpu), reinterpret_cast<float*> (src_cpu), dim_0, dim_1, dim_2, dim_3);
        else if(fp_str == "16")
            cpu_tensor_reorder<ushort, dst_order>(reinterpret_cast<ushort*>(dst_cpu), reinterpret_cast<ushort*>(src_cpu), dim_0, dim_1, dim_2, dim_3);
        else if(fp_str == "8")
            cpu_tensor_reorder<int8_t, dst_order>(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(src_cpu), dim_0, dim_1, dim_2, dim_3);
    };

    auto test_batched_transpose = [&](const transpose_kernel_param_t *transpose_kparam){
        float kernel_time = 0;
        bool valid = false;
        bool is_kernel_valid = false;

        if(dst_order::at(0)==0 && dst_order::at(1)==2 && dst_order::at(2)==3 && dst_order::at(3)==1){
            is_kernel_valid = transpose_kernel_is_valid(dim_0, dim_1, dim_2 * dim_3, transpose_kparam);
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==1 && dst_order::at(2)==3 && dst_order::at(3)==2){
            is_kernel_valid = transpose_kernel_is_valid(dim_0 * dim_1, dim_2, dim_3, transpose_kparam);
        }
        else if(dst_order::at(0)==0 && dst_order::at(1)==3 && dst_order::at(2)==1 && dst_order::at(3)==2){
            is_kernel_valid = transpose_kernel_is_valid(dim_0, dim_1 * dim_2, dim_3, transpose_kparam);
        }
        else if(dst_order::at(0)==3 && dst_order::at(1)==0 && dst_order::at(2)==1 && dst_order::at(3)==2){
            is_kernel_valid = transpose_kernel_is_valid(1, dim_0 * dim_1 * dim_2, dim_3, transpose_kparam);
        }
        //dst_order::at(0)==2 && dst_order::at(1)==3 && dst_order::at(2)==0 && dst_order::at(3)==1
        else{
            is_kernel_valid = transpose_kernel_is_valid(1, dim_0 * dim_1, dim_2 * dim_3, transpose_kparam);
        }
        if(is_kernel_valid){
            hipEvent_t start, stop;
            HIP_CALL(hipMemset(dst_gpu, 0, dim_0*dim_1*dim_2*dim_3*size_byte));

            for(int i=0; i< warmup; i++){
                launch_gpu_tensor_reorder(transpose_kparam);
            }

            HIP_CALL(hipEventCreate(&start));
            HIP_CALL(hipEventCreate(&stop));
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(start, 0) );

            for(int i=0; i< repeat; i++){
                launch_gpu_tensor_reorder(transpose_kparam);
            }
            HIP_CALL(hipEventRecord(stop, 0) );
            HIP_CALL(hipEventSynchronize(stop) );
            HIP_CALL(hipEventElapsedTime(&kernel_time, start, stop) );
            HIP_CALL(hipEventDestroy(start) );
            HIP_CALL(hipEventDestroy(stop) );
            kernel_time = kernel_time / repeat;

            launch_cpu_tensor_reorder();

            HIP_CALL(hipMemcpy(dst_gpu_valid, dst_gpu, dim_0*dim_1*dim_2*dim_3*size_byte, hipMemcpyDeviceToHost));

            valid = valid_vector_binary(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(dst_gpu_valid), dim_0*dim_1*dim_2*dim_3*size_byte);
        }

        double flop_cnt = 2 * dim_0*dim_1*dim_2*dim_3*size_byte;
        double bw = is_kernel_valid ? flop_cnt / kernel_time / 1e6 : 0;

        printf("[tensor_reorder fp%s] tensor_size:(%lu, %lu, %lu, %lu), flop:%.0f, time:%fms, bw:%.4fGB/s, valid:%s (%dx%d, %dx%d, %dx%d)\n",
            fp_str.c_str(), dim_0, dim_1, dim_2, dim_3, flop_cnt, kernel_time, bw, is_kernel_valid ? (valid ? "y" : "n") : "x",
            transpose_kparam->tile_x, transpose_kparam->tile_y, transpose_kparam->pack_x, transpose_kparam->pack_y, transpose_kparam->ediv_x, transpose_kparam->ediv_y);
        fflush(stdout);

        return valid && is_kernel_valid ? kernel_time : FLT_MAX;
    };

    auto test_general_tensor_reorder = [&](const transpose_kernel_param_t *transpose_kparam){
        float kernel_time = 0;
        bool valid = false;

        bool is_kernel_valid = true;
        if(is_kernel_valid){
            hipEvent_t start, stop;
            HIP_CALL(hipMemset(dst_gpu, 0, dim_0*dim_1*dim_2*dim_3*size_byte));

            for(int i=0; i< warmup; i++){
                launch_gpu_tensor_reorder(transpose_kparam);
            }

            HIP_CALL(hipEventCreate(&start));
            HIP_CALL(hipEventCreate(&stop));
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(start, 0) );

            for(int i=0; i< repeat; i++){
                launch_gpu_tensor_reorder(transpose_kparam);
            }
            HIP_CALL(hipEventRecord(stop, 0) );
            HIP_CALL(hipEventSynchronize(stop) );
            HIP_CALL(hipEventElapsedTime(&kernel_time, start, stop) );
            HIP_CALL(hipEventDestroy(start) );
            HIP_CALL(hipEventDestroy(stop) );
            kernel_time = kernel_time / repeat;

            launch_cpu_tensor_reorder();

            HIP_CALL(hipMemcpy(dst_gpu_valid, dst_gpu, dim_0*dim_1*dim_2*dim_3*size_byte, hipMemcpyDeviceToHost));

            valid = valid_vector_binary(reinterpret_cast<int8_t*>(dst_cpu), reinterpret_cast<int8_t*>(dst_gpu_valid), dim_0*dim_1*dim_2*dim_3*size_byte);
        }

        double flop_cnt = 2 * dim_0*dim_1*dim_2*dim_3*size_byte;
        double bw = is_kernel_valid ? flop_cnt / kernel_time / 1e6 : 0;

        printf("[tensor_reorder fp%s] tensor_size:(%lu, %lu, %lu, %lu), flop:%.0f, time:%fms, bw:%.4fGB/s, valid:%s (256x%d)\n",
            fp_str.c_str(), dim_0, dim_1, dim_2, dim_3, flop_cnt, kernel_time, bw, is_kernel_valid ? (valid ? "y" : "n") : "x",
            transpose_kparam->tile_x);
        fflush(stdout);

        return valid && is_kernel_valid ? kernel_time : FLT_MAX;
    };

    auto get_transpose_all_kernel = [&](){
        if(fp_str == "32")
            return transpose_kernel_get_all_param_t<4>::get();
        else if(fp_str == "16")
            return transpose_kernel_get_all_param_t<2>::get();
        else if(fp_str == "8")
            return transpose_kernel_get_all_param_t<1>::get();
        else
            assert(false);
    };

    auto get_tensor_reorder_all_kernel = [&](){
        if(fp_str == "32")
            return tensor_reorder_kernel_get_all_param_t<4>::get();
        else if(fp_str == "16")
            return tensor_reorder_kernel_get_all_param_t<2>::get();
        else if(fp_str == "8")
            return tensor_reorder_kernel_get_all_param_t<1>::get();
        else
            assert(false);
    };

    batched = false;
    launch_gpu_init();
    float min_tensor_reorder_time = FLT_MAX;
    transpose_kernel_param_t min_tensor_reorder_kparam;
    if(batched){
        for(auto kparam : get_transpose_all_kernel()){
            float current_time = test_batched_transpose(&kparam);
            if(current_time < min_tensor_reorder_time){
                min_tensor_reorder_time = current_time;
                min_tensor_reorder_kparam = kparam;
            }
        }
        printf("-> min time:%fms, kparam: %dx%d, %dx%d, %dx%d\n", min_tensor_reorder_time,
        min_tensor_reorder_kparam.tile_x, min_tensor_reorder_kparam.tile_y, min_tensor_reorder_kparam.pack_x, min_tensor_reorder_kparam.pack_y, min_tensor_reorder_kparam.ediv_x, min_tensor_reorder_kparam.ediv_y);
        fflush(stdout);
        printf("-------------------------\n");
    }
    else{
        for(auto kparam : get_tensor_reorder_all_kernel()){
            float current_time = test_general_tensor_reorder(&kparam);
            if(current_time < min_tensor_reorder_time){
                min_tensor_reorder_time = current_time;
                min_tensor_reorder_kparam = kparam;
            }
        }
        printf("-> min time:%fms, kparam: 256x%d\n", min_tensor_reorder_time, min_tensor_reorder_kparam.tile_x);
        fflush(stdout);
        printf("-------------------------\n");
    }
});

    free(src_cpu);
    free(dst_cpu);
    free(dst_gpu_valid);
}
