/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#define NUM_PERF_RUNS 5
#define NUM_WARMUP_RUNS 3

template <typename T>
struct PerfHelper
{
    std::vector<std::tuple<std::string, T, T, double, double, double>> kernelTestStats;

    // hold the min, max, mean, median, and standard deviation
    std::tuple<T, T, double, double, double> gpuStats;

    static T perf_min(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return *std::min_element(data.begin(), data.end());
    }

    static T perf_max(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return *std::max_element(data.begin(), data.end());
    }

    static double perf_mean(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    static double perf_median(std::vector<T> data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        size_t size = data.size();
        std::sort(data.begin(), data.end());
        if(size % 2 == 0)
        {
            return (data[size / 2 - 1] + data[size / 2]) / 2.0;
        }
        else
        {
            return data[size / 2];
        }
    }

    static double perf_standardDeviation(const std::vector<T>& data)
    {
        if(data.empty())
            throw std::invalid_argument("Empty vector");
        double data_mean = perf_mean(data);
        double sq_sum    = std::inner_product(
            data.begin(), data.end(), data.begin(), 0.0, std::plus<>(), [data_mean](T a, T b) {
                return (a - data_mean) * (b - data_mean);
            });
        return std::sqrt(sq_sum / data.size());
    }

    static std::tuple<T, T, double, double, double> calcStats(const std::vector<T>& data)
    {
        T min_val         = perf_min(data);
        T max_val         = perf_max(data);
        double mean_val   = perf_mean(data);
        double median_val = perf_median(data); // Note: This modifies the data(sorts it)
        double sd_val     = perf_standardDeviation(data);
        return {min_val, max_val, mean_val, median_val, sd_val};
    }

    void writeStatsToCSV(const std::string& filename, std::string test_info)
    {
        std::ofstream file;

        // Open the file in append mode. Create it if it doesn't exist.
        file.open(filename, std::ios::app);

        // Check if the file is open, throw an exception if not.
        if(!file.is_open())
        {
            throw std::runtime_error("Failed to open file");
        }

        // If the file was just created (i.e., its size is 0), write the header.
        if(miopen::fs::file_size(filename) == 0)
        {
            file << "KernelAndTestInfo,min_exec_time_ratio,max_exec_time_ratio,mean_exec_time_"
                    "ratio,median_exec_time_ratio,SD_ocl,SD_hip\n";
        }

        // if the number of entries in the kernelTestStats vector is odd, throw an exception
        if(kernelTestStats.size() % 2 != 0)
        {
            throw std::runtime_error("The number of entries in the kernelTestStats vector is odd");
        }

        // Calculate the half size of the kernelTestStats vector
        size_t halfSize = kernelTestStats.size() / 2;

        // Iterate over the first half of the kernelTestStats vector
        for(size_t i = 0; i < halfSize; ++i)
        {
            // Access the i-th element from the first half and (i + halfSize)-th element from the
            // second half
            auto& firstHalfElement  = kernelTestStats[i];
            auto& secondHalfElement = kernelTestStats[i + halfSize];

            // Write the perf data to the file
            file << std::get<0>(firstHalfElement) + test_info << "," // KernelAndTestInfo
                 << std::get<1>(firstHalfElement) / std::get<1>(secondHalfElement)
                 << "," // min_exec_time_ratio
                 << std::get<2>(firstHalfElement) / std::get<2>(secondHalfElement)
                 << "," // max_exec_time_ratio
                 << std::get<3>(firstHalfElement) / std::get<3>(secondHalfElement)
                 << "," // mean_exec_time_ratio
                 << std::get<4>(firstHalfElement) / std::get<4>(secondHalfElement)
                 << "," // median_exec_time_ratio
                 << std::get<5>(firstHalfElement) << "," << std::get<5>(secondHalfElement)
                 << "\n"; // SD_ocl, SD_hip
        }

        file.close();
    }

    template <typename... Args>
    void perfTest(miopen::Handle& handle,
                  const std::string& kernel_name,
                  const std::string& network_config,
                  bool append,
                  Args&&... args)
    {
        // Get kernels matching the kernel_name and network_config from the cache
        auto&& kernels = handle.GetKernels(kernel_name, network_config);
        // Ensure we have at least one kernel
        assert(!kernels.empty());
        // Vector to hold the execution times
        std::vector<T> elapsedTime_ms;

        if(handle.IsProfilingEnabled())
        { // If profiling was enabled elsewhere, reset the kernel time
            handle.ResetKernelTime();
        }
        else
        {
            handle.EnableProfiling(); // Enable profiling
            handle.ResetKernelTime(); // for good measure?
        }
        // Optionally ignore the first few runs to allow for warm-up
        for(size_t i = 0; i < NUM_PERF_RUNS + NUM_WARMUP_RUNS; i++)
        {
            // Execute the kernel
            kernels.front()(std::forward<Args>(args)...);
            // Append the elapsed time to the vector
            if(i >= NUM_WARMUP_RUNS)
                elapsedTime_ms.push_back(handle.GetKernelTime());
            handle.ResetKernelTime();
        }

        handle.EnableProfiling(false); // Disable profiling

        gpuStats = calcStats(elapsedTime_ms);
        kernelTestStats.push_back({kernel_name,
                                   std::get<0>(gpuStats),
                                   std::get<1>(gpuStats),
                                   std::get<2>(gpuStats),
                                   std::get<3>(gpuStats),
                                   std::get<4>(gpuStats)});
    }
};
