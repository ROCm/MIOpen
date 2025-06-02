#!/usr/bin/env python3

import argparse
import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MIOpen test bench results.")
    # Add variable length argument for the results file names
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for the CSV files.")
    parser.add_argument("--files", nargs="+", help="Kernel times file names to analyze.")
    parser.add_argument("--labels", nargs="+", help="Labels for the kernel time files.")

    args, _ = parser.parse_known_args()
    
    return args

def read_csv_file(file_path):
    """
    Read a CSV file and return the data as a list of dictionaries.
    Each dictionary corresponds to a row in the CSV file.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(',')
            data = []
            for line in lines[1:]:
                values = [value.strip() for value in list(csv.reader([line.strip()]))[0]]
                data.append(dict(zip(headers, values)))
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def calculate_statistics(data):
    kernel_times = []
    inference_times = []
    failed_cases = 0
    for row in data:
      try:
        inference_time = float(row['KTN_inference_time_ms'])
        if inference_time > 0:
          kernel_times.append(float(row['Elapsed_GPU_time_average_ms']))
          inference_times.append(inference_time)
        else:
           failed_cases += 1
      except ValueError:
        failed_cases += 1

    kernel_times = np.array(kernel_times, dtype=float)
    inference_times = np.array(inference_times, dtype=float)
    stats = {
        'kernel_time_mean': np.mean(kernel_times),
        'kernel_time_median': np.median(kernel_times),
        'kernel_time_max': np.max(kernel_times),
        'inference_time_mean': np.mean(inference_times),
        'inference_time_median': np.median(inference_times),
        'inference_time_max': np.max(inference_times),
        'inference_time_min': np.std(inference_times),
        'failed_cases': failed_cases,
        'total_cases': len(data)
    }
    return stats

def run_worst_cases_analysis(results, top_number=10, print_results=False):
    """
    Analyze the worst cases based on kernel times and inference times.
    """
    
    # For each label, find the top worst cases based on kernel times.
    # Then, compare the performance from other labels in these cases.
    worst_cases = {}
    all_kernel_times = {}
    for label, data in results.items():
        kernel_times = {}
        for row in data:
            try:
                inference_time = float(row['KTN_inference_time_ms'])
                if inference_time > 0:
                    configuration = row['Configuration']
                    time = float(row['Elapsed_GPU_time_average_ms'])
                    if configuration not in kernel_times:
                        kernel_times[configuration] = [time]
                    else:
                        kernel_times[configuration].append(time)
            except ValueError:
                continue

        # Convert lists to average times
        for config, times in kernel_times.items():
            kernel_times[config] = np.mean(times)

        # Get top_number worst cases based on kernel times
        sorted_cases = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:top_number]
        worst_cases[label] = {
            'kernel_times': [time for _, time in sorted_cases],
            'configurations': [config for config, _ in sorted_cases]
        }
        all_kernel_times[label] = kernel_times

    for label, worst_cases_data in worst_cases.items():
        if print_results:
            print(f"Worst cases for {label}:")
 
        plt.figure(figsize=(15, 10))
        
        all_labels = list(all_kernel_times.keys())
        num_labels = len(all_labels)
        x = np.arange(len(worst_cases_data['configurations']))
        width = 0.8 / num_labels
        
        for i, (config, time) in enumerate(zip(worst_cases_data['configurations'], worst_cases_data['kernel_times'])):
            if print_results:
                print(f"  {i+1}. Configuration: {config}")
            
            for j, other_label in enumerate(all_labels):
                if config in all_kernel_times[other_label]:
                    other_time = all_kernel_times[other_label][config]
                    if print_results:
                        print(f"    {other_label}: {other_time:.2f} ms")
                    plt.bar(x[i] + (j - num_labels/2 + 0.5) * width, other_time, 
                        width=width, color=f'C{j}', label=other_label if i == 0 else "")
                    plt.yscale('log')  # Set y-axis to logarithmic scale
                else:
                    if print_results:
                        print(f"    {other_label}: N/A")
        
        plt.xlabel('Configuration')
        plt.ylabel('Kernel Time (ms)')
        plt.title(f"Top-10 longest running cases for '{label}' model")
        plt.xticks(x, [f"{i+1}" for i in range(len(worst_cases_data['configurations']))], rotation=0)
        if num_labels > 1:
            plt.legend()

        plt.savefig(f"worst_cases_{label}.png")

def get_kernel_times(data):
    kernel_times = {}
    for row in data:
        try:
            inference_time = float(row['KTN_inference_time_ms'])
            if inference_time > 0:
                configuration = row['Configuration']
                time = float(row['Elapsed_GPU_time_average_ms'])
                if configuration not in kernel_times:
                    kernel_times[configuration] = [time]
                else:
                    kernel_times[configuration].append(time)
        except ValueError:
            continue

    # Convert lists to average times
    for config, times in kernel_times.items():
        kernel_times[config] = np.mean(times)

    # Convert negative times to maximum value of float
    for config, time in kernel_times.items():
        if time < 0:
            kernel_times[config] = float('inf')

    return kernel_times

def run_pairwise_comparison(results, tol=0.025):
    """
    Run pairwise comparison of kernel times and inference times between different labels.
    """
    labels = list(results.keys())
    num_labels = len(labels)

    for i in range(num_labels):
        label1 = labels[i]
        data1 = results[label1]
        kernel_times1 = get_kernel_times(data1)
        for j in range(i + 1, num_labels):
            label2 = labels[j]
            data2 =results[label2]
            kernel_times2 = get_kernel_times(data2)
            
            # Prepare data for boxplot
            selected_configs = {}
            key1 = label1 + ' better'
            key2 = label2 + ' better'
            selected_configs[key1] = 0
            selected_configs[key2] = 0
            equal_key = f'equal (tolerance: {100*tol:.1f}%)' 
            selected_configs[equal_key] = 0
            for config in kernel_times1.keys():
                time1 = kernel_times1.get(config, float('inf'))
                time2 = kernel_times2.get(config, float('inf'))
                if abs(time1 - time2) < tol * min(time1, time2):
                    selected_configs[equal_key] += 1
                else:
                    if time1 < time2:
                        selected_configs[key1] += 1
                    elif time2 < time1:
                        selected_configs[key2] += 1

            # Create a boxplot for the pairwise comparison
            plt.figure(figsize=(10, 6))
            plt.bar(selected_configs.keys(), selected_configs.values(), color=['blue', 'orange', 'green'])
            plt.ylabel('Number of Configurations')
            plt.title(f"Kernel runtime comparison for predicted parameters: {label1} vs {label2}")
            plt.savefig(f"pairwise_comparison_{label1}_vs_{label2}.png")

def main():
    base_dir, kernel_time_files, kernel_time_labels = parse_args().base_dir, parse_args().files, parse_args().labels

    # Check if the base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The base directory {base_dir} does not exist.")

    # Check if the number of files and labels match
    if len(kernel_time_files) != len(kernel_time_labels):
        raise ValueError("The number of kernel time files must match the number of labels.")

    # Create a list to hold the file paths
    file_paths = [os.path.join(base_dir, f) for f in kernel_time_files]

    results = {}
    for path, label in zip(file_paths, kernel_time_labels):
        res = read_csv_file(path)
        results[label] = res

    statistics = {}
    for label, data in results.items():
      stats = calculate_statistics(data)
      statistics[label] = stats
      print(f"Statistics for {label}: {stats}")
      print()

    run_worst_cases_analysis(results)

    run_pairwise_comparison(results)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
    