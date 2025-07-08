#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import json
import re
import time
import shlex
import csv
from pathlib import Path
from datetime import datetime

class ProgressBar():
  """ progress bar to track progress
  inspired from stackoverflow.com/a/13685020/5046433 """

  def __init__(self,
               end_val,
               title='Progress',
               bar_length=100,
               char_at_end='\n'):
    self.title = title
    self.end_val = end_val
    self.bar_length = bar_length
    self.char_at_end = char_at_end
    self.wheel = ['|', '/', '\\', '|', '/', '\\']
    self.wheel_length = len(self.wheel)
    self.wheel_count = 0

  def display(self, progress=0):
    """ display progress bar filled to the ratio of progress:end_val """
    percent = float(progress) / self.end_val
    hashes = '#' * int(round(percent * self.bar_length))
    wheel = self.wheel[self.wheel_count % self.wheel_length]
    self.wheel_count += 1
    spaces = ' ' * max(0,(self.bar_length - len(hashes) - 1))
    sys.stdout.write("\r{0}: [{1}] {2}%".format(self.title, hashes + wheel + spaces,
                                                int(round(percent * 100))))
    sys.stdout.flush()

    if progress == self.end_val:
      if self.char_at_end:
        sys.stdout.write(self.char_at_end)
        sys.stdout.flush()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        # Check for jq
        subprocess.run(["jq", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: jq is required but not installed. Please install jq.")
        sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run MIOpen driver with various configuration options")
    
    parser.add_argument("--log-level", type=int, help="Set log level (optional, default: 5)")
    parser.add_argument("--gpu-id", type=int, help="Set GPU ID (optional, default: 0)")
    parser.add_argument("--algorithms", type=str, help="Set which algorithms to use (optional, default: ALL)")
    parser.add_argument("--tuning-db-dir", type=str, help="Set tuning DB directory path")
    parser.add_argument("--incremental-tuning", action="store_true", help="Use incremental tuning")
    parser.add_argument("--exhaustive-tuning", action="store_true", help="Use exhaustive tuning")
    parser.add_argument("--config-file", type=str, help="Specify the configuration file")
    parser.add_argument("--config", type=str, help="Specify the configuration name to use from the config file")
    parser.add_argument("--disable-kernel-cache", action="store_true", help="Disable kernel cache")
    parser.add_argument("--onnx-model-path", type=str, help="Specify the ONNX model path")
    parser.add_argument("--frugal-model-path", type=str, help="Specify the Frugal model path")
    parser.add_argument("--no-heuristics", dest="no_heuristics", action="store_true", help="Run only the cases without heuristics")
    parser.add_argument("--only-heuristics", dest="only_heuristics", action="store_true", help="Run only cases with AI heuristics enabled.")
    parser.add_argument("--run-id", type=str, dest="run_id", help="Run ID for the test case")
    parser.add_argument("--log-to-file", dest="log_to_file",action="store_true", help="Log output to individual log files.")
    parser.add_argument("--start-from", type=int, dest="start_from", default=0, help="Start from a specific test case index (optional, default: 0)")
    parser.add_argument("--pid", type=int, dest="pid", default=os.getpid(), help="Process ID for the test case (optional, default: current process ID)")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args


def extract_kernel_times(log_file, times_file, config):
    """Extract kernel times from log file and append to CSV"""
    algo_num = ""
    sol_id = ""
    kernel_name = ""
    time_value = ""
    ktn_inference_time = "-1"  # The AI heuristics may not be enabled, set negative value to indicate this
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for algorithm info
            match = re.search(r'MIOpen\s+(.*)\s+Algorithm:\s+(\d+),\s+Solution:\s+(\d+)/([A-Za-z0-9_]+)', line)
            if match:
                algo_num = match.group(2)
                sol_id = match.group(3)
                kernel_name = match.group(4)
            
            # Look for kernel time
            match = re.search(r'GPU\s+Kernel\s+Time\s+(.*)\s+Elapsed:\s+([0-9.]+)\s+ms', line)
            if match:
                time_value = match.group(2)
            
            # Look for KTN inference time
            match = re.search(r'\[ModelSetParams\]\s+KTN\s+ran\s+for\s+(\d+)\s+micro-seconds', line)
            if match:
                ktn_micro = int(match.group(1))
                ktn_inference_time = f"{ktn_micro / 1000:.3f}"
    
    # Append to CSV
    with open(times_file, 'a') as f:
        data_row = [config, algo_num, sol_id, kernel_name, time_value, ktn_inference_time]
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(data_row)

def process_single_config(test_case, kernel_times_path, algs, tuning, pid, log_to_file=False):
    """Process a single configuration"""

    # Get the verb from the command.
    # The command can be in the form of: ./bin/MIOpenDriver convfp16 XXX, where we want to remove the "./bin/MIOpenDriver" part.
    command = test_case['mi_open_driver_command'].split(" ")[1:]
    test_name = " ".join(command)
    test_name= test_name.strip()
    
    # Set environment variables from config
    set_env_vars = []
    env_vars = test_case.get("env_vars", {})
    
    # Record the case specific variable so that they can be later unset.
    for key, value in env_vars.items():
        os.environ[key] = str(value)
        set_env_vars.append(key)

    # Append the current environment variables from os.environ
    for key, value in os.environ.items():
        if key not in env_vars:
            env_vars[key] = value
    
    # calculate hash for the test name
    test_name_hash = hash(test_name)
    test_name_hash = f"{test_name_hash:08x}"
    
    # Setup log path
    log_path_config = f"../logs/individual_logs/{test_name_hash}-{algs}{tuning}{pid}.log"
    os.makedirs(os.path.dirname(log_path_config), exist_ok=True)
    
    # Prepare the MIOpenDriver command
    miopen_driver_path = os.path.abspath(os.path.join(os.getcwd(), "../build/bin/MIOpenDriver"))
    cmd_str = test_case['mi_open_driver_command']
    cmd_str = cmd_str.replace("./bin/MIOpenDriver", "")
    
    # Split the command string into list of arguments
    cmd_args = shlex.split(cmd_str.strip())
    
    # Construct the final command list with executable as first element
    full_cmd = [miopen_driver_path] + cmd_args

    # Write to log file
    with open(log_path_config, 'w') as log_file:
        log_file.write(f"=== Execution started at {datetime.now()} ===\n\n")
        
        # Run the MIOpenDriver command and capture start/end times
        start_time = time.time()
        try:
            result = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env_vars
            )
            log_file.write(result.stdout)
            if result.returncode != 0:
                log_file.write(f"ERROR: MIOpenDriver command failed with return code {result.returncode}.\n")
        except Exception as e:
            log_file.write(f"ERROR: Failed to execute MIOpenDriver: {e}\n")
        end_time = time.time()
        
        # Record execution time
        log_file.write(f"\n=== MIOpenDriver execution time ===\n")
        log_file.write(f"real\t{end_time - start_time:.3f}s\n\n")
        
        # Record environment variables
        log_file.write("=== Environment Variables ===\n")
        for key, value in env_vars.items():
            log_file.write(f"{key}={value}\n")
        
        log_file.write(f"\n=== Execution completed at {datetime.now()} ===\n")
    
    extract_kernel_times(log_path_config, kernel_times_path, test_name)
    if not log_to_file:
      # If the file based logging is not enabled, remove the log file.
      # By default, we don't stire the log files since there are lot of them.
      os.remove(log_path_config)
    
    # Unset environment variables specific to the test case
    for var in set_env_vars:
        if var in os.environ:
            del os.environ[var]


def main():
    # Check dependencies
    check_dependencies()
    
    # Handle help flag
    if len(sys.argv) == 1 or "--help" in sys.argv:
        print("Usage: python run_miopen_driver.py --config <name> [OPTIONS]")
        print("Options:")
        print("  --log-level <level>: Set log level (optional, default: 5)")
        print("  --gpu-id <device id>: Set GPU ID (optional, default: 0)")
        print("  --algorithms <alg1>, <alg2>, ...: Set which algorithms to use (optional, default: ALL)")
        print("  --tuning-db-dir <dir path>: Set tuning DB directory path (required for --incremental-tuning and --exhaustive-tuning flags)")
        print("  --incremental-tuning: Use incremental tuning (optional)")
        print("  --exhaustive-tuning: Use exhaustive tuning (optional)")
        print("  --config-file <file>: Specify the configuration file (optional, default: configs.json)")
        print("  --config <name>: Specify the configuration name to use from the config file (required)")
        print("  --disable-kernel-cache: Disable kernel cache (optional)")
        print("  --onnx-model-path <path>: Specify the ONNX model path (optional)")
        print("  --frugal-model-path <path>: Specify the Frugal model path (optional)")
        print("  --no-heuristics: Run only the cases without heuristics (optional)")
        print("  --only-heuristics: Run only cases with AI heuristics enabled (optional)")
        print("  --run-id <id>: Run ID for the test case (optional)")
        print("  --log-to-file: Log output to individual log files (optional)")
        print("  --help: Show this help message")
        sys.exit(0)
    
    # Parse arguments
    args = parse_args()
    
    if args.onnx_model_path:
        os.environ["MIOPEN_KTN_MODELS_PATH"] = args.onnx_model_path
        os.environ["MIOPEN_USE_ONNX_KTN"] = "1"
        if args.frugal_model_path:
            raise ValueError("Cannot specify both --onnx-model-path and --frugal-model-path")
    
    if args.frugal_model_path:
        os.environ["MIOPEN_KTN_MODELS_PATH"] = args.frugal_model_path
        os.environ["MIOPEN_USE_ONNX_KTN"] = "0"
        if args.onnx_model_path:
            raise ValueError("Cannot specify both --onnx-model-path and --frugal-model-path")

    # Fix path to ONNX runtime
    os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:/opt/onnxruntime/lib:/usr/local/lib"
    
    # Set config file
    config_file = args.config_file if args.config_file else "configs.json"
    print(f"Using configuration file: {config_file}")
    
    # Check if config file exists
    if not os.path.isfile(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Handle tuning DB path
    tuning_db = args.tuning_db_dir
    if tuning_db:
        print(f"Using tuning DB: {tuning_db}")
        # Create the tuning DB directory if it doesn't exist
        os.makedirs(tuning_db, exist_ok=True)
        os.environ["MIOPEN_USER_DB_PATH"] = tuning_db
    
    # Handle kernel cache
    if args.disable_kernel_cache:
        print("Disabling kernel cache")
        os.environ["MIOPEN_DISABLE_CACHE"] = "1"
    else:
        print("Kernel cache is enabled by default")
    
    # Set tuning flags
    tuning = ""
    
    # Handle incremental tuning
    if args.incremental_tuning:
        print("Running with incremental tuning")
        os.environ["MIOPEN_FIND_MODE"] = "3"
        os.environ["MIOPEN_FIND_ENFORCE"] = "3"
        
        if not tuning_db:
            print("Error: Tuning DB path must be provided with --tuning-db-dir when using --incremental-tuning", file=sys.stderr)
            sys.exit(1)
            
        if args.disable_kernel_cache:
            print("Error: Kernel cache must be enabled when using --incremental-tuning", file=sys.stderr)
            sys.exit(1)
            
        tuning = "-incremental_tuning-"
    
    # Handle exhaustive tuning
    if args.exhaustive_tuning:
        print("Running with exhaustive tuning")
        os.environ["MIOPEN_FIND_MODE"] = "3"
        os.environ["MIOPEN_FIND_ENFORCE"] = "3"
        
        if not tuning_db:
            print("Error: Tuning DB path must be provided with --tuning-db-dir when using --exhaustive-tuning", file=sys.stderr)
            sys.exit(1)
            
        if args.disable_kernel_cache:
            print("Error: Kernel cache must be enabled when using --exhaustive-tuning", file=sys.stderr)
            sys.exit(1)
            
        os.environ["MIOPEN_SYSTEM_DB_PATH"] = os.environ.get("MIOPEN_USER_DB_PATH", "")
        tuning = "-exhaustive_tuning-"
    
    # Set log level
    log_level = args.log_level if args.log_level is not None else 5
    print(f"Using log level: {log_level}")
    
    # Set GPU ID
    gpu_id = args.gpu_id if args.gpu_id is not None else 0
    print(f"Using GPU ID: {gpu_id}")
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set algorithms
    if args.algorithms:
        algorithms_raw = args.algorithms
        algorithms = algorithms_raw.replace(" ", "")
        print(f"Using algorithms: {algorithms}")
    else:
        algorithms = "ALL"
        print(f"Using default algorithms: {algorithms}")
    
    # Set algorithm environment variables
    os.environ["MIOPEN_DEBUG_CONV_FFT"] = "0"
    os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "0"
    os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "0"
    os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "0"
    os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "0"
    
    algs = ""
    if "ALL" in algorithms:
        os.environ["MIOPEN_DEBUG_CONV_FFT"] = "1"
        os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "1"
        os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "1"
        os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "1"
        os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
        algs = "ALL"
    else:
        algorithm_array = algorithms.split(",")
        for algorithm in algorithm_array:
            if algorithm == "FFT":
                os.environ["MIOPEN_DEBUG_CONV_FFT"] = "1"
                algs += "FFT-"
            elif algorithm == "DIRECT":
                os.environ["MIOPEN_DEBUG_CONV_DIRECT"] = "1"
                algs += "DIRECT-"
            elif algorithm == "WINOGRAD":
                os.environ["MIOPEN_DEBUG_CONV_WINOGRAD"] = "1"
                algs += "WINOGRAD-"
            elif algorithm == "GEMM":
                os.environ["MIOPEN_DEBUG_CONV_GEMM"] = "1"
                algs += "GEMM-"
            elif algorithm == "IMPLICIT_GEMM":
                os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "1"
                algs += "IMPLICIT_GEMM-"
            else:
                print(f"Error: Unknown algorithm: {algorithm}", file=sys.stderr)
                sys.exit(1)
    
    # Set debug logging flags
    os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
    os.environ["MIOPEN_LOG_LEVEL"] = str(log_level)
    os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
    
    # Set process ID
    pid = args.pid

    continue_previous_run = args.start_from > 0
    
    # Create CSV file for kernel times
    run_id = ""
    if args.run_id:
        run_id = f"{args.run_id}-"
    kernel_times_path = f"../logs/kernel_times-{run_id}{algs}{tuning}{pid}.csv"
    os.makedirs(os.path.dirname(kernel_times_path), exist_ok=True)
    if not continue_previous_run or not os.path.exists(kernel_times_path):
        with open(kernel_times_path, 'w') as f:
            f.write("Configuration,Algo_num,Solution_ID,Solver_name,Elapsed_GPU_time_average_ms,KTN_inference_time_ms\n")
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    configs = config_data['test_cases']
    if args.only_heuristics:
        configs = [case for case in configs if case.get("heuristics_enabled", False)]

    if args.no_heuristics:
        configs = [case for case in configs if not case.get("heuristics_enabled", False)]

    # Process configurations
    if args.config:
        config_name = args.config
        try:
            test_cases = [case for case in configs if case['name'] == config_name]
            if not test_cases:
                print(f"Error: Configuration '{config_name}' not found in {config_file}")
                exit(1)
            else:
                progressbar = ProgressBar(end_val=len(test_cases), title='Progress', char_at_end='\t')
                print(f"Running {len(test_cases)} test cases...")
                for i, case in enumerate(test_cases):
                    progressbar.display(progress=i+1)
                    process_single_config(case, kernel_times_path, algs, tuning, pid, args.log_to_file)
        except Exception as e:
            print(f"Error processing configuration {config_name}: {e}")
            
    else:
        # Process all configurations
        start_from = args.start_from
        if start_from > 0:
            print(f"Starting from test case index: {start_from}")
            configs = configs[start_from:]
        progressbar = ProgressBar(end_val=len(configs), title='Progress', char_at_end='\t')
        print(f"Running {len(configs)} test cases...")
        for i, case in enumerate(configs):
            progressbar.display(progress=i+1)
            process_single_config(case, kernel_times_path, algs, tuning, pid, args.log_to_file)

if __name__ == "__main__":
    main()