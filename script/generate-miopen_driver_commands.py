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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate MIOpenDriver commands from test bench JSON files.")
    
    parser.add_argument("--json-path", type=str, help="Path to test data JSON files")
    
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args:
        print(f"Unknown arguments: {unknown_args}", file=sys.stderr)
        sys.exit(1)
    
    return args

def main():
    args = parse_args()
    
    if not args.json_path:
        print("Error: --json-path argument is required.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.json_path):
        print(f"Error: The path {args.json_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    driver_commands_per_group_size = {}
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
        for item in data.get("test_cases"):
            if bool(item.get("heuristics_enabled")):
                cmd_str = item.get("mi_open_driver_command")
                cmd_str = cmd_str.replace('./bin/MIOpenDriver', '')
                match = re.search(r'--group_count\s+(\d+)', cmd_str)
                group_size = int(match.group(1))
                if driver_commands_per_group_size.get(group_size) is None:
                    driver_commands_per_group_size[group_size] = []
                driver_commands_per_group_size[group_size].append(cmd_str)
    # Print the number of commands per group size
    for group_size, commands in driver_commands_per_group_size.items():
        print(f"Group Size: {group_size}, Number of Commands: {len(commands)}")
        
    # Save the commands to a CSV file
    base_path = os.path.dirname(args.json_path)
    csv_file_path = os.path.join(base_path, 'miopen_driver_commands.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Group Size', 'Command'])
        for group_size, commands in driver_commands_per_group_size.items():
            for command in commands:
                writer.writerow([group_size, command])
    print(f"Commands saved to {csv_file_path}")

if __name__ == "__main__":
    main()