#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Get the directory where the script is located
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go one level up to PACKAGE_HOME
PACKAGE_HOME="$(dirname "$BUILD_DIR")"

SCRIPT_DIR="$PACKAGE_HOME/script/"

# Search for build.ninja under PACKAGE_HOME
BUILD_NINJA_FILE="$PACKAGE_HOME/build/build.ninja"

if [ -z "$BUILD_NINJA_FILE" ]; then
    echo "Error: build.ninja not found under $PACKAGE_HOME"
    exit 1
fi

python3 "$SCRIPT_DIR/dependency-parser/main.py" parse "$BUILD_NINJA_FILE" --workspace-root "$PACKAGE_HOME"

# Get the directory containing build.ninja
BUILD_DIR=$(dirname "$BUILD_NINJA_FILE")

# Path to enhanced_dependency_mapping.json in the same directory
JSON_FILE="$BUILD_DIR/enhanced_dependency_mapping.json"

# Check if the JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: $JSON_FILE not found."
    exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)

# Run the command
python3 "$SCRIPT_DIR/dependency-parser/main.py" select "$JSON_FILE" origin/develop $branch --folder ../

# Path to tests_to_run.json in the same directory
TEST_FILE="tests_to_run.json"

command=$(python3 -c "
import json
import os
with open('$TEST_FILE', 'r') as f:
    data = json.load(f)
    tests = data.get('tests_to_run', [])
    if tests:
        # Extract just the filename after the last '/'
        clean_tests = [os.path.basename(test) for test in tests]
        print('ctest -R \"' + '|'.join(clean_tests) + '\"')
    else:
        print('# No tests to run')
")

echo "$command"

eval "$command"


