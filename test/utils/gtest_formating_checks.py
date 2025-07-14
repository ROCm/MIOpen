#*******************************************************************************
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#*******************************************************************************
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FOLDER_PATH = "../../test/gtest"

# Ignore list: Add test names or file paths you want to exclude
#              (keep the list sorted to min. git conflicts)
IGNORE_LIST = {
    "../../test/gtest/binary_tensor_ops.cpp",
    "../../test/gtest/graphapi_conv_bias_res_add_activ_fwd.cpp",
    "../../test/gtest/graphapi_operation_rng.cpp",
    "../../test/gtest/layout_transpose.cpp",
    "../../test/gtest/reduce_custom_fp32.cpp",
    "../../test/gtest/unary_tensor_ops.cpp",
    "CPU_MIOpenDriverRegressionBigTensorTest_FP32",
    "GPU_UnitTestConvSolverAsmBwdWrW3x3Wrw_FP32",
}

# Valid enums and Regex for validation
VALID_HW_TYPES = {"CPU", "GPU"}
VALID_DATATYPES = {"FP8", "FP16", "FP32", "FP64", "BFP16", "BFP8", "I64", "I32", "I16", "I8", "NONE"}
# Our suite (or fixture) naming convention: must start with CPU or GPU, followed by one or more alphanum groups, and end with a valid datatype.
TESTSUITE_REGEX = re.compile(
    r"^(CPU|GPU)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_(" + "|".join(VALID_DATATYPES) + r")$"
)
# Test type for instantiations must be one of these words, optionally with extra alphanum characters.
TEST_TYPE_REGEX = re.compile(r"^(Smoke|Full|Perf|Unit)([A-Za-z0-9]*)?$")

# Updated regexes that do not allow newlines in the macro arguments
TEST_P_REGEX = re.compile(
    r"\bTEST_P\(\s*([^\n,]+?)\s*,\s*([^\n\)]+?)\s*\)"
)
INSTANTIATE_TEST_REGEX = re.compile(
    r"\bINSTANTIATE_TEST_SUITE_P\(\s*([^\n,]+?)\s*,\s*([^\n,]+?)\s*,"
)
ALLOW_UNINSTANTIATED_REGEX = re.compile(
    r"\bGTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST\(\s*([^\n\)]+?)\s*\)"
)
TEST_REGEX = re.compile(
    r"\bTEST\(\s*([^\n,]+?)\s*,\s*([^\n\)]+?)\s*\)"
)
TEST_F_REGEX = re.compile(
    r"\bTEST_F\(\s*([^\n,]+?)\s*,\s*([^\n\)]+?)\s*\)"
)


def analyze_tests(folder_path):
    errors = []

    # Walk over all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".cpp"):
                continue

            file_path = os.path.join(root, file)

            # Skip file if it is in the ignore list
            if file_path in IGNORE_LIST:
                logging.info(f"Skipping ignored file: {file_path}")
                continue

            with open(file_path, "r") as f:
                content = f.read()

            # Use the content for proper line-number computation
            def get_line_number(position):
                # Count the number of newline characters before the position; add 1 so that numbering starts at 1.
                return content.count("\n", 0, position) + 1

            # Dictionaries to record macro definitions.
            # For TEST_P, TEST, and TEST_F we key on (suite_or_fixture, test_name)
            test_p_definitions = {}  # key: (suite, test_name) -> line number
            test_definitions = {}    # key: (suite, test_name) -> line number
            test_f_definitions = {}  # key: (fixture, test_name) -> line number

            # For INSTANTIATE_TEST_SUITE_P we group by test suite; for each suite, instantiation names must be unique.
            instantiations = {}  # key: suite -> dict of {instantiation_name: line number}

            # For GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST, we store a list of line numbers per suite.
            allowed_uninstantiated = {}  # key: suite -> list of line numbers

            # === Process TEST_P macros ===
            for m in TEST_P_REGEX.finditer(content):
                suite, test_name = m.groups()
                line = get_line_number(m.start())
                if suite in IGNORE_LIST or test_name in IGNORE_LIST:
                    logging.info(f"Skipping ignored test suite: {suite}")
                    continue
                if not TESTSUITE_REGEX.match(suite):
                    errors.append(f"{file_path}:{line}: Invalid TESTSUITE_NAME '{suite}' in TEST_P.")
                key = (suite, test_name)
                if key in test_p_definitions:
                    prev_line = test_p_definitions[key]
                    errors.append(
                        f"{file_path}:{line}: Duplicate TEST_P for '{suite}.{test_name}' (previously defined at line {prev_line})."
                    )
                else:
                    test_p_definitions[key] = line

            # === Process TEST macros ===
            for m in TEST_REGEX.finditer(content):
                suite, test_name = m.groups()
                line = get_line_number(m.start())
                if suite in IGNORE_LIST or test_name in IGNORE_LIST:
                    logging.info(f"Skipping ignored test suite: {suite}")
                    continue
                if not TESTSUITE_REGEX.match(suite):
                    errors.append(f"{file_path}:{line}: Invalid TEST suite name '{suite}' in TEST.")
                key = (suite, test_name)
                if key in test_definitions:
                    prev_line = test_definitions[key]
                    errors.append(
                        f"{file_path}:{line}: Duplicate TEST for '{suite}.{test_name}' (previously defined at line {prev_line})."
                    )
                else:
                    test_definitions[key] = line

            # === Process TEST_F macros ===
            for m in TEST_F_REGEX.finditer(content):
                fixture, test_name = m.groups()
                line = get_line_number(m.start())
                if fixture in IGNORE_LIST or test_name in IGNORE_LIST:
                    logging.info(f"Skipping ignored test fixture: {fixture}")
                    continue
                if not TESTSUITE_REGEX.match(fixture):
                    errors.append(f"{file_path}:{line}: Invalid TEST_F fixture name '{fixture}'.")
                key = (fixture, test_name)
                if key in test_f_definitions:
                    prev_line = test_f_definitions[key]
                    errors.append(
                        f"{file_path}:{line}: Duplicate TEST_F for '{fixture}.{test_name}' (previously defined at line {prev_line})."
                    )
                else:
                    test_f_definitions[key] = line

            # === Process INSTANTIATE_TEST_SUITE_P macros ===
            for m in INSTANTIATE_TEST_REGEX.finditer(content):
                instantiation_name, suite = m.groups()
                line = get_line_number(m.start())
                if suite in IGNORE_LIST or instantiation_name in IGNORE_LIST:
                    logging.info(f"Skipping ignored instantiation for suite: {suite}")
                    continue

                normalized_instantiation = instantiation_name.replace("\\", "").strip()
                if not TEST_TYPE_REGEX.match(normalized_instantiation):
                    errors.append(
                        f"{file_path}:{line}: Invalid TEST_TYPE '{instantiation_name}' in INSTANTIATE_TEST_SUITE_P."
                    )

                # For each suite, the instantiation names must be unique.
                if suite not in instantiations:
                    instantiations[suite] = {}
                if instantiation_name in instantiations[suite]:
                    prev_line = instantiations[suite][instantiation_name]
                    errors.append(
                        f"{file_path}:{line}: Duplicate INSTANTIATE_TEST_SUITE_P instantiation name '{instantiation_name}' for suite '{suite}' (previously defined at line {prev_line})."
                    )
                else:
                    instantiations[suite][instantiation_name] = line

            # === Process GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST macros ===
            for m in ALLOW_UNINSTANTIATED_REGEX.finditer(content):
                suite = m.group(1).strip()
                line = get_line_number(m.start())
                if suite in IGNORE_LIST:
                    logging.info(f"Skipping ignored allowed uninstantiated suite: {suite}")
                    continue
                allowed_uninstantiated.setdefault(suite, []).append(line)

            # === Validate TEST_P definitions have a corresponding instantiation or allowed exception ===
            for (suite, test_name), line in test_p_definitions.items():
                if suite not in instantiations and suite not in allowed_uninstantiated:
                    errors.append(
                        f"{file_path}:{line}: TEST_P '{suite}.{test_name}' does not have a matching INSTANTIATE_TEST_SUITE_P or GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST."
                    )

            # === Validate INSTANTIATE_TEST_SUITE_P references an existing TEST_P suite ===
            for suite, inst_map in instantiations.items():
                # If there is no TEST_P for this suite, then flag an error for each instantiation occurrence.
                if not any(suite == tp_suite for (tp_suite, _) in test_p_definitions.keys()):
                    for instantiation_name, line in inst_map.items():
                        errors.append(
                            f"{file_path}:{line}: INSTANTIATE_TEST_SUITE_P references non-existent TEST_P suite '{suite}'."
                        )

    return errors


def main():
    errors = analyze_tests(FOLDER_PATH)

    if errors:
        logging.error("The following issues were found:")
        for error in errors:
            logging.error(f"  {error}")
        raise ValueError("Validation failed. See the errors above.")
    else:
        logging.info("All tests meet the criteria.")


if __name__ == "__main__":
    main()
