import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FOLDER_PATH = "../../test/gtest"

# Ignore list: Add test names or file paths you want to exclude
IGNORE_LIST = {
    "CPU_MIOpenDriverRegressionBigTensorTest_FP32",
    "../../test/gtest/binary_tensor_ops.cpp",
    "../../test/gtest/unary_tensor_ops.cpp",
}

# Valid enums and Regex for validation
VALID_HW_TYPES = {"CPU", "GPU"}
VALID_DATATYPES = {"FP8", "FP16", "FP32", "FP64", "BFP16", "BFP8", "I64", "I32", "I16", "I8", "NONE"}
TESTSUITE_REGEX = re.compile(
    r"^(CPU|GPU)_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_(" + "|".join(VALID_DATATYPES) + r")$"
)
TEST_P_REGEX = re.compile(r"\bTEST_P\(([^,]+),\s*([^)]+)\)")
INSTANTIATE_TEST_REGEX = re.compile(r"\bINSTANTIATE_TEST_SUITE_P\(\s*([^\n,]+),\s*([^\n,]+),")
ALLOW_UNINSTANTIATED_REGEX = re.compile(r"\bGTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST\(\s*([^\)]+)\)")
TEST_TYPE_REGEX = re.compile(r"^(Smoke|Full|Perf|Unit)([A-Za-z0-9]*)?$")


def analyze_tests(folder_path):
    errors = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)

                if file_path in IGNORE_LIST:
                    logging.info(f"Skipping ignored file: {file_path}")
                    continue

                with open(file_path, "r") as f:
                    content = f.read()

                # Extract TEST_P, INSTANTIATE_TEST_SUITE_P, and GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST
                test_p_matches = TEST_P_REGEX.findall(content)
                instantiate_matches = INSTANTIATE_TEST_REGEX.findall(content)
                allow_uninstantiated_matches = ALLOW_UNINSTANTIATED_REGEX.findall(content)

                test_p_suites = {suite: info for suite, info in test_p_matches}
                instantiated_suites = {suite: test_type for test_type, suite in instantiate_matches}
                allowed_uninstantiated_suites = set(allow_uninstantiated_matches)

                # Validate TEST_P suites
                for suite, info in test_p_suites.items():
                    if suite in IGNORE_LIST:
                        logging.info(f"Skipping ignored test suite: {suite}")
                        continue

                    if not TESTSUITE_REGEX.match(suite):
                        errors.append(f"{file_path}: Invalid TESTSUITE_NAME '{suite}' in TEST_P.")

                    if suite not in instantiated_suites and suite not in allowed_uninstantiated_suites:
                        errors.append(
                            f"{file_path}: Test '{suite}.{info}' does not have a matching "
                            f"INSTANTIATE_TEST_SUITE_P or GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST."
                        )

                # Validate instantiated suites
                for suite, test_type in instantiated_suites.items():
                    normalized_test_type = test_type.replace("\\", "").strip()

                    if suite in IGNORE_LIST:
                        logging.info(f"Skipping ignored instantiated suite: {suite}")
                        continue

                    if suite not in test_p_suites:
                        errors.append(f"{file_path}: INSTANTIATE_TEST_SUITE_P references non-existent TESTSUITE_NAME '{suite}'.")
                    if not TEST_TYPE_REGEX.match(normalized_test_type):
                        errors.append(f"{file_path}: Invalid TEST_TYPE '{test_type}' in INSTANTIATE_TEST_SUITE_P.")

                # Validate allowed uninstantiated suites
                for suite in allowed_uninstantiated_suites:
                    if suite in IGNORE_LIST:
                        logging.info(f"Skipping ignored allowed uninstantiated suite: {suite}")
                        continue

                    if suite not in test_p_suites:
                        errors.append(f"{file_path}: GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST references non-existent TESTSUITE_NAME '{suite}'.")

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