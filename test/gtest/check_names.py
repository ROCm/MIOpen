#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#################################################################################

"""gtest name linter"""
import re
import sys
import argparse
from collections import defaultdict
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("GTest name checker")

"""regexp based on https://github.com/ROCm/MIOpen/wiki/GTest-development#naming"""
re_prefix = re.compile(r"^((Smoke.*)|(Full.*)|(Perf.*)|(Unit.*))$")
re_hw = re.compile(r"^((CPU)|(GPU))$")
re_datatype = re.compile(
    r"^((FP((8)|(16)|(32)|(64)))|(BFP((8)|(16)))|(I((8)|(16)|(32)|(64)))|(NONE))\.?$"
)


def parse_args():
    """Function to parse cmd line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list",
        dest="list",
        type=str,
        required=True,
        help="Specify gtest test list file",
    )
    args = parser.parse_args()

    return args


def parse_tests(args):

    mismatches = defaultdict(str)

    with open(args.list) as fp:
        for line in fp.readlines()[2:]:
            if not line.strip():
                continue
            if line[0] == " ":
                continue
            line = line.split("#")[0].strip()

            full_name = line.split("/")

            if len(full_name) == 2:
                prefix = re.search(re_prefix, full_name[0])
                name = full_name[1].split("_")

                # Try to recognize pattern like HW_NAME_TYPE/<number>. This pattern is used in case of TYPED_TEST_SUITE
                if not prefix and full_name[1].replace(".", "").isnumeric():
                    name = full_name[0].split("_")
                    prefix = ["empty"]

            else:
                prefix = ["empty"]
                name = full_name[0].split("_")

            hw = re.search(re_hw, name[0])
            datatype = re.search(re_datatype, name[-1])
            if not prefix:
                mismatches[line] += " Prefix"
            if not hw:
                mismatches[line] += " Hw"
            if not datatype:
                mismatches[line] += " Datatype"
            if hw and hw.group() == "GPU" and datatype and ("NONE" in datatype.group()):
                mismatches[line] += " Hw and Datatype combination (GPU+NONE)"

        for l, k in mismatches.items():
            logger.warning("Name: " + l + " Mismatch types:" + k)

        if mismatches:
            logger.critical(
                "Tests do not match to the test naming scheme (see https://github.com/ROCm/MIOpen/wiki/GTest-development#naming )"
            )
            return -1  # uncomment when all the tests will be renamed
    return 0


def main():
    """Main function"""
    args = parse_args()
    return parse_tests(args)


if __name__ == "__main__":
    sys.exit(main())
