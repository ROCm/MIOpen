# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CLANG_FORMAT_PRUNE -path "./build" -prune -o -path "./install" -prune -o -path "./fin" -prune -o)

# Note: The clang-format in /opt/rocm produces different results than the one in /usr/bin.  MIOpen
# formatting is based on the one in /usr/bin so we use that one
# set(CLANG_FORMAT_BINARY /opt/rocm/llvm/bin/clang-format)
set(CLANG_FORMAT_BINARY /usr/bin/clang-format-12)

add_custom_target(
    check_format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --dry-run --Werror --verbose {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)

add_custom_target(
    format
    COMMAND  find . ${CLANG_FORMAT_PRUNE} -regex ".*\\.\\(cpp\\|hpp\\|h.in\\|hpp.in\\|cpp.in\\|cl\\)" -exec ${CLANG_FORMAT_BINARY} --verbose -i {} +
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
