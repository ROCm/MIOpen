#!/bin/bash
find . -iname '*.h' \
    -o -iname '*.hpp' \
    -o -iname '*.cpp' \
    -o -iname '*.h.in' \
    -o -iname '*.hpp.in' \
    -o -iname '*.cpp.in' \
    -o -iname '*.cl' \
| grep -v -E '(build/)|(install/)' \
| xargs -n 1 -P $(nproc) -I{} -t clang-format-10 -style=file {} -i 