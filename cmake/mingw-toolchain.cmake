set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)
set(CMAKE_SYSTEM_NAME Windows)

# cross compilers to use for C and C++
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
set(CMAKE_RC_COMPILER ${TOOLCHAIN_PREFIX}-windres)

# target environment on the build host system
#   set 1st to dir with the cross compiler's C/C++ headers/libs
#   set 2nd to dir containing personal cross development headers/libs
set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX} /usr/local/${TOOLCHAIN_PREFIX})

# modify default behavior of FIND_XXX() commands to
# search for headers/libs in the target environment and
# search for programs in the build host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

include_directories(/usr/local/x86_64-w64-mingw32/include)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS On CACHE BOOL "Export windows symbols")

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -static-libgcc -static-libstdc++" CACHE STRING "")
