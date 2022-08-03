include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

message(STATUS "Fetching GoogleTest")

list(APPEND GTEST_CMAKE_CXX_FLAGS 
     -Wno-undef
     -Wno-reserved-identifier
     -Wno-global-constructors
     -Wno-missing-noreturn
     -Wno-disabled-macro-expansion
     -Wno-used-but-marked-unused
     -Wno-switch-enum
     -Wno-zero-as-null-pointer-constant
     -Wno-unused-member-function
     -Wno-comma
     -Wno-old-style-cast
)
message(STATUS "Suppressing googltest warnings with flags: ${GTEST_CMAKE_CXX_FLAGS}")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        e2239ee6043f73722e7aa812a459f54a28552929
)

# Will be necessary for windows build
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

target_compile_options(gtest PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
