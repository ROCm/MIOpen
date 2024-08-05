add_executable(sqlite2txt
        main.cpp
)

target_link_libraries(sqlite2txt SQLite::SQLite3 Threads::Threads)

if (NOT WIN32)
    target_link_libraries(sqlite2txt dl)
endif()

if(MIOPEN_USE_CLANG_TIDY)
  clang_tidy_check(sqlite2txt)
endif()
