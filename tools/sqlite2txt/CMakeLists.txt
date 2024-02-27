add_executable(sqlite2txt
        main.cpp
)

target_link_libraries(sqlite2txt SQLite::SQLite3 pthread dl)
clang_tidy_check(sqlite2txt)
