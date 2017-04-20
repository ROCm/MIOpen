cmake_minimum_required( VERSION 2.8.12 )

#add_executable(MIOpenDriver MIOpenDriver.cpp InputFlags.cpp)

add_executable(MIOpenDriver EXCLUDE_FROM_ALL main.cpp InputFlags.cpp)
target_link_libraries(MIOpenDriver MIOpen)
install(TARGETS MIOpenDriver 
    OPTIONAL 
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
