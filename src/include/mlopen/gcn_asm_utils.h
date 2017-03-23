#ifndef GCN_ASM_UTILS_H
#define GCN_ASM_UTILS_H

#include <string>
#include <vector>

std::string GetGcnAssemblerPath();
bool ValidateGcnAssembler();
int ExecuteGcnAssembler(std::string& path, std::vector<std::string>& args, std::istream* clang_stdin_content = nullptr, std::ostream* clang_stdout_content = nullptr);
int ExecuteGcnAssembler(std::vector<std::string>& args, std::istream* clang_stdin_content = nullptr, std::ostream* clang_stdout_content = nullptr);
void CleanExecutablePath(std::string& path);

#endif
