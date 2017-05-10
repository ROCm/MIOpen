#ifndef GCN_ASM_UTILS_H
#define GCN_ASM_UTILS_H

#include <string>
#include <vector>

std::string GetGcnAssemblerPath();
bool ValidateGcnAssembler();
int ExecuteGcnAssembler(std::vector<std::string>& args, std::istream* clang_stdin_content, std::ostream* clang_stdout_content);
void ExperimentalAmdgcnAssemble(std::string device_name, std::string& source, const std::string& params);

#endif //GCN_ASM_UTILS_H
