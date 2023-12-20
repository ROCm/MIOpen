
#pragma once

#include <miopen/invoke_params.hpp>

namespace miopen {
namespace matrixOps {

struct GemmAddActiv : public miopen::InvokeParams
{
    GemmAddActiv() = default;

    ConstData_t a_buff  = nullptr;
    ConstData_t b_buff  = nullptr;
    ConstData_t d0_buff = nullptr;
    Data_t e_buff       = nullptr;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};
} // namespace matrixOps

} // namespace miopen
