#ifndef CK_MLIR_UTIL_HPP_
#define CK_MLIR_UTIL_HPP_

namespace miopen {
namespace solver {

static inline std::string InsertGToLayout(const std::string& layout, char dim)
{
    std::string layout_with_g = layout;
    std::size_t index         = layout.find(dim);
    if(index == std::string::npos)
        MIOPEN_THROW(std::string("Failed to find dim '") + dim + "' in the layout " + layout);
    return layout_with_g.insert(index, 1, 'G');
}

} // namespace solver
} // namespace miopen

#endif
