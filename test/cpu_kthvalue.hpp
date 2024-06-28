#pragma once

#include "miopen/tensor.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <vector>

template <typename TIO>
void cpu_kthvalue(tensor<TIO> input,
                  tensor<TIO>& outputHost,
                  std::vector<size_t>& indices,
                  miopen::TensorDescriptor indiceDesc,
                  size_t k,
                  int dim)
{
    size_t inputSize       = input.desc.GetElementSize();
    size_t dimSize         = input.desc.GetLengths()[dim];
    size_t dimStride       = input.desc.GetStrides()[dim];
    auto inputTv           = miopen::get_inner_expanded_tv<5>(input.desc);
    auto inputTvWithoutDim = miopen::get_tv_without_dim<5, 4>(inputTv, dim);
    auto outputTv          = miopen::get_inner_expanded_tv<5>(outputHost.desc);
    auto indicesTv         = miopen::get_inner_expanded_tv<5>(indiceDesc);

    size_t numSlice = inputSize / dimSize;

    std::vector<float> elements;
    std::vector<size_t> ids(dimSize);
    for(int i = 0; i < dimSize; ++i)
    {
        ids[i] = i;
    }

    for(int slideID = 0; slideID < numSlice; ++slideID)
    {
        elements.clear();
        tensor_layout_t<4> layout(inputTvWithoutDim, slideID);
        auto idx = inputTvWithoutDim.get_tensor_view_idx(layout);

        for(int j = 0; j < dimSize; ++j)
        {
            elements.push_back(static_cast<float>(input[idx + j * dimStride]));
        }

        std::sort(ids.begin(), ids.end(), [=](size_t x, size_t y) -> bool {
            return elements[x] < elements[y];
        });
        auto output_layout  = tensor_layout_t<5>(outputTv, slideID);
        auto indices_layout = tensor_layout_t<5>(indicesTv, slideID);
        outputHost[outputTv.get_tensor_view_idx(output_layout)] =
            static_cast<TIO>(elements[ids[k - 1]]);
        indices[indicesTv.get_tensor_view_idx(indices_layout)] = ids[k - 1];
    }
}
