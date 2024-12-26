/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/tensor.hpp>

#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_layout.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>

namespace miopen {

namespace {

bool IsDataTypeSupported(miopenDataType_t t)
{
    switch(t)
    {
    case miopenHalf:
    case miopenFloat:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenDouble:
    case miopenInt64: return true;
    }
    return false;
}

bool IsLayoutSupported(miopenTensorLayout_t layout, unsigned num_dims)
{
    switch(layout)
    {
    case miopenTensorNCHW:
    case miopenTensorNHWC:
    case miopenTensorCHWN:
    case miopenTensorNCHWc4:
    case miopenTensorNCHWc8:
    case miopenTensorCHWNc4:
    case miopenTensorCHWNc8: return num_dims == 4;
    case miopenTensorNCDHW:
    case miopenTensorNDHWC: return num_dims == 5;
    }

    return false;
}

// In this case, the "default layout" is the layout that needs to be set if the layout is not passed
// explicitly or implicitly.
std::optional<miopenTensorLayout_t> GetDefaultLayout(unsigned num_dims)
{
    switch(num_dims)
    {
    case 4: return miopenTensorNCHW;
    case 5: return miopenTensorNCDHW;
    default: return std::nullopt;
    }
}

template <class T>
bool CheckLengths(const std::vector<T>& lens, T maxval = 0)
{
    if(lens.empty())
        return false;
    if(!std::all_of(lens.cbegin(), lens.cend(), [](T x) { return x > 0; }))
        return false;
    if(maxval)
    {
        if(!std::all_of(lens.cbegin(), lens.cend(), [maxval](T x) { return x <= maxval; }))
            return false;
    }
    return true;
}

std::vector<std::size_t> ConvertLengthsOrThrow(const std::vector<int>& lens_in,
                                               [[maybe_unused]] const std::string& err_msg)
{
    if(!CheckLengths(lens_in))
        MIOPEN_THROW(miopenStatusBadParm, err_msg);

    std::vector<std::size_t> lens(lens_in.cbegin(), lens_in.cend());
    return lens;
}

std::string GetStorageLayout4D5D(unsigned num_dims, bool is_CHWNc = false)
{
    // For some reason we have CHWN storage layout for CHWNc
    if(is_CHWNc)
        return "CHWN";

    switch(num_dims)
    {
    case 4: return "NCHW";
    case 5: return "NCDHW";
    default: MIOPEN_THROW(miopenStatusInternalError);
    }
}

// Relevant for NCHWc and CHWNc
std::size_t GetVectorLengthForLayout(const std::optional<miopenTensorLayout_t>& layout)
{
    std::size_t vector_length = 1;

    if(layout)
    {
        switch(layout.value())
        {
        case miopenTensorCHWNc8:
        case miopenTensorNCHWc8: vector_length = 8; break;
        case miopenTensorCHWNc4:
        case miopenTensorNCHWc4: vector_length = 4; break;
        default: break;
        }
    }

    return vector_length;
}

void ReorderVector(std::vector<size_t>& lens, const std::initializer_list<size_t>& indices)
{
    std::vector<size_t> out_lens;
    out_lens.reserve(indices.size());
    for(size_t index : indices)
    {
        assert(index < lens.size());
        out_lens.push_back(lens[index]);
    }
    lens = std::move(out_lens);
}

// Relevant for NCHWc and CHWNc
void VectLensReorder(miopenTensorLayout_t layout, std::vector<size_t>& lens)
{
    switch(layout)
    {
    case miopenTensorNCHWc4:
    case miopenTensorNCHWc8:
        // Do nothing, MIOpen implicit logic that lens are in NCHW order.
        break;
    case miopenTensorCHWNc4:
    case miopenTensorCHWNc8:
        // For some reason we have CHWN storage layout for CHWNc
        ReorderVector(lens, {1, 2, 3, 0});
        break;
    default: break;
    }
}

// Relevant for NCHWc and CHWNc
void VectLensRecalc(miopenTensorLayout_t layout,
                    std::size_t vector_length,
                    std::vector<size_t>& lens)
{
    unsigned c_pos;

    switch(layout)
    {
    case miopenTensorNCHWc4:
    case miopenTensorNCHWc8: c_pos = 1; break;
    case miopenTensorCHWNc4:
    case miopenTensorCHWNc8:
        // For some reason we have CHWN storage layout for CHWNc
        c_pos = 0;
        break;
    default: return;
    }

    if(lens[c_pos] % vector_length != 0)
        MIOPEN_THROW(miopenStatusBadParm, "Wrong C, C % Vect != 0");
    lens[c_pos] /= vector_length;
}

void CalculateStrides(std::size_t vector_length,
                      const std::vector<size_t>& lens,
                      std::vector<size_t>& strides)
{
    if(lens.empty())
        MIOPEN_THROW(miopenStatusInternalError);
    strides.clear();
    strides.resize(lens.size(), 0);
    strides.back() = vector_length;
    std::partial_sum(
        lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
    for(int i = 0; i < strides.size() - 1; i++)
        strides[i] *= vector_length;
}

void SetStrides(const std::optional<miopenTensorLayout_t>& layout,
                std::size_t vector_length,
                const std::vector<size_t>& lens,
                std::vector<size_t>& strides)
{
    const bool is_vectorized = vector_length > 1;
    if(!layout || layout == miopenTensorNCHW || layout == miopenTensorNCDHW || is_vectorized)
    {
        CalculateStrides(vector_length, lens, strides);
    }
    else
    {
        const auto num_dims       = lens.size();
        const auto storage_layout = GetStorageLayout4D5D(num_dims);
        const auto layout_str     = TensorDescriptor::LayoutEnumToStr(layout.value());
        tensor_layout_to_strides(lens, storage_layout, layout_str, strides);
    }
}

bool CheckDimsFitIntoInt(const std::vector<std::size_t>& v)
{
    if(std::any_of(
           v.cbegin(), v.cend(), [](std::size_t x) { return x > std::numeric_limits<int>::max(); }))
    {
        return false;
    }
    return true;
}

} // namespace

TensorDescriptor::TensorDescriptor() : packed(true) {}

TensorDescriptor::TensorDescriptor(miopenDataType_t t) : packed(true), type(t) {}

// The delegation constructor should be placed above the target constructor in the
// code for better dependency tracking

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const std::initializer_list<int>& lens_in)
    : TensorDescriptor(t, std::vector<int>(lens_in))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const std::vector<int>& lens_in)
    : TensorDescriptor(t,
                       GetDefaultLayout(lens_in.size()),
                       ConvertLengthsOrThrow(lens_in, "Lengths must be > 0"),
                       {},
                       false)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::initializer_list<std::size_t>& lens_in)
    : TensorDescriptor(t, std::vector<std::size_t>(lens_in))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const std::vector<std::size_t>& lens_in)
    : TensorDescriptor(t, GetDefaultLayout(lens_in.size()), lens_in, {}, false)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, std::vector<std::size_t>&& lens_in)
    : TensorDescriptor(t, GetDefaultLayout(lens_in.size()), std::move(lens_in), {}, false)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   const std::vector<int>& lens_in)
    : TensorDescriptor(t, layout_in, ConvertLengthsOrThrow(lens_in, "Lengths must be > 0"))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   const std::initializer_list<std::size_t>& lens_in)
    : TensorDescriptor(t, layout_in, std::vector<std::size_t>(lens_in))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   const std::vector<std::size_t>& lens_in)
    : TensorDescriptor(t, layout_in, lens_in, {}, false)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   std::vector<std::size_t>&& lens_in)
    : TensorDescriptor(t, layout_in, std::move(lens_in), {}, false)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::vector<int>& lens_in,
                                   const std::vector<int>& strides_in)
    : TensorDescriptor(t,
                       ConvertLengthsOrThrow(lens_in, "Lengths must be > 0"),
                       ConvertLengthsOrThrow(strides_in, "Strides must be > 0"))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::initializer_list<std::size_t>& lens_in,
                                   const std::initializer_list<std::size_t>& strides_in)
    : TensorDescriptor(t, std::vector<std::size_t>(lens_in), std::vector<std::size_t>(strides_in))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::vector<std::size_t>& lens_in,
                                   const std::vector<std::size_t>& strides_in)
    : TensorDescriptor(t, std::nullopt, lens_in, strides_in, true)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   std::vector<std::size_t>&& lens_in,
                                   std::vector<std::size_t>&& strides_in)
    : TensorDescriptor(t, std::nullopt, std::move(lens_in), std::move(strides_in), true)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   const std::vector<std::size_t>& lens_in,
                                   const std::vector<std::size_t>& strides_in)
    : TensorDescriptor(t, layout_in, lens_in, strides_in, true)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   std::vector<std::size_t>&& lens_in,
                                   std::vector<std::size_t>&& strides_in)
    : TensorDescriptor(t, layout_in, std::move(lens_in), std::move(strides_in), true)
{
}

// Main private constructor
TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::optional<miopenTensorLayout_t>& layout_in,
                                   const std::vector<std::size_t>& lens_in,
                                   const std::vector<std::size_t>& strides_in,
                                   bool use_strides)
    : lens(lens_in),
      strides(use_strides ? strides_in : std::vector<std::size_t>()),
      type(t),
      tensorLayout(layout_in)
{
    this->CheckArgsAndInit(use_strides);
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::optional<miopenTensorLayout_t>& layout_in,
                                   std::vector<std::size_t>&& lens_in,
                                   std::vector<std::size_t>&& strides_in,
                                   bool use_strides)
    : lens(std::move(lens_in)),
      strides(use_strides ? std::move(strides_in) : std::vector<std::size_t>()),
      type(t),
      tensorLayout(layout_in)
{
    this->CheckArgsAndInit(use_strides);
}

void TensorDescriptor::CheckArgsAndInit(bool use_strides)
{
    if(!IsDataTypeSupported(type))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported data type");

    if(lens.empty())
        MIOPEN_THROW(miopenStatusBadParm, "Number of dimensions must be > 1");

    if(tensorLayout && !IsLayoutSupported(tensorLayout.value(), lens.size()))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported layout");

    if(!CheckLengths(lens, static_cast<std::size_t>(std::numeric_limits<int64_t>::max())))
        MIOPEN_THROW(miopenStatusBadParm, "Lengths must be > 0 and <= INT64_MAX");

    vector_length = GetVectorLengthForLayout(tensorLayout);

    if(use_strides)
    {
        if(lens.size() != strides.size())
            MIOPEN_THROW(miopenStatusBadParm, "Lengths and strides dimensions must be equal");

        if(!CheckLengths(strides, static_cast<std::size_t>(std::numeric_limits<int64_t>::max())))
            MIOPEN_THROW(miopenStatusBadParm, "Strides must be > 0 and <= INT64_MAX");

        packed = (this->GetElementSize() == this->GetElementSpace());

        if(tensorLayout)
        {
            if(!this->IsPossibleLayout4D5D(TensorDescriptor::LayoutEnumToStr(tensorLayout.value())))
                MIOPEN_THROW(miopenStatusBadParm, "Mismatch of layout and strides");
        }
    }
    else
    {
        packed = true;

        if(this->IsVectorized())
        {
            // clang-tidy: bugprone-unchecked-optional-access
            if(!tensorLayout)
                MIOPEN_THROW(miopenStatusInternalError);
            VectLensReorder(tensorLayout.value(), lens);
            VectLensRecalc(tensorLayout.value(), vector_length, lens);
        }

        SetStrides(tensorLayout, vector_length, lens, strides);
    }
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t, const int* plens, int size)
{
    if(plens == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t, std::vector<int>(plens, plens + size)};
}

TensorDescriptor
TensorDescriptor::MakeDescriptor(miopenDataType_t t, const std::size_t* plens, int size)
{
    if(plens == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t, std::vector<std::size_t>(plens, plens + size)};
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t,
                                                  miopenTensorLayout_t layout,
                                                  const int* plens,
                                                  int size)
{
    if(plens == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t, layout, std::vector<int>(plens, plens + size)};
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t,
                                                  miopenTensorLayout_t layout,
                                                  const std::size_t* plens,
                                                  int size)
{
    if(plens == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t, layout, std::vector<std::size_t>(plens, plens + size)};
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t,
                                                  const int* plens,
                                                  const int* pstrides,
                                                  int size)
{
    if(plens == nullptr || pstrides == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t, std::vector<int>(plens, plens + size), std::vector<int>(pstrides, pstrides + size)};
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t,
                                                  const std::size_t* plens,
                                                  const std::size_t* pstrides,
                                                  int size)
{
    if(plens == nullptr || pstrides == nullptr || size <= 0)
        MIOPEN_THROW(miopenStatusInvalidValue);

    return {t,
            std::vector<std::size_t>(plens, plens + size),
            std::vector<std::size_t>(pstrides, pstrides + size)};
}

bool TensorDescriptor::IsVectorized() const { return vector_length > 1; }

const std::vector<std::size_t>& TensorDescriptor::GetLengths() const { return lens; }

const std::vector<std::size_t>& TensorDescriptor::GetStrides() const { return strides; }

unsigned TensorDescriptor::GetNumDims() const { return lens.size(); }

std::size_t TensorDescriptor::GetElementSize() const
{
    return std::accumulate(lens.begin(), lens.end(), vector_length, std::multiplies<std::size_t>());
}

miopenDataType_t TensorDescriptor::GetType() const { return this->type; }

std::optional<miopenDataType_t> TensorDescriptor::GetCastType() const { return this->cast_type; }

void TensorDescriptor::SetCastType(const miopenDataType_t cast_type_)
{
    this->cast_type = cast_type_;
}

// Deprecated
miopenTensorLayout_t TensorDescriptor::GetLayout_t() const
{
    const auto layout = this->GetLayoutEnum();
    if(layout)
        return layout.value();

    MIOPEN_THROW(miopenStatusInternalError, "Unknown layout");
}

const std::optional<miopenTensorLayout_t>& TensorDescriptor::GetLayoutEnum() const
{
    if(!cached_layout_enum_calculated)
    {
        cached_layout_enum = [&]() -> std::optional<miopenTensorLayout_t> {
            if(tensorLayout)
                return tensorLayout;

            const auto known_layouts = {std::make_pair("NCHW", miopenTensorNCHW),
                                        std::make_pair("NHWC", miopenTensorNHWC),
                                        std::make_pair("NCDHW", miopenTensorNCDHW),
                                        std::make_pair("NDHWC", miopenTensorNDHWC),
                                        std::make_pair("CHWN", miopenTensorCHWN)};
            for(const auto& [layout_str, layout_enum] : known_layouts)
            {
                if(this->IsPossibleLayout4D5D(layout_str))
                    return layout_enum;
            }

            return std::nullopt;
        }();

        cached_layout_enum_calculated = true;
    }

    return cached_layout_enum;
}

std::string TensorDescriptor::LayoutEnumToStr(miopenTensorLayout_t layout)
{
    switch(layout)
    {
    case miopenTensorNCHW: return "NCHW";
    case miopenTensorNHWC: return "NHWC";
    case miopenTensorNCHWc4:
    case miopenTensorNCHWc8: return "NCHWc";
    case miopenTensorCHWN: return "CHWN";
    case miopenTensorCHWNc4:
    case miopenTensorCHWNc8: return "CHWNc";
    case miopenTensorNCDHW: return "NCDHW";
    case miopenTensorNDHWC: return "NDHWC";
    default: MIOPEN_THROW(miopenStatusInternalError, "Unknown layout");
    }
}

const std::string& TensorDescriptor::GetLayout_str() const
{
    if(cached_layout_str.empty())
    {
        cached_layout_str = [&]() -> std::string {
            if(tensorLayout)
                return TensorDescriptor::LayoutEnumToStr(tensorLayout.value());

            switch(this->GetNumDims())
            {
            case 4:
            case 5: return this->GetLayout(GetStorageLayout4D5D(this->GetNumDims()));
            default: return "UNKNOWN";
            }
        }();
    }

    return cached_layout_str;
}

std::size_t TensorDescriptor::GetVectorLength() const { return this->vector_length; }

std::size_t TensorDescriptor::GetIndex(std::initializer_list<int> l) const
{
    // l is in NCHW order (MIOpen implicit logic)
    if(tensorLayout == miopenTensorCHWNc4 || tensorLayout == miopenTensorCHWNc8)
    {
        assert(l.size() - 1 <= this->GetNumDims());
        std::initializer_list<int> l_chwn{
            *(l.begin()), *(l.begin() + 2), *(l.begin() + 3), *(l.begin() + 4), *(l.begin() + 1)};
        return std::inner_product(l_chwn.begin() + 1,
                                  l_chwn.end(),
                                  strides.begin(),
                                  static_cast<std::size_t>(*(l_chwn.begin())));
    }
    else
    {
        if(!this->IsVectorized())
        {
            assert(l.size() <= this->GetNumDims());
            return std::inner_product(l.begin(), l.end(), strides.begin(), std::size_t{0});
        }
        else
        {
            assert(l.size() - 1 <= this->GetNumDims());
            return std::inner_product(
                l.begin() + 1, l.end(), strides.begin(), static_cast<std::size_t>(*(l.begin())));
        }
    }
}

std::size_t TensorDescriptor::GetElementSpace() const
{
    return std::inner_product(lens.begin(),
                              lens.end(),
                              strides.begin(),
                              vector_length,
                              std::plus<size_t>(),
                              [](size_t len, size_t stride) { return (len - 1) * stride; });
}

// For vectorized layouts storage_layout must be without the ending 'c'
bool TensorDescriptor::IsPossibleLayout(const std::string& storage_layout,
                                        const std::string& layout) const
{
    if(storage_layout.size() != this->GetNumDims())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "storage_layout.size() must be equal to the number of the tensor dimensions");
    }

    auto layout_vect = (*(layout.end() - 1) == 'c');
    if(this->IsVectorized() != layout_vect)
        return false;

    auto layout_size = this->GetNumDims();
    if(this->IsVectorized())
        layout_size++; // last char (c)
    if(layout.size() != layout_size)
        return false;

    const auto base_layout = layout.substr(0, this->GetNumDims());

    if(this->GetNumDims() < 2)
    {
        if(storage_layout != base_layout)
            MIOPEN_THROW(miopenStatusInternalError, "storage_layout and layout mismatch");
        return true;
    }

    auto op = [&](char cur_char) {
        const auto pos = storage_layout.find(cur_char);
        if(pos == std::string::npos)
            MIOPEN_THROW(miopenStatusInternalError, "wrong layout format");
        return strides[pos];
    };

    std::vector<std::size_t> layout_strides(base_layout.size());
    std::transform(base_layout.cbegin(), base_layout.cend(), layout_strides.begin(), op);

    // Check monotonic decreasing
    for(unsigned i = 0; i < (layout_strides.size() - 1); i++)
    {
        if(layout_strides[i] < layout_strides[i + 1])
            return false;
    }

    return true;
}

// Layout could be NCHW, NHWC, NCDHW, NDHWC, NCHWc, ...
bool TensorDescriptor::IsPossibleLayout4D5D(const std::string& layout) const
{
    if(tensorLayout)
    {
        if(this->tensorLayout == miopenTensorCHWNc4 || this->tensorLayout == miopenTensorCHWNc8)
            return this->IsPossibleLayout(GetStorageLayout4D5D(4, true), layout);
    }

    switch(this->GetNumDims())
    {
    case 4:
    case 5: return this->IsPossibleLayout(GetStorageLayout4D5D(this->GetNumDims()), layout);
    default: return false;
    }
}

// See https://github.com/ROCm/MIOpen/pull/765#discussion_r596465551
std::vector<int64_t> TensorDescriptor::find_permutation(const std::vector<std::size_t>& lens,
                                                        const std::vector<std::size_t>& strides)
{
    std::vector<int64_t> result(lens.size());
    std::iota(result.begin(), result.end(), 0);
    std::stable_sort(result.begin(), result.end(), by(std::greater<>{}, [&](auto x) {
                         return std::make_tuple(strides[x], lens[x]);
                     }));
    return result;
}

// storage_layout must be NCHW or NCHWc for NCHWc, CHWN or CHWNc for CHWNc, NCHW for other 4D
// layouts, NCDHW for 5D layouts
std::string TensorDescriptor::GetLayout(std::string storage_layout) const
{
    const bool is_vectorized_sl = (*(storage_layout.end() - 1) == 'c');
    if(is_vectorized_sl && !this->IsVectorized())
    {
        MIOPEN_THROW(miopenStatusInternalError, "Invalid storage_layout");
    }

    const std::string base_storage_layout =
        is_vectorized_sl ? storage_layout.substr(0, storage_layout.size() - 1) : storage_layout;
    if(base_storage_layout.size() != strides.size())
    {
        MIOPEN_THROW("Invalid storage_layout size. storage_layout size must be equavalent to the "
                     "stride size");
    }

    // Copy construct the result string from storage_layout. This allocates the space at one go
    // and is faster than calling push_back in transform.
    auto result = base_storage_layout;

    if(cached_permutation.size() == 0)
        cached_permutation = find_permutation(lens, strides);
    const auto& p = cached_permutation;

    std::transform(
        p.cbegin(), p.cend(), result.begin(), [&](auto i) { return base_storage_layout[i]; });

    if(this->IsVectorized())
        result += 'c';

    return result;
}

std::size_t TensorDescriptor::GetNumBytes() const
{
    std::size_t typesize = GetTypeSize(this->type);
    return typesize * this->GetElementSpace();
}

bool TensorDescriptor::IsPacked() const { return this->packed; }

bool TensorDescriptor::IsContiguous() const
{
    size_t plane_size    = 1;
    size_t dims_of_shape = lens.size();

    for(int index = dims_of_shape - 1; index >= 0; --index)
    {
        if((lens[index] != 1) && (strides[index] != plane_size))
        {
            return false;
        }
        plane_size *= lens[index];
    }
    return true;
}

bool TensorDescriptor::AllLengthsFitIntoInt() const
{
    if(!cached_lengths_fit_into_int)
        cached_lengths_fit_into_int = CheckDimsFitIntoInt(lens);

    return cached_lengths_fit_into_int.value();
}

bool TensorDescriptor::AllDimsFitIntoInt() const
{
    if(!this->AllLengthsFitIntoInt())
        return false;

    if(!cached_strides_fit_into_int)
        cached_strides_fit_into_int = CheckDimsFitIntoInt(strides);

    return cached_strides_fit_into_int.value();
}

bool TensorDescriptor::operator==(const TensorDescriptor& rhs) const
{
    assert(this->lens.size() == rhs.strides.size());
    return this->type == rhs.type && this->lens == rhs.lens && this->strides == rhs.strides;
}

bool TensorDescriptor::operator!=(const TensorDescriptor& rhs) const { return !(*this == rhs); }

bool TensorDescriptor::operator<(const TensorDescriptor& rhs) const
{
    return (std::tie(this->GetLengths(), this->GetStrides()) <
            std::tie(rhs.GetLengths(), rhs.GetStrides()));
}

bool TensorDescriptor::operator>(const TensorDescriptor& rhs) const
{
    return (std::tie(this->GetLengths(), this->GetStrides()) >
            std::tie(rhs.GetLengths(), rhs.GetStrides()));
}

std::string TensorDescriptor::ToString() const
{
    std::string result;
    if(this->lens.empty())
        return result;
    for(auto i : this->lens)
    {
        result += std::to_string(i) + ", ";
    }
    return result.substr(0, result.length() - 2);
}

std::ostream& operator<<(std::ostream& stream, const TensorDescriptor& t)
{
    LogRange(stream << "{", t.lens, ", ") << "}, ";
    LogRange(stream << "{", t.strides, ", ") << "}, ";
    if(t.packed)
    {
        stream << "packed"
               << ", ";
    }

    if(t.cast_type)
    {
        stream << "cast_type: ";
        const auto ct = *t.cast_type;
        if(ct == miopenFloat8)
            stream << "miopenFloat8";
        else if(ct == miopenBFloat8)
            stream << "miopenBFloat8";
        else
            stream << "Other";
    }

    return stream;
}

void to_json(nlohmann::json& j, const TensorDescriptor& descriptor)
{
    j = nlohmann::json{
        {"lengths", descriptor.lens},
        {"strides", descriptor.strides},
        {"packed", descriptor.packed},
        {"type", descriptor.type},
    };
}

void from_json(const nlohmann::json& j, TensorDescriptor& descriptor)
{
    j.at("lengths").get_to(descriptor.lens);
    j.at("strides").get_to(descriptor.strides);
    j.at("packed").get_to(descriptor.packed);
    j.at("type").get_to(descriptor.type);
}

} // namespace miopen

int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices)
{
    return miopen::deref(tensorDesc).GetIndex(indices);
}
