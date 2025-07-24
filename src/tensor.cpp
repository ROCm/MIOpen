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
#include <miopen/float_equal.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/datatype.hpp>
#include <miopen/tensorOp/invoke_params.hpp>
#include <miopen/tensorOp/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/util.hpp>

#include <boost/range/combine.hpp>

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
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
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
            if(!this->IsPossibleLayout4D5D(TensorDescriptor::LayoutEnumToStr(tensorLayout.value()),
                                           LayoutValidationMode::IgnoreDegenerateStrides))
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

            auto layout = GetLayout_str();

            try
            {
                return StringToLayoutType(layout, IsVectorized(), vector_length);
            }
            catch(const miopen::Exception& e)
            {
                // If the layout cannot be determined by the string, then we
                // can fall back to the known layouts to check if they are applicable.
                static const auto known_layouts = {std::make_pair("NCHW", miopenTensorNCHW),
                                                   std::make_pair("NHWC", miopenTensorNHWC),
                                                   std::make_pair("NCDHW", miopenTensorNCDHW),
                                                   std::make_pair("NDHWC", miopenTensorNDHWC),
                                                   std::make_pair("CHWN", miopenTensorCHWN)};
                for(const auto& [layout_str, layout_enum] : known_layouts)
                {
                    if(IsPossibleLayout4D5D(layout_str,
                                            LayoutValidationMode::IgnoreDegenerateStrides))
                    {
                        return layout_enum;
                    }
                }

                MIOPEN_LOG_W("Failed to convert layout string '" << layout
                                                                 << "' to enum: " << e.what());
                return std::nullopt;
            }
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
                                        const std::string& layout,
                                        LayoutValidationMode validationMode) const
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

    // Build layout_strides using the provided validation mode, storage_layout, and layout.
    // If we are using IgnoreDegenerateStrides, then we are ignoring the strides when lengths == 1.
    // E.G NCHW layout with lens = {5, 1, 10, 10} Is actually NHW since there is no
    // channels dimension. Both NHWC & NCHW layouts are valid for this tensor as channels is not
    // relevant.
    std::vector<std::size_t> layout_strides;
    layout_strides.reserve(base_layout.size());
    for(const auto& cur_char : base_layout)
    {
        const auto pos = storage_layout.find(cur_char);
        if(pos == std::string::npos)
            MIOPEN_THROW(miopenStatusInternalError, "wrong layout format");

        switch(validationMode)
        {
        case LayoutValidationMode::IgnoreDegenerateStrides:
            if(lens[pos] == 1)
            {
                continue;
            }
            break;
        case LayoutValidationMode::StrictDecreasingStrides: break;
        default: MIOPEN_THROW(miopenStatusInternalError, "Unknown validation mode provided");
        }

        layout_strides.push_back(strides[pos]);
    }

    // Check monotonic decreasing
    for(size_t i = 1; i < layout_strides.size(); ++i)
    {
        if(layout_strides[i - 1] < layout_strides[i])
            return false;
    }
    return true;
}

// Layout could be NCHW, NHWC, NCDHW, NDHWC, NCHWc, ...
bool TensorDescriptor::IsPossibleLayout4D5D(const std::string& layout,
                                            LayoutValidationMode validationMode) const
{
    if(tensorLayout)
    {
        if(this->tensorLayout == miopenTensorCHWNc4 || this->tensorLayout == miopenTensorCHWNc8)
            return this->IsPossibleLayout(GetStorageLayout4D5D(4, true), layout, validationMode);
    }

    switch(this->GetNumDims())
    {
    case 4:
    case 5:
        return this->IsPossibleLayout(
            GetStorageLayout4D5D(this->GetNumDims()), layout, validationMode);
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

miopenTensorLayout_t
TensorDescriptor::StringToLayoutType(std::string layout_str, bool vectorized, int vector_length)
{
    if(vectorized)
    {
        if(vector_length == 4)
        {
            return layout_str == "CHWNc" ? miopenTensorCHWNc4 : miopenTensorNCHWc4;
        }
        else if(vector_length == 8)
        {
            return layout_str == "CHWNc" ? miopenTensorCHWNc8 : miopenTensorNCHWc8;
        }
        else
        {
            MIOPEN_THROW("C-vectorized tensor only support vector length 4 and 8");
        }
    }
    else
    {
        if(layout_str == "NCHW")
        {
            return miopenTensorNCHW;
        }
        else if(layout_str == "NHWC")
        {
            return miopenTensorNHWC;
        }
        else if(layout_str == "NDHWC")
        {
            return miopenTensorNDHWC;
        }
        else if(layout_str == "NCDHW")
        {
            return miopenTensorNCDHW;
        }
        else if(layout_str == "CHWN")
        {
            return miopenTensorCHWN;
        }
        else
        {
            MIOPEN_THROW("Non-vectorized tensor only support layout NCHW, NHWC, NCDHW and NDHWC");
        }
    }
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
        if(ct == miopenFloat8_fnuz)
            stream << "miopenFloat8_fnuz";
        else if(ct == miopenBFloat8_fnuz)
            stream << "miopenBFloat8_fnuz";
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

TensorDescriptor GetFlattenedTensorDescriptor(const TensorDescriptor& desc)
{
    // is packed
    if(desc.IsPacked())
        return {desc.GetType(), {desc.GetElementSize()}, {static_cast<std::size_t>(1)}};

    // start flattening tensor
    std::vector<std::size_t> flat_lengths;
    std::vector<std::size_t> flat_strides;

    auto non1_length_strides = boost::combine(desc.GetLengths(), desc.GetStrides()) |
                               boost::adaptors::filtered(f_length_is_not_1_t());

    auto i               = non1_length_strides.begin();
    std::size_t flat_len = boost::get<0>(*i);
    auto i_previous      = i++;

    // the 0-th dimension full-length doesn't matter
    for(; i != non1_length_strides.end(); ++i)
    {
        std::size_t len             = boost::get<0>(*i);
        std::size_t stride          = boost::get<1>(*i);
        std::size_t previous_stride = boost::get<1>(*i_previous);
        std::size_t full_len        = previous_stride / stride;

        if(len == full_len)
        {
            flat_len *= len;
        }
        else
        {
            flat_lengths.push_back(flat_len);
            flat_strides.push_back(previous_stride);
            flat_len = len;
        }
        i_previous = i;
    }
    flat_lengths.push_back(flat_len);
    flat_strides.push_back(boost::get<1>(*i_previous));

    return {desc.GetType(), flat_lengths, flat_strides};
}

struct two_exp_ceiling_t
{
    std::size_t operator()(std::size_t n) const
    {
        assert(n > 0);

        std::size_t i = 1;

        n--;
        while(n != 0)
        {
            i *= 2;
            n /= 2;
        }

        return i;
    }
};

static std::vector<std::size_t> get_worker_sizes(const std::vector<std::size_t>& data_sizes)
{
    const std::size_t dim = data_sizes.size();

    std::vector<std::size_t> worker_sizes(dim);

    std::transform(data_sizes.begin(), data_sizes.end(), worker_sizes.begin(), two_exp_ceiling_t{});

    std::size_t wgd = std::accumulate(
        worker_sizes.begin(), worker_sizes.end(), std::size_t{1}, std::multiplies<std::size_t>());

    if(wgd > 65536)
    {
        std::size_t n = wgd / 65536;

        int i = 0;
        while(n > 1 && i < dim)
        {
            std::size_t size_old = worker_sizes[i];
            worker_sizes[i]      = (size_old - 1) / n + 1;
            n /= size_old / worker_sizes[i];
            ++i;
        }
    }

    return worker_sizes;
}

void SetTensor(const Handle& handle,
               const TensorDescriptor& yDesc,
               Data_t y,
               const void* alpha,
               const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const TensorDescriptor yDesc_flat = GetFlattenedTensorDescriptor(yDesc);

#ifndef NDEBUG
    if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("real descriptor: " << yDesc);
        MIOPEN_LOG_I2("flat descriptor: " << yDesc_flat);
    }
#endif

    const std::size_t yDim_flat = yDesc_flat.GetNumDims();

    assert(yDim_flat > 0 && yDim_flat <= 5);

    std::string kernel_name = "SubTensorOpWithScalar" + std::to_string(yDim_flat) + "d";

    const miopenDataType_t dataType = yDesc_flat.GetType();

    std::string network_config = "set " + std::to_string(dataType);
    for(auto& len : yDesc_flat.GetLengths())
    {
        network_config += " " + std::to_string(len);
    }

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    KernelInvoke kernel;

    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
        std::string program_name = "MIOpenSubTensorOpWithScalarKernel.cl";

        std::vector<std::size_t> worker_sizes = get_worker_sizes(yDesc_flat.GetLengths());

        std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                          worker_sizes.end(),
                                          std::size_t{1},
                                          std::multiplies<std::size_t>());

        std::size_t wld = 256 < wgd ? 256 : wgd;
        std::stringstream ss;
        ss << "-DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_SET"
           << GetDataTypeKernelParams(dataType);
        for(int i = 0; i < yDim_flat; ++i)
        {
            ss << " -DWORK_LENGTH_" << std::to_string(i) << "=" << std::to_string(worker_sizes[i]);
        }

        kernel = handle.AddKernel(kernel_name,
                                  network_config,
                                  program_name,
                                  kernel_name,
                                  {wld, 1, 1},
                                  {wgd, 1, 1},
                                  ss.str());
    }

    switch(yDim_flat)
    {
    case 1: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]));
        });

        break;
    }
    case 2: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]));
        });

        break;
    }
    case 3: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]));
        });

        break;
    }
    case 4: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]));
        });

        break;
    }
    case 5: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetStrides()[4]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[4]));
        });

        break;
    }
    default: assert(false);
    }
}

void ScaleTensor(const Handle& handle,
                 const TensorDescriptor& yDesc,
                 Data_t y,
                 const void* alpha,
                 const int offset)
{
    if(y == nullptr || alpha == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const TensorDescriptor yDesc_flat = GetFlattenedTensorDescriptor(yDesc);

#ifndef NDEBUG
    if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("real descriptor: " << yDesc);
        MIOPEN_LOG_I2("flat descriptor: " << yDesc_flat);
    }
#endif

    const std::size_t yDim_flat = yDesc_flat.GetNumDims();

    assert(yDim_flat > 0 && yDim_flat <= 5);

    const miopenDataType_t dataType = yDesc_flat.GetType();

    if(!(dataType == miopenHalf     //
         || dataType == miopenFloat //
         || dataType == miopenInt32 //
         || dataType == miopenDouble))
    {
        MIOPEN_THROW(miopenStatusBadParm, "ScaleTensor: unsupported data type.");
    }

    std::string kernel_name = "SubTensorOpWithScalar" + std::to_string(yDim_flat) + "d";

    const std::vector<std::size_t>& lens = yDesc_flat.GetLengths();

    std::string network_config = "scale " + std::to_string(yDesc_flat.GetType());
    for(auto& len : lens)
    {
        network_config += " " + std::to_string(len);
    }

    auto&& kernels = handle.GetKernels(kernel_name, network_config);

    KernelInvoke kernel;

    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
        std::string program_name = "MIOpenSubTensorOpWithScalarKernel.cl";

        std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

        std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                          worker_sizes.end(),
                                          std::size_t{1},
                                          std::multiplies<std::size_t>());

        std::size_t wld = 256 < wgd ? 256 : wgd;

        std::string parms = "-DSUBTENSOR_OP_WITH_SCALAR=SUBTENSOR_OP_WITH_SCALAR_MULTIPLY" +
                            GetDataTypeKernelParams(dataType);
        for(int i = 0; i < yDim_flat; ++i)
        {
            parms += " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
        }

        kernel = handle.AddKernel(kernel_name,
                                  network_config,
                                  program_name,
                                  kernel_name,
                                  {wld, 1, 1},
                                  {wgd, 1, 1},
                                  parms);
    }

    switch(yDim_flat)
    {
    case 1: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]));
        });

        break;
    }
    case 2: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]));
        });

        break;
    }
    case 3: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]));
        });

        break;
    }
    case 4: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]));
        });

        break;
    }
    case 5: {
        visit_float(dataType, [&](auto as_float) {
            kernel(y,
                   *as_float(alpha),
                   offset,
                   static_cast<int>(yDesc_flat.GetStrides()[0]),
                   static_cast<int>(yDesc_flat.GetStrides()[1]),
                   static_cast<int>(yDesc_flat.GetStrides()[2]),
                   static_cast<int>(yDesc_flat.GetStrides()[3]),
                   static_cast<int>(yDesc_flat.GetStrides()[4]),
                   static_cast<int>(yDesc_flat.GetLengths()[0]),
                   static_cast<int>(yDesc_flat.GetLengths()[1]),
                   static_cast<int>(yDesc_flat.GetLengths()[2]),
                   static_cast<int>(yDesc_flat.GetLengths()[3]),
                   static_cast<int>(yDesc_flat.GetLengths()[4]));
        });

        break;
    }
    default: assert(false);
    }
}

void CopyTensor(const Handle& handle,
                const TensorDescriptor& srcDesc,
                ConstData_t src,
                const TensorDescriptor& dstDesc,
                Data_t dst,
                int srcOffset,
                int dstOffset,
                bool forseAsync)
{
    if(src == nullptr || dst == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(srcDesc.GetType() != dstDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
    }

    if(srcDesc.GetLengths() != dstDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    auto flat_descriptors = GetConsistentFlattenedTensorDescriptors(srcDesc, dstDesc);
    const TensorDescriptor& srcDesc_flat = std::get<0>(flat_descriptors);
    const TensorDescriptor& dstDesc_flat = std::get<1>(flat_descriptors);

#ifndef NDEBUG
    if(srcDesc.GetNumDims() != srcDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("src real descriptor: " << srcDesc);
        MIOPEN_LOG_I2("src flat descriptor: " << srcDesc_flat);
        MIOPEN_LOG_I2("dst real descriptor: " << dstDesc);
        MIOPEN_LOG_I2("dst flat descriptor: " << dstDesc_flat);
    }
#endif

    std::size_t srcDim_flat = srcDesc_flat.GetNumDims();

    if(srcDim_flat < 1 || srcDim_flat > 5)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension sizes unsupported.");
    }

    if(forseAsync || srcOffset > 0 || dstOffset > 0 ||
       (!(srcDesc_flat.IsPacked() && dstDesc_flat.IsPacked())))
    {
        std::string kernel_name = "SubTensorOpWithSubTensor" + std::to_string(srcDim_flat) + "d";

        const std::vector<std::size_t>& lens = srcDesc_flat.GetLengths();

        std::string network_config = "copy " + std::to_string(srcDesc_flat.GetType());
        for(auto& len : lens)
        {
            network_config += " " + std::to_string(len);
        }

        auto&& kernels = handle.GetKernels(kernel_name, network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenSubTensorOpWithSubTensorKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms = "-DSUBTENSOR_OP_WITH_SUBTENSOR=SUBTENSOR_OP_WITH_SUBTENSOR_COPY" +
                                GetDataTypeKernelParams(srcDesc_flat.GetType());
            for(std::size_t i = 0; i < srcDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        switch(srcDim_flat)
        {
        case 1: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]));

            break;
        }
        case 2: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]));

            break;
        }
        case 3: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]));

            break;
        }
        case 4: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]));

            break;
        }
        case 5: {
            kernel(src,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetStrides()[4]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[4]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]),
                   static_cast<int>(dstDesc_flat.GetStrides()[4]));

            break;
        }
        default: assert(false);
        }
    }
    else
    {
        handle.Copy(src, dst, srcDesc_flat.GetElementSize() * GetTypeSize(srcDesc_flat.GetType()));
    }
}

std::string GetCastTensorBuildOptionFromType(const std::string& buildOption, miopenDataType_t type)
{
    std::string option(buildOption);
    switch(type)
    {
    case miopenInt8: return option += "0";
    case miopenInt32: return option += "1";
    case miopenHalf: return option += "2";
    case miopenFloat: return option += "3";
    case miopenBFloat16: return option += "4";
    case miopenFloat8_fnuz:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenFloat8_fnuz data type not supported in cast tensor.");
    case miopenBFloat8_fnuz:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenBFloat8_fnuz data type not supported in cast tensor.");
    case miopenDouble:
        // TODO
        MIOPEN_THROW(miopenStatusBadParm, "miopenDouble data type not supported in cast tensor.");
    case miopenInt64:
        MIOPEN_THROW(miopenStatusBadParm, "miopenInt64 data type not supported in cast tensor.");
    default: MIOPEN_THROW(miopenStatusBadParm, "Invalid data type in cast tensor desc.");
    }
}

void CastTensor(const Handle& handle,
                const void* alpha,
                const bool clamping,
                const TensorDescriptor& srcDesc,
                ConstData_t src,
                const TensorDescriptor& dstDesc,
                Data_t dst,
                int srcOffset,
                int dstOffset)
{
    if(src == nullptr || dst == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    if(srcDesc.GetLengths() != dstDesc.GetLengths())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
    }

    auto flat_descriptors = GetConsistentFlattenedTensorDescriptors(srcDesc, dstDesc);
    const TensorDescriptor& srcDesc_flat = std::get<0>(flat_descriptors);
    const TensorDescriptor& dstDesc_flat = std::get<1>(flat_descriptors);

#ifndef NDEBUG
    if(srcDesc.GetNumDims() != srcDesc_flat.GetNumDims())
    {
        MIOPEN_LOG_I2("src real descriptor: " << srcDesc);
        MIOPEN_LOG_I2("src flat descriptor: " << srcDesc_flat);
        MIOPEN_LOG_I2("dst real descriptor: " << dstDesc);
        MIOPEN_LOG_I2("dst flat descriptor: " << dstDesc_flat);
    }
#endif

    std::size_t srcDim_flat = srcDesc_flat.GetNumDims();

    if(srcDim_flat < 1 || srcDim_flat > 5)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension sizes unsupported.");
    }

    auto miopen_alpha = *(static_cast<const float*>(alpha));

    if(srcDesc.GetType() == dstDesc.GetType() && srcOffset == 0 && dstOffset == 0 &&
       srcDesc_flat.IsPacked() && dstDesc_flat.IsPacked() && float_equal(miopen_alpha, 1.0))
    {
        handle.Copy(src, dst, srcDesc_flat.GetElementSize() * GetTypeSize(srcDesc_flat.GetType()));
    }
    else
    {
        std::string kernel_name = "SubTensorOpWithCastTensor" + std::to_string(srcDim_flat) + "d";

        const std::vector<std::size_t>& lens = srcDesc_flat.GetLengths();

        // TODO: make proper network config
        std::string network_config = "cast " + std::to_string(srcDesc_flat.GetType()) +
                                     std::to_string(dstDesc_flat.GetType());
        for(auto& len : lens)
        {
            network_config += " " + std::to_string(len);
        }

        auto&& kernels = handle.GetKernels(kernel_name, network_config);
        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenSubTensorOpWithCastTensorKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms =
                GetCastTensorBuildOptionFromType(" -DMIOPEN_SRC_TYPE=", srcDesc_flat.GetType()) +
                GetCastTensorBuildOptionFromType(" -DMIOPEN_DST_TYPE=", dstDesc_flat.GetType());

            for(std::size_t i = 0; i < srcDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            if(dstDesc_flat.GetType() == miopenBFloat16)
            {
                parms += " -DMIOPEN_USE_RNE_BFLOAT16=1";
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        const int clamping_arg = clamping ? 1 : 0;
        switch(srcDim_flat)
        {
        case 1: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]));

            break;
        }
        case 2: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]));

            break;
        }
        case 3: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]));

            break;
        }
        case 4: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]));

            break;
        }
        case 5: {
            kernel(src,
                   miopen_alpha,
                   clamping_arg,
                   srcOffset,
                   static_cast<int>(srcDesc_flat.GetStrides()[0]),
                   static_cast<int>(srcDesc_flat.GetStrides()[1]),
                   static_cast<int>(srcDesc_flat.GetStrides()[2]),
                   static_cast<int>(srcDesc_flat.GetStrides()[3]),
                   static_cast<int>(srcDesc_flat.GetStrides()[4]),
                   static_cast<int>(srcDesc_flat.GetLengths()[0]),
                   static_cast<int>(srcDesc_flat.GetLengths()[1]),
                   static_cast<int>(srcDesc_flat.GetLengths()[2]),
                   static_cast<int>(srcDesc_flat.GetLengths()[3]),
                   static_cast<int>(srcDesc_flat.GetLengths()[4]),
                   dst,
                   dstOffset,
                   static_cast<int>(dstDesc_flat.GetStrides()[0]),
                   static_cast<int>(dstDesc_flat.GetStrides()[1]),
                   static_cast<int>(dstDesc_flat.GetStrides()[2]),
                   static_cast<int>(dstDesc_flat.GetStrides()[3]),
                   static_cast<int>(dstDesc_flat.GetStrides()[4]));

            break;
        }
        default: assert(false);
        }
    }
}

void TransformTensor(const Handle& handle,
                     const void* alpha,
                     const TensorDescriptor& xDesc,
                     ConstData_t x,
                     const void* beta,
                     const TensorDescriptor& yDesc,
                     Data_t y,
                     size_t Xoffset,
                     size_t Yoffset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(alpha == nullptr || beta == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    auto x_len = xDesc.GetLengths();
    auto y_len = yDesc.GetLengths();

    if(x_len.size() != y_len.size())
    {
        MIOPEN_THROW("Tensor dimension must be the same");
    }

    if(x_len[0] != y_len[0])
    {
        MIOPEN_THROW("Tensor x and y batch sizes do not match");
    }

    const auto is_alpha_one = float_equal(*(static_cast<const float*>(alpha)), 1);
    const auto is_beta_zero = float_equal(*(static_cast<const float*>(beta)), 0);

    if(xDesc.GetType() == miopenInt8 && yDesc.GetType() == miopenInt8 && x_len.size() >= 3)
    {
        if(x_len[1] <= y_len[1])
        {
            if(x_len[1] <= (y_len[1] - 4) || y_len[1] % 4 != 0)
            {
                MIOPEN_THROW("Invalid y channel size");
            }

            int8_t zero = 0;
            SetTensor(handle, yDesc, y, &zero);
        }
        else if(x_len[1] % 4 != 0)
        {
            MIOPEN_THROW("Invalid x channel size");
        }

        size_t batch_n = x_len[0];

        x_len[0] = 1;
        y_len[0] = 1;

        miopen::TensorDescriptor x_batch_desc, y_batch_desc;
        x_batch_desc = miopen::TensorDescriptor(miopenInt8, x_len);
        y_batch_desc = miopen::TensorDescriptor(miopenInt8, y_len);

        size_t x_batch_sz = x_batch_desc.GetElementSize();
        size_t y_batch_sz = y_batch_desc.GetElementSize();

        for(size_t i = 0; i < batch_n; i++)
        {
            size_t x_offset = i * x_batch_sz;
            size_t y_offset = i * y_batch_sz;

            if(is_alpha_one && is_beta_zero)
            {
                CopyTensor(handle,
                           ((x_len[1] <= y_len[1]) ? x_batch_desc : y_batch_desc),
                           x,
                           ((x_len[1] <= y_len[1]) ? x_batch_desc : y_batch_desc),
                           y,
                           x_offset,
                           y_offset);
            }
            else
            {
                MIOPEN_THROW(miopenStatusNotImplemented,
                             "y=alpha*x+beta*y is not supported for int8 yet");
            }
        }
    }
    else
    {
        auto x_y_len          = boost::combine(x_len, y_len);
        bool same_spatial_len = std::all_of(x_y_len.begin(), x_y_len.end(), [](auto v) {
            return boost::get<0>(v) == boost::get<1>(v);
        });

        if(!same_spatial_len)
        {
            MIOPEN_THROW("Tensor x and y spatial sizes do not match");
        }

        auto flat_descriptors              = GetConsistentFlattenedTensorDescriptors(xDesc, yDesc);
        const TensorDescriptor& xDesc_flat = std::get<0>(flat_descriptors);
        const TensorDescriptor& yDesc_flat = std::get<1>(flat_descriptors);

        if(xDesc.GetNumDims() != xDesc_flat.GetNumDims())
        {
            MIOPEN_LOG_I2("x real descriptor: " << xDesc);
            MIOPEN_LOG_I2("x flat descriptor: " << xDesc_flat);
        }

        if(yDesc.GetNumDims() != yDesc_flat.GetNumDims())
        {
            MIOPEN_LOG_I2("y real descriptor: " << yDesc);
            MIOPEN_LOG_I2("y flat descriptor: " << yDesc_flat);
        }

        const std::size_t yDim_flat = yDesc_flat.GetNumDims();

        assert(yDim_flat > 0 && yDim_flat <= 5);

        const miopenDataType_t dataTypex = xDesc_flat.GetType();
        const miopenDataType_t dataTypey = yDesc_flat.GetType();

        if(!(dataTypex == miopenHalf        //
             || dataTypex == miopenFloat    //
             || dataTypex == miopenInt32    //
             || dataTypex == miopenBFloat16 //
             || dataTypex == miopenDouble))
        {
            MIOPEN_THROW("Tensor x is a unsupported data type");
        }

        if(!(dataTypey == miopenHalf        //
             || dataTypey == miopenFloat    //
             || dataTypey == miopenInt32    //
             || dataTypey == miopenBFloat16 //
             || dataTypey == miopenDouble))
        {
            MIOPEN_THROW("Tensor y is a unsupported data type");
        }

        if(dataTypex != dataTypey)
        {
            MIOPEN_THROW("Tensor x and y have different data types");
        }

        std::string kernel_name = "SubTensorOpWithTransform" + std::to_string(yDim_flat) + "d";

        const std::vector<std::size_t>& lens = yDesc_flat.GetLengths();

        std::string network_config = "transform " + std::to_string(yDesc_flat.GetType());
        for(auto& len : lens)
        {
            network_config += "x" + std::to_string(len);
        }

        if(is_beta_zero)
            network_config += "xBETA_IS_ZERO";
        if(is_alpha_one)
            network_config += "xALPHA_IS_ONE";

        auto&& kernels = handle.GetKernels(kernel_name, network_config);

        KernelInvoke kernel;

        if(!kernels.empty())
        {
            kernel = kernels.front();
        }
        else
        {
            std::string program_name = "MIOpenSubTensorOpWithTransformKernel.cl";

            std::vector<std::size_t> worker_sizes = get_worker_sizes(lens);

            std::size_t wgd = std::accumulate(worker_sizes.begin(),
                                              worker_sizes.end(),
                                              std::size_t{1},
                                              std::multiplies<std::size_t>());

            std::size_t wld = 256 < wgd ? 256 : wgd;

            std::string parms =
                GetDataTypeKernelParams(dataTypey)                                           //
                + " -DMIOPEN_BETA_IS_ZERO=" + std::to_string(static_cast<int>(is_beta_zero)) //
                + " -DMIOPEN_ALPHA_IS_ONE=" + std::to_string(static_cast<int>(is_alpha_one));

            for(int i = 0; i < yDim_flat; ++i)
            {
                parms +=
                    " -DWORK_LENGTH_" + std::to_string(i) + "=" + std::to_string(worker_sizes[i]);
            }

            kernel = handle.AddKernel(kernel_name,
                                      network_config,
                                      program_name,
                                      kernel_name,
                                      {wld, 1, 1},
                                      {wgd, 1, 1},
                                      parms);
        }

        switch(yDim_flat)
        {
        case 1: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]));
            });

            break;
        }
        case 2: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]));
            });

            break;
        }
        case 3: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]));
            });

            break;
        }
        case 4: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[3]));
            });

            break;
        }
        case 5: {
            visit_float(dataTypey, [&](auto as_float) {
                kernel(x,
                       *as_float(alpha),
                       y,
                       *as_float(beta),
                       static_cast<unsigned>(Xoffset),
                       static_cast<unsigned>(Yoffset),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(xDesc_flat.GetStrides()[4]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[0]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[1]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[2]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[3]),
                       static_cast<unsigned>(yDesc_flat.GetStrides()[4]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[0]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[1]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[2]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[3]),
                       static_cast<unsigned>(yDesc_flat.GetLengths()[4]));
            });

            break;
        }
        default: assert(false);
        }
    }
}

void OpTensor(const Handle& handle,
              miopenTensorOp_t tensorOp,
              const void* alpha0,
              const TensorDescriptor& aTensorDesc,
              ConstData_t ATensor,
              const void* alpha1,
              const TensorDescriptor& bTensorDesc,
              ConstData_t BTensor,
              const void* beta,
              const TensorDescriptor& cTensorDesc,
              Data_t CTensor,
              const size_t Aoffset,
              const size_t Boffset,
              const size_t Coffset,
              bool nonStandardSquash)
{
    if(ATensor == nullptr || BTensor == nullptr || CTensor == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    if(alpha0 == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Alpha0 value is nullptr");
    }

    if(alpha1 == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Alpha1 value is nullptr");
    }

    const auto problem = tensorOp::ProblemDescription{
        tensorOp, beta, aTensorDesc, bTensorDesc, cTensorDesc, nonStandardSquash};

    const auto invoke_params = tensorOp::InvokeParams{
        alpha0, ATensor, alpha1, BTensor, beta, CTensor, Aoffset, Boffset, Coffset};

    const auto algo    = AlgorithmName{"TensorOpSolver"};
    const auto solvers = solver::SolverContainer<solver::tensorOp::OpTensorFwdBias>{} +
                         solver::SolverContainer<solver::tensorOp::Op4dTensorLite>{} +
                         solver::SolverContainer<solver::tensorOp::OpTensorLeadingOnes>{} +
                         solver::SolverContainer<solver::tensorOp::Op2dTensorLite>{} +
                         solver::SolverContainer<solver::tensorOp::Op2dTensorSquash>{} +
                         solver::SolverContainer<solver::tensorOp::Op5dTensorGeneric>{} +
                         solver::SolverContainer<solver::tensorOp::Op4dTensorGeneric>{} +
                         solver::SolverContainer<solver::tensorOp::Op3dTensorGeneric>{} +
                         solver::SolverContainer<solver::tensorOp::Op2dTensorGeneric>{} +
                         solver::SolverContainer<solver::tensorOp::Op1dTensorGeneric>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
}

} // namespace miopen

int miopenGetTensorIndex(miopenTensorDescriptor_t tensorDesc, std::initializer_list<int> indices)
{
    return miopen::deref(tensorDesc).GetIndex(indices);
}
