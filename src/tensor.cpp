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

bool IsLayoutSupported(miopenTensorLayout_t layout)
{
    switch(layout)
    {
    case miopenTensorNCHW:
    case miopenTensorNHWC:
    case miopenTensorCHWN:
    case miopenTensorNCHWc4:
    case miopenTensorNCHWc8:
    case miopenTensorCHWNc4:
    case miopenTensorCHWNc8:
    case miopenTensorNCDHW:
    case miopenTensorNDHWC: return true;
    }
    return false;
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
    : TensorDescriptor(t, GetDefaultLayout(), lens_in)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   const std::initializer_list<std::size_t>& lens_in)
    : TensorDescriptor(t, std::vector<std::size_t>(lens_in))
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, const std::vector<std::size_t>& lens_in)
    : TensorDescriptor(t, GetDefaultLayout(), lens_in)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t, std::vector<std::size_t>&& lens_in)
    : TensorDescriptor(t, GetDefaultLayout(), std::move(lens_in))
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
    : TensorDescriptor(t, GetDefaultLayout(), lens_in, strides_in)
{
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   std::vector<std::size_t>&& lens_in,
                                   std::vector<std::size_t>&& strides_in)
    : TensorDescriptor(t, GetDefaultLayout(), std::move(lens_in), std::move(strides_in))
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
                                   miopenTensorLayout_t layout_in,
                                   const std::vector<std::size_t>& lens_in,
                                   const std::vector<std::size_t>& strides_in,
                                   bool use_strides)
    : lens(lens_in),
      strides(use_strides ? strides_in : std::vector<std::size_t>()),
      type(t),
      tensorLayout(layout_in)
{
    CheckArgsAndInit(use_strides);
}

TensorDescriptor::TensorDescriptor(miopenDataType_t t,
                                   miopenTensorLayout_t layout_in,
                                   std::vector<std::size_t>&& lens_in,
                                   std::vector<std::size_t>&& strides_in,
                                   bool use_strides)
    : lens(std::move(lens_in)),
      strides(use_strides ? std::move(strides_in) : std::vector<std::size_t>()),
      type(t),
      tensorLayout(layout_in)
{
    CheckArgsAndInit(use_strides);
}

void TensorDescriptor::CheckArgsAndInit(bool use_strides)
{
    if(!IsDataTypeSupported(type))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported data type");

    if(!IsLayoutSupported(tensorLayout))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported layout");

    if(lens.empty())
        MIOPEN_THROW(miopenStatusBadParm, "Number of dimensions must be > 1");

    if(!CheckLengths(lens, static_cast<std::size_t>(std::numeric_limits<int64_t>::max())))
        MIOPEN_THROW(miopenStatusBadParm, "Lengths must be > 0 and <= INT64_MAX");

    this->CalculateVectorLength();

    if(use_strides)
    {
        if(lens.size() != strides.size())
            MIOPEN_THROW(miopenStatusBadParm, "Lengths and strides dimensions must be equal");

        if(!CheckLengths(strides, static_cast<std::size_t>(std::numeric_limits<int64_t>::max())))
            MIOPEN_THROW(miopenStatusBadParm, "Strides must be > 0 and <= INT64_MAX");

        packed = (this->GetElementSize() == this->GetElementSpace());
    }
    else
    {
        packed = true;
        // Since strides is not passed it is computed based on tensorLayout.
        SetStrideNd(GetLayout_str());
    }
}

void TensorDescriptor::SetStrideNd(const std::string& layout)
{
    std::string default_layout = miopen::tensor_layout_get_default(layout.size());
    if(layout == default_layout)
    {
        CalculateStrides();
    }
    else if(layout.find('c') != std::string::npos)
    {
        LensReorder(layout);
        CalculateStrides();
    }
    else
    {
        miopen::tensor_layout_to_strides(lens, default_layout, layout, strides);
    }
}

void TensorDescriptor::LensReorder(const std::string& layout)
{
    if(layout == "NCHWc")
    {
        // Do nothing, MIOpen implicit logic that lens are in NCHW order.
    }
    else if(layout == "CHWNc")
    {
        ReorderVector(lens, {1, 2, 3, 0});
    }
    else
    {
        MIOPEN_THROW("We only support NCHWc4, NCHWc8, CHWNc4, CHWNc8 vectorized tensor layout.");
    }
}

TensorDescriptor TensorDescriptor::MakeDescriptor(miopenDataType_t t, const int* plens, int size)
{
    return MakeDescriptor(t, GetDefaultLayout(), plens, size);
}

TensorDescriptor
TensorDescriptor::MakeDescriptor(miopenDataType_t t, const std::size_t* plens, int size)
{
    return MakeDescriptor(t, GetDefaultLayout(), plens, size);
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

void TensorDescriptor::CalculateStrides()
{
    if(lens.empty())
        MIOPEN_THROW(miopenStatusInternalError, "lens must be non-empty");
    strides.clear();
    strides.resize(lens.size(), 0);
    if(tensorLayout == miopenTensorNCHWc4 || tensorLayout == miopenTensorNCHWc8)
    {
        lens[1] /= vector_length;
    }
    else if(tensorLayout == miopenTensorCHWNc4 || tensorLayout == miopenTensorCHWNc8)
    {
        lens[0] /= vector_length;
    }

    strides.back() = vector_length;
    std::partial_sum(
        lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
    for(int i = 0; i < strides.size() - 1; i++)
        strides[i] *= vector_length;
}

void TensorDescriptor::CalculateVectorLength()
{
    vector_length =
        ((tensorLayout == miopenTensorCHWNc8 || tensorLayout == miopenTensorNCHWc8)
             ? 8
             : ((tensorLayout == miopenTensorCHWNc4 || tensorLayout == miopenTensorNCHWc4) ? 4
                                                                                           : 1));
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

miopenTensorLayout_t TensorDescriptor::GetLayout_t() const { return this->tensorLayout; }

std::string TensorDescriptor::GetLayoutStr(miopenTensorLayout_t tensorLayout)
{
    switch(tensorLayout)
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
    default: MIOPEN_THROW(miopenStatusInternalError, "Unknown tensor layout");
    }
}

std::string TensorDescriptor::GetLayout_str() const { return GetLayoutStr(this->tensorLayout); }

std::size_t TensorDescriptor::GetVectorLength() const { return this->vector_length; }

std::size_t TensorDescriptor::GetIndex(std::initializer_list<int> l) const
{
    // l is in NCHW order (MIOpen implicit logic)
    if(this->GetLayout_str() == "CHWNc")
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
    std::vector<std::size_t> maxIndices(lens.size());
    std::transform(lens.begin(),
                   lens.end(),
                   std::vector<std::size_t>(lens.size(), 1).begin(),
                   maxIndices.begin(),
                   std::minus<std::size_t>());
    return std::inner_product(
               maxIndices.begin(), maxIndices.end(), strides.begin(), std::size_t{0}) +
           vector_length;
}

bool TensorDescriptor::IsPossibleLayout(const std::string& labels, const std::string& layout) const
{
    std::vector<size_t> derived_strides;
    tensor_layout_to_strides(lens, labels, layout, derived_strides);
    return derived_strides == strides;
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
    if(std::any_of(lens.cbegin(), lens.cend(), [](std::size_t x) {
           return x > std::numeric_limits<int>::max();
       }))
    {
        return false;
    }
    return true;
}

bool TensorDescriptor::AllDimsFitIntoInt() const
{
    if(!AllLengthsFitIntoInt())
        return false;
    if(std::any_of(strides.cbegin(), strides.cend(), [](std::size_t x) {
           return x > std::numeric_limits<int>::max();
       }))
    {
        return false;
    }
    return true;
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
