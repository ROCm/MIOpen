#include <miopen/driver_arguments.hpp>

namespace miopen {
namespace debug {

void ConvDataType(std::stringstream& ss, const miopenTensorDescriptor_t& desc)
{
    if(miopen::deref(desc).GetType() == miopenHalf)
    {
        ss << "convfp16";
    }
    else if(miopen::deref(desc).GetType() == miopenBFloat16)
    {
        ss << "convbfp16";
    }
    else if(miopen::deref(desc).GetType() == miopenInt8 ||
            miopen::deref(desc).GetType() == miopenInt8x4)
    {
        ss << "convint8";
    }
    else
    {
        ss << "conv";
    }
}

std::string ConvArgsForMIOpenDriver(const miopenTensorDescriptor_t& xDesc,
                                    const miopenTensorDescriptor_t& wDesc,
                                    const miopenConvolutionDescriptor_t& convDesc,
                                    const miopenTensorDescriptor_t& yDesc,
                                    const ConvDirection& conv_dir,
                                    bool is_immediate,
                                    bool print_all_args_info)
{
    std::stringstream ss;
    if(print_all_args_info)
        ConvDataType(ss, xDesc);
    if(miopen::deref(convDesc).GetSpatialDimension() == 2)
    {
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
            << " -c " << miopen::deref(xDesc).GetLengths()[1]
            << " -H " << miopen::deref(xDesc).GetLengths()[2]
            << " -W " << miopen::deref(xDesc).GetLengths()[3]
            << " -k " << miopen::deref(wDesc).GetLengths()[0]
            << " -y " << miopen::deref(wDesc).GetLengths()[2]
            << " -x " << miopen::deref(wDesc).GetLengths()[3]
            << " -p " << miopen::deref(convDesc).GetConvPads()[0]
            << " -q " << miopen::deref(convDesc).GetConvPads()[1]
            << " -u " << miopen::deref(convDesc).GetConvStrides()[0]
            << " -v " << miopen::deref(convDesc).GetConvStrides()[1]
            << " -l " << miopen::deref(convDesc).GetConvDilations()[0]
            << " -j " << miopen::deref(convDesc).GetConvDilations()[1]; // clang-format on
        std::string x_layout = miopen::deref(xDesc).GetLayout("NCHW");
        std::string w_layout = miopen::deref(wDesc).GetLayout("NCHW");
        std::string y_layout = miopen::deref(yDesc).GetLayout("NCHW");
        if(x_layout != "NCHW")
        {
            ss << " --in_layout " << x_layout;
        }
        if(w_layout != "NCHW")
        {
            ss << " --fil_layout " << w_layout;
        }
        if(y_layout != "NCHW")
        {
            ss << " --out_layout " << y_layout;
        }
    }
    else if(miopen::deref(convDesc).GetSpatialDimension() == 3)
    {
        ss << " -n " << miopen::deref(xDesc).GetLengths()[0] // clang-format off
            << " -c " << miopen::deref(xDesc).GetLengths()[1]
            << " --in_d " << miopen::deref(xDesc).GetLengths()[2]
            << " -H " << miopen::deref(xDesc).GetLengths()[3]
            << " -W " << miopen::deref(xDesc).GetLengths()[4]
            << " -k " << miopen::deref(wDesc).GetLengths()[0]
            << " --fil_d " << miopen::deref(wDesc).GetLengths()[2]
            << " -y " << miopen::deref(wDesc).GetLengths()[3]
            << " -x " << miopen::deref(wDesc).GetLengths()[4]
            << " --pad_d " << miopen::deref(convDesc).GetConvPads()[0]
            << " -p " << miopen::deref(convDesc).GetConvPads()[1]
            << " -q " << miopen::deref(convDesc).GetConvPads()[2]
            << " --conv_stride_d " << miopen::deref(convDesc).GetConvStrides()[0]
            << " -u " << miopen::deref(convDesc).GetConvStrides()[1]
            << " -v " << miopen::deref(convDesc).GetConvStrides()[2]
            << " --dilation_d " << miopen::deref(convDesc).GetConvDilations()[0]
            << " -l " << miopen::deref(convDesc).GetConvDilations()[1]
            << " -j " << miopen::deref(convDesc).GetConvDilations()[2]
            << " --spatial_dim 3"; // clang-format on
        std::string x_layout = miopen::deref(xDesc).GetLayout("NCDHW");
        std::string w_layout = miopen::deref(wDesc).GetLayout("NCDHW");
        std::string y_layout = miopen::deref(yDesc).GetLayout("NCDHW");
        if(x_layout != "NCDHW")
        {
            ss << " --in_layout " << x_layout;
        }
        if(w_layout != "NCDHW")
        {
            ss << " --fil_layout " << w_layout;
        }
        if(y_layout != "NCDHW")
        {
            ss << " --out_layout " << y_layout;
        }
    }
    if(print_all_args_info)
        ss << " -m " << (miopen::deref(convDesc).mode == 1 ? "trans" : "conv"); // clang-format off
    ss << " -g " << miopen::deref(convDesc).group_count;
    if(print_all_args_info)
        ss << " -F " << std::to_string(static_cast<int>(conv_dir)) << " -t 1"; // clang-format on
    if(miopen::deref(xDesc).GetType() == miopenInt8x4)
        ss << " -Z 1";
    if(is_immediate)
        ss << " -S 0";

    return ss.str();
}

} // namespace debug
} // namespace miopen
