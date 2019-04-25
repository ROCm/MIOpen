/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/ctc.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>

namespace miopen {

CTCLossDescriptor::CTCLossDescriptor()
{
    dataType            = miopenFloat;
    apply_softmax_layer = true;
    blank_label_id      = 0;
}

size_t CTCLossDescriptor::GetCTCLossWorkspaceSize(Handle& handle,
                                                  const TensorDescriptor& probsDesc,
                                                  const TensorDescriptor& gradientsDesc,
                                                  const int* labels,
                                                  const int* labelLengths,
                                                  const int* inputLengths,
                                                  miopenCTCLossAlgo_t algo) const
{
    (void)algo;

    if(probsDesc.GetLengths()[0] != gradientsDesc.GetLengths()[0] ||
       probsDesc.GetLengths()[1] != gradientsDesc.GetLengths()[1] ||
       probsDesc.GetLengths()[2] != gradientsDesc.GetLengths()[2])
    {
        MIOPEN_THROW(
            miopenStatusBadParm,
            "The probability tensor's dimensions do not match the gradient tensor's dimensions");
    }

    int class_sz        = probsDesc.GetLengths()[2];
    int batch_size      = probsDesc.GetLengths()[1];
    int max_time_step   = probsDesc.GetLengths()[0];
    int max_label_len   = 0;
    int total_label_len = 0;
    std::vector<int> repeat(batch_size, 0);
    std::vector<int> labels_offset(batch_size, 0);
    size_t wksp_sz_lb  = 0;
    size_t wksp_sz_dat = 0;

    for(int i = 0; i < batch_size; i++)
    {
        if(inputLengths[i] > max_time_step)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Wrong input time step");
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                MIOPEN_THROW(miopenStatusBadParm, "Wrong label id at batch");
            }
            if(j > 0)
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            MIOPEN_THROW(miopenStatusBadParm, "Error: label length exceeds input time step");
        }
    }

    // input length
    wksp_sz_lb += batch_size;

    // label length
    wksp_sz_lb += batch_size;

    // label offset
    wksp_sz_lb += batch_size;

    // label repeat
    wksp_sz_lb += batch_size;

    // labels
    wksp_sz_lb += total_label_len;

    // labels with blanks
    wksp_sz_lb += batch_size * (2 * max_label_len + 1);

    // logsoftmax of probs
    wksp_sz_dat += max_time_step * batch_size * class_sz;

    // alphas
    wksp_sz_dat += max_time_step * batch_size * (2 * max_label_len + 1);

    // beta buffer
    wksp_sz_dat += 2 * batch_size * (2 * max_label_len + 1);

    size_t total_size = wksp_sz_dat * sizeof(float) + wksp_sz_lb * sizeof(int);
    if(total_size > handle.GetMaxMemoryAllocSize())
        MIOPEN_THROW(miopenStatusBadParm, "Error: Workspace size exceeds GPU memory capacity");

    return total_size;
}

std::ostream& operator<<(std::ostream& stream, const CTCLossDescriptor& r)
{
    stream << r.dataType << ", ";
    return stream;
}

} // namespace miopen
