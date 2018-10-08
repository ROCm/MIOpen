/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
s_endpgm
.Lfunc_end0:
   .size sp3AsmConvRxSU, .Lfunc_end0 - sp3AsmConvRxSU

.amd_amdgpu_hsa_metadata
Version:         [ 1, 0 ]
Kernels:
  - Name:            sp3AsmConvRxSU
    SymbolName:      'sp3AsmConvRxSU@kd'
    Language:        OpenCL C
    LanguageVersion: [ 1, 2 ]
    Attrs:
      ReqdWorkGroupSize: [ 512, 1, 1 ]
    Args:
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        'float*'
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Global
        AccQual:         Default
      - TypeName:        'float*'
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Global
        AccQual:         Default
      - TypeName:        'float*'
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Global
        AccQual:         Default
      - TypeName:        'int*'
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       I32
        AddrSpaceQual:   Global
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - TypeName:        int
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
        AccQual:         Default
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetX
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetY
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenGlobalOffsetZ
        ValueType:       I64
      - Size:            8
        Align:           8
        ValueKind:       HiddenNone
        ValueType:       I8
        AddrSpaceQual:   Global
      - Size:            8
        Align:           8
        ValueKind:       HiddenNone
        ValueType:       I8
        AddrSpaceQual:   Global
      - Size:            8
        Align:           8
        ValueKind:       HiddenNone
        ValueType:       I8
        AddrSpaceQual:   Global
    CodeProps:
      KernargSegmentSize: 136
      GroupSegmentFixedSize: 0
      PrivateSegmentFixedSize: 0
      KernargSegmentAlign: 8
      WavefrontSize:   64
      NumVGPRs:       128
      MaxFlatWorkGroupSize: 512
...
.end_amd_amdgpu_hsa_metadata
