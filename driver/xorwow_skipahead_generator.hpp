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
#pragma once

#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <iomanip>
#include <miopen/dropout.hpp>

#define XORWOW_DIM 5
#define XORWOW_BITS 32
#define XORWOW_PRECALC_MATRICES_SZ (XORWOW_BITS * XORWOW_DIM * XORWOW_DIM)
#define XORWOW_PRECALC_MATRICES_NUM 32
#define XORWOW_JUMP_LOG2 2
#define XORWOW_JUMP_LOG2_MASK ((1 << XORWOW_JUMP_LOG2) - 1)
#define XORWOW_SEQUENCE_JUMP_LOG2 67

unsigned int xorwow_next(prngStates* cur_state)
{

    const unsigned int t = cur_state->x ^ (cur_state->x >> 2);
    cur_state->x         = cur_state->y;
    cur_state->y         = cur_state->z;
    cur_state->z         = cur_state->w;
    cur_state->w         = cur_state->v;
    cur_state->v         = (cur_state->v ^ (cur_state->v << 4)) ^ (t ^ (t << 1));

    cur_state->d += 362437;

    return cur_state->d + cur_state->v;
}

// multiply vector by 32-bit-unit matrix, store result in vector
inline void mat_vec(const unsigned int* matrix, unsigned int* vector)
{
    unsigned int result[XORWOW_DIM] = {0};
    for(unsigned int i = 0; i < XORWOW_DIM; i++)
    {
        for(unsigned int j = 0; j < XORWOW_BITS; j++)
        {
            if(bool(vector[i] & (1U << j)))
            {
                std::transform(result,
                               result + XORWOW_DIM,
                               matrix + (XORWOW_DIM * (i * XORWOW_BITS + j)),
                               result,
                               std::bit_xor<unsigned int>{});
            }
        }
    }
    std::copy(std::begin(result), std::end(result), vector);
}

// multiply matrixA by matrixB, store result in matrixA, matrixA and matrixB address cannot be the
// same
inline void mat_mat(unsigned int* matrixA, const unsigned int* matrixB)
{
    for(int i = 0; i < XORWOW_DIM * XORWOW_BITS; i++)
    {
        mat_vec(matrixB, matrixA + i * XORWOW_DIM);
    }
}

// generate identity matrix
inline void mat_identity(unsigned int* matrix)
{
    for(unsigned int i = 0; i < XORWOW_DIM; i++)
    {
        for(unsigned int j = 0; j < XORWOW_BITS; j++)
        {
            for(unsigned int k = 0; k < XORWOW_DIM; k++)
            {
                matrix[(i * XORWOW_BITS + j) * XORWOW_DIM + k] = i == k ? (1 << j) : 0;
            }
        }
    }
}

// compute matrix^power, store result in matrixP
inline void mat_pow(unsigned int* matrixP, const unsigned int* matrix, unsigned long long power)
{
    mat_identity(matrixP);

    unsigned int matrixA[XORWOW_PRECALC_MATRICES_SZ];
    unsigned int matrixB[XORWOW_PRECALC_MATRICES_SZ];
    std::copy(matrix, matrix + XORWOW_PRECALC_MATRICES_SZ, std::begin(matrixA));
    while(bool(power))
    {
        if(bool(power & 1))
        {
            mat_mat(matrixP, matrixA);
        }

        std::copy(std::begin(matrixA), std::end(matrixA), std::begin(matrixB));
        mat_mat(matrixA, matrixB);
        power >>= 1;
    }
}

// Generate matrix one-step advanced
void skipahead_one_step(unsigned int* matrix)
{
    xorwowStates init_state;

    for(unsigned int i = 0; i < XORWOW_DIM; i++)
    {
        for(unsigned int j = 0; j < XORWOW_BITS; j++)
        {
            unsigned int* p = &(init_state.x);
            for(unsigned int k = 0; k < XORWOW_DIM; k++)
            {
                *(p + k) = i == k ? (1 << j) : 0;
            }
            init_state.d = 0;

            xorwow_next(&init_state);

            for(int k = 0; k < XORWOW_DIM; k++)
            {
                matrix[(i * XORWOW_BITS + j) * XORWOW_DIM + k] = *(p + k);
            }
        }
    }
}

// Generate (2^67)-step-ahead matrices
void generate_skipahead_matrices(unsigned int* matrix, bool is_skip_seq)
{
    unsigned int matrixA[XORWOW_PRECALC_MATRICES_SZ];
    unsigned int matrixB[XORWOW_PRECALC_MATRICES_SZ];
    skipahead_one_step(matrixA);

    // skipahead sequence
    if(is_skip_seq)
    {
        // split A^(2^67)
        // A^(2^33)
        mat_pow(matrixB, matrixA, 1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 / 2));
        // (A^(2^33))^(2^34)
        mat_pow(
            matrixA, matrixB, 1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 - XORWOW_SEQUENCE_JUMP_LOG2 / 2));
    }

    std::copy(std::begin(matrixA), std::end(matrixA), matrix);
    for(int k = 1; k < XORWOW_PRECALC_MATRICES_NUM; k++)
    {
        std::copy(std::begin(matrixA), std::end(matrixA), std::begin(matrixB));
        mat_pow(matrixA, matrixB, 1ULL << XORWOW_JUMP_LOG2);
        std::copy(std::begin(matrixA), std::end(matrixA), &matrix[k * XORWOW_PRECALC_MATRICES_SZ]);
    }
}

// write macros in file
void write_macro(std::ofstream& os)
{
    os << "#define XORWOW_DIM " << XORWOW_DIM << std::endl;
    os << "#define XORWOW_BITS " << XORWOW_BITS << std::endl;
    os << "#define XORWOW_PRECALC_MATRICES_SZ (XORWOW_BITS * XORWOW_DIM * XORWOW_DIM)" << std::endl;
    os << "#define XORWOW_PRECALC_MATRICES_NUM " << XORWOW_PRECALC_MATRICES_NUM << std::endl;
    os << "#define XORWOW_JUMP_LOG2 2" << std::endl;
    os << "#define XORWOW_JUMP_LOG2_MASK ((1 << XORWOW_JUMP_LOG2) - 1)" << std::endl;
    os << "#define XORWOW_SEQUENCE_JUMP_LOG2 67" << std::endl;
    os << std::endl;
}

// write matrices in file
void write_mat(std::ofstream& os, const std::string name, unsigned int* matrix, bool is_device)
{
    os << "static " << (is_device ? "__constant " : "const ") << "unsigned int " << name
       << "[XORWOW_PRECALC_MATRICES_NUM][XORWOW_PRECALC_MATRICES_SZ] = {" << std::endl;
    for(int k = 0; k < XORWOW_PRECALC_MATRICES_NUM; k++)
    {
        os << "    {";
        for(int j = 0; j < XORWOW_PRECALC_MATRICES_SZ; j++)
        {
            os << matrix[k * XORWOW_PRECALC_MATRICES_SZ + j] << ", ";
        }
        os << "}," << std::endl;
    }
    os << "};" << std::endl;
    os << std::endl;
}

// generate header files with precalculated skip-ahead matrices
void generate_skipahead_file()
{
    static unsigned int skipahead_matrices[XORWOW_PRECALC_MATRICES_NUM][XORWOW_PRECALC_MATRICES_SZ];
    static unsigned int skipahead_matrices_sequence[XORWOW_PRECALC_MATRICES_NUM]
                                                   [XORWOW_PRECALC_MATRICES_SZ];

    generate_skipahead_matrices(&skipahead_matrices[0][0], false);
    generate_skipahead_matrices(&skipahead_matrices_sequence[0][0], true);

    std::ofstream os;
    os.open("../src/include/miopen/precalc_xorwow_skipahead_matrices.hpp");
    write_macro(os);
    write_mat(os,
              "precalc_xorwow_skipahead_matrices",
              static_cast<unsigned int*>(&skipahead_matrices[0][0]),
              false);
    os.close();
    os.clear();

    os.open("../src/kernels/precalc_xorwow_skipahead_matrices_kernel.h");
    write_macro(os);
    write_mat(os,
              "precalc_xorwow_skipahead_matrices",
              static_cast<unsigned int*>(&skipahead_matrices[0][0]),
              true);
    os.close();
    os.clear();

    os.open("../src/include/miopen/precalc_xorwow_skipahead_sequence_matrices.hpp");
    write_macro(os);
    write_mat(os,
              "precalc_xorwow_skipahead_sequence_matrices",
              static_cast<unsigned int*>(&skipahead_matrices_sequence[0][0]),
              false);
    os.close();
    os.clear();

    os.open("../src/kernels/precalc_xorwow_skipahead_sequence_matrices_kernel.h");
    write_macro(os);
    write_mat(os,
              "precalc_xorwow_skipahead_sequence_matrices",
              static_cast<unsigned int*>(&skipahead_matrices_sequence[0][0]),
              true);
    os.close();
}
