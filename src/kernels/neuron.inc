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


MIOPEN_NEURON_PASTHRU = 0      // x
MIOPEN_NEURON_LOGISTIC = 1     // 1 / (1 + e^-x)	//Sigmoid
MIOPEN_NEURON_TANH = 2         // beta * tanh(alpha * x)
MIOPEN_NEURON_RELU = 3         // max(0, x)
MIOPEN_NEURON_SOFTRELU = 4     // log(1 + e^x)   // bonomial normal log likelihood
MIOPEN_NEURON_ABS = 5          // abs(x)
MIOPEN_NEURON_POWER = 6        // (alpha + beta * x )^gamma
MIOPEN_NEURON_CLIPPED_RELU = 7 // min(alpha, max(0, x))
MIOPEN_NEURON_LEAKY_RELU = 8   // alpha * x | x <= 0; x | x > 0
MIOPEN_NEURON_ELU = 9          // alpha * (e^x - 1) | x <= 0; x | x > 0

EPSILON_float = 0x358637bd
EPSILON_half = 0x00100010

.macro exp_f_float base, sign, vtmp //e^x = 2^(xlog2e)
    .if \sign < 0
        v_mov_b32 v[\vtmp], 0xbfb8aa3b //-log2e
    .else
        v_mov_b32 v[\vtmp], 0x3fb8aa3b //log2e
    .endif
    v_mul_f32 v[\base], v[\base], v[\vtmp]
    v_exp_f32 v[\base], v[\base]
.endm

.macro exp_f_half base, sign, vtmp //e^x = 2^(xlog2e)
    .if \sign < 0
        v_mov_b32 v[\vtmp], 0xbdc5bdc5
    .else
        v_mov_b32 v[\vtmp], 0x3dc53dc5
    .endif
    v_pk_mul_f16 v[\base], v[\base], v[\vtmp]
    v_exp_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
    v_exp_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
.endm

.macro ln_f_float base, vtmp // ln(x) = log2x * 1 / (log2e)
    v_log_f32 v[\base], v[\base]
    v_mov_b32 v[\vtmp], 0x3f317218 // 1/(log2e)
    v_mul_f32 v[\base], v[\base], v[\vtmp]
.endm

.macro ln_f_half base, vtmp // ln(x) = log2x * 1 / (log2e)
    v_log_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
    v_log_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
    v_mov_b32 v[\vtmp], 0x398b398b // 1/(log2e)
    v_pk_mul_f16 v[\base], v[\base], v[\vtmp]
.endm

.macro activ_f_half base, activ_mode, alpha, beta, gamma, vtmp0, vtmp1
    .if \activ_mode == MIOPEN_NEURON_LOGISTIC //1 / (1 + e^-x)
        exp_f_half \base, -1, \vtmp0
        v_mov_b32 v[\vtmp0], 0x3c003c00
        v_pk_add_f16 v[\base], v[\vtmp0], v[\base]
        v_rcp_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
        v_rcp_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
    .elseif \activ_mode == MIOPEN_NEURON_TANH // \beta * tanh(\alpha * x)
        v_pk_mul_f16 v[\base], s[\alpha], v[\base]
        v_mov_b32 v[\vtmp1], 0x40004000
        v_pk_mul_f16 v[\base], v[\vtmp1], v[\base]
        exp_f_half \base, 1, \vtmp0
        v_mov_b32 v[\vtmp0], 0x3c003c00
        v_pk_add_f16 v[\base], v[\vtmp0], v[\base]
        v_rcp_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
        v_rcp_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
        v_mov_b32 v[\vtmp1], 0xc000c000
        v_pk_mul_f16 v[\base], v[\vtmp1], v[\base]
        v_pk_add_f16 v[\base], v[\vtmp0], v[\base]
        v_pk_mul_f16 v[\base], s[\beta], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_RELU //max(0, x)
        v_pk_max_f16 v[\base], v[\base], 0
    .elseif \activ_mode == MIOPEN_NEURON_SOFTRELU //log(1 + e^x)
        exp_f_half \base, 1, \vtmp0
        v_mov_b32 v[\vtmp0], 0x3c003c00
        v_pk_add_f16 v[\base], v[\vtmp0], v[\base]
        ln_f_half \base, \vtmp0
    .elseif \activ_mode == MIOPEN_NEURON_ABS //abs(x)
        v_max_f16 v[\base], v[\base], -v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
        v_max_f16 v[\base], v[\base], -v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
    .elseif \activ_mode == MIOPEN_NEURON_POWER //(\alpha + \beta * x )^\gamma
        v_pk_mul_f16 v[\base], s[\beta], v[\base]
        v_pk_add_f16 v[\base], s[\alpha], v[\base]
        v_mov_b32 v[\vtmp0], v[\base]
        v_log_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
        v_log_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
        v_pk_mul_f16 v[\base], s[\gamma], v[\base]
        v_exp_f16 v[\base], v[\base] dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0
        v_exp_f16 v[\base], v[\base] dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1
        v_mov_b32 v[\vtmp1], EPSILON_half
        v_cmp_lt_f16 vcc, v[\vtmp1], v[\vtmp0] src0_sel:WORD_0 src1_sel:WORD_0
        v_cndmask_b32 v[\base], 0, v[\base], vcc dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
        v_cmp_lt_f16 vcc, v[\vtmp1], v[\vtmp0] src0_sel:WORD_0 src1_sel:WORD_1
        v_cndmask_b32 v[\base], 0, v[\base], vcc dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
    .elseif \activ_mode == MIOPEN_NEURON_CLIPPED_RELU //min(\alpha, max(0, x))
        v_pk_max_f16 v[\base], v[\base], 0
        v_pk_min_f16 v[\base], s[\alpha], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_LEAKY_RELU //\alpha * x | x <= 0; x | x > 0
        v_mov_b32 v[\vtmp1], 0x3c003c00
        v_mov_b32 v[\vtmp0], s[\alpha]
        v_cmp_lt_f16 vcc, 0, v[\base] src0_sel:WORD_0 src1_sel:WORD_0
        v_cndmask_b32 v[\vtmp0], v[\vtmp0], v[\vtmp1], vcc dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
        v_cmp_lt_f16 vcc, 0, v[\base] src0_sel:WORD_0 src1_sel:WORD_1
        v_cndmask_b32 v[\vtmp0], v[\vtmp0], v[\vtmp1], vcc dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
        v_pk_mul_f16 v[\base], v[\base], v[\vtmp0]
    .elseif \activ_mode == MIOPEN_NEURON_ELU //\alpha * (e^x - 1) | x <= 0; x | x > 0
        v_mov_b32 v[\vtmp1], v[\base]
        exp_f_half \base, 1, \vtmp0
        v_mov_b32 v[\vtmp0], 0xbc00bc00
        v_pk_add_f16 v[\base], v[\vtmp0], v[\base]
        v_pk_mul_f16 v[\base], s[\alpha], v[\base]
        v_cmp_lt_f16 vcc, 0, v[\vtmp1] src0_sel:WORD_0 src1_sel:WORD_0
        v_cndmask_b32 v[\base], v[\base], v[\vtmp1], vcc dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0
        v_cmp_lt_f16 vcc, 0, v[\vtmp1] src0_sel:WORD_0 src1_sel:WORD_1
        v_cndmask_b32 v[\base], v[\base], v[\vtmp1], vcc dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_1
    .endif
.endm

.macro activ_f_float base, activ_mode, alpha, beta, gamma, vtmp0, vtmp1
    .if \activ_mode == MIOPEN_NEURON_LOGISTIC //1 / (1 + e^-x)
        exp_f_float \base, -1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        v_rcp_f32 v[\base], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_TANH // \beta * tanh(\alpha * x)
        v_mul_f32 v[\base], s[\alpha], v[\base]
        v_mul_f32 v[\base], 2.0, v[\base]
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        v_rcp_f32 v[\base], v[\base]
        v_mul_f32 v[\base], 2.0, v[\base]
        v_sub_f32 v[\base], 1.0, v[\base]
        v_mov_b32 v[\vtmp0], 1.0
        v_mul_f32 v[\base], s[\beta], v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_RELU //max(0, x)
        v_max_f32 v[\base], v[\base], 0
    .elseif \activ_mode == MIOPEN_NEURON_SOFTRELU //log(1 + e^x)
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], 1.0, v[\base]
        ln_f_float \base, \vtmp0
    .elseif \activ_mode == MIOPEN_NEURON_ABS //abs(x)
        v_max_f32 v[\base], v[\base], -v[\base]
    .elseif \activ_mode == MIOPEN_NEURON_POWER //(\alpha + \beta * x )^\gamma
        v_mul_f32 v[\base], s[\beta], v[\base]
        v_add_f32 v[\base], s[\alpha], v[\base]
        v_mov_b32 v[\vtmp0], v[\base]
        v_log_f32 v[\base], v[\base]
        v_mul_f32 v[\base], s[\gamma], v[\base]
        v_exp_f32 v[\base], v[\base]
        v_cmp_lt_f32 vcc, EPSILON_float, v[\vtmp0]
        v_cndmask_b32 v[\base], 0, v[\base], vcc
    .elseif \activ_mode == MIOPEN_NEURON_CLIPPED_RELU //min(\alpha, max(0, x))
        v_max_f32 v[\base], v[\base], 0
        v_min_f32 v[\base], s[\alpha], v[\base] 
    .elseif \activ_mode == MIOPEN_NEURON_LEAKY_RELU //\alpha * x | x <= 0; x | x > 0
        v_cmp_lt_f32 vcc, 0, v[\base]
        v_mov_b32 v[\vtmp0], s[\alpha]
        v_cndmask_b32 v[\vtmp0], v[\vtmp0], 1.0, vcc
        v_mul_f32 v[\base], v[\base], v[\vtmp0]
    .elseif \activ_mode == MIOPEN_NEURON_ELU //\alpha * (e^x - 1) | x <= 0; x | x > 0
        v_cmp_lt_f32 vcc, 0, v[\base]
        v_mov_b32 v[\vtmp1], v[\base]
        exp_f_float \base, 1, \vtmp0
        v_add_f32 v[\base], -1.0, v[\base]
        v_mul_f32 v[\base], s[\alpha], v[\base]
        v_cndmask_b32 v[\base], v[\base], v[\vtmp1], vcc
    .endif
.endm
