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

.macro m_buffer_load_dwordx size, dst, off, desc, soff, ioff=0
	.if \size == 1
		buffer_load_dword v[\dst], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
	.elseif \size == 2
		buffer_load_dwordx2 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
	.elseif \size == 3
		buffer_load_dwordx3 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
	.elseif \size == 4
		buffer_load_dwordx4 v[\dst:\dst+\size-1], v[\off], s[\desc:\desc + 3], s[\soff] offen offset:0+\ioff
	.endif
.endm

.macro m_conv_accums dwords
	rotates_inflight = 0
	k_ds = 0
	.rept k_ds_rotates
		i = 0
		.rept \dwords
			kx = 0
			.rept k_mult
				base_out = lines_out + kx * read_size
				
				.if k_ds > 0
					rotates_inflight = rotates_inflight - 1
					s_wait , rotates_inflight
				.endif
				
				cx = 0
				.rept c_mult
					base_in = lines_in + cx * read_size
					acc = accums + k_per_gpr * (cx * k_mult + kx) + k_ds * k_dpp_rotates
					v_mac_f32 v[acc], v[base_out+i], v[base_in+i]
					k_dpp = 0
					.rept k_dpp_rotates - 1
						k_dpp = k_dpp + 1
						v_mac_f32 v[acc+k_dpp], v[base_out+i], v[base_in+i] row_ror:16*k_dpp/k_dpp_rotates
					.endr
					cx = cx + 1
				.endr
				
				.if (k_ds + 1) < k_ds_rotates
					static_assert (c_quads == 2 || c_quads == 4)
					.if c_quads == 2
						ds_swizzle_b32 v[base_out+i], v[base_out+i] offset:0xc200
					.elseif c_quads == 4
						ds_bpermute_b32 v[base_out+i], v[permute_addr], v[base_out+i]
					.endif
					rotates_inflight = rotates_inflight + 1
				.endif
				
				kx = kx + 1
			.endr
			i = i + 1
		.endr
		k_ds = k_ds + 1
	.endr
.endm

.macro m_acc_reduction first_round, rounds
	i = 0
	.rept \rounds
		round = i + \first_round
		acc = accums
		.rept accums_cnt
			.if i >= 1 && accums_cnt <= 2
				s_nop 2 - accums_cnt
			.endif
			.if round == 0
				v_add_f32 v[acc], v[acc], v[acc] quad_perm:[1,0,3,2]
			.elseif round == 1
				v_add_f32 v[acc], v[acc], v[acc] quad_perm:[2,3,0,1]
			.elseif round == 2
				v_add_f32 v[acc], v[acc], v[acc] row_ror:12
			.elseif round == 3
				v_add_f32 v[acc], v[acc], v[acc] row_ror:8
			.elseif round == 4
				static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:15
			.elseif round == 5
				static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:31
			.else
				static_assert (0)
			.endif
			acc = acc + 1
		.endr
		i = i + 1
	.endr
.endm

