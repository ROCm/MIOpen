.include "gpr_alloc.inc"
.include "common.inc"

.altmacro		 
// limits:
// N, C, H, W, K, R, S, n_groups < 2^16
// n_groups * tiles_per_wave < 2^16
// wino_c < 2^24
// out_w == 3
// out_h == 3
// pad_w <= 1
// input layout NCHW or CNHW
// output layout HWNC

// kernarg layout:
// dwords 0 	uint32_t N;
// dwords 1 	uint32_t C;
// dwords 2 	uint32_t H;
// dwords 3 	uint32_t W;
//
// dwords 4 	uint32_t K;
// dwords 5 	uint32_t n_groups;
// dwords 6 	uint32_t flags;
// dwords 7 	uint32_t reserved;
//
// dwords 8:9	uint64_t  data_addr;
// dwords 10:11	uint64_t  filter_addr;
// dwords 12:13 uint64_t  output_addr;
// dwords 14:15	uint64_t  return_addr;
//
// dwords 16	uint32_t  R;	// filter height
// dwords 17	uint32_t  S;	// filter width
// dwords 18	int32_t   pad_h;	// padding
// dwords 19	int32_t   pad_w;	// padding
//
// dwords 20	uint32_t  out_h;	// output height
// dwords 21	uint32_t  out_w;	// output width
//
// dwords 22:23	uint64_t bias_addr;
// dwords 24	float RELU_alpha;
//
// dwords 25	uint32_t d_N_stride;
// dwords 26	uint32_t d_C_stride;
// dwords 27	uint32_t d_H_stride;
// dwords 28	uint32_t d_W_stride;
//
// dwords 29	uint32_t f_K_stride;
// dwords 30	uint32_t f_C_stride;
// dwords 31	uint32_t f_R_stride;
// dwords 32	uint32_t f_S_stride;
//
// dwords 33	uint32_t o_N_stride;
// dwords 34	uint32_t o_K_stride;
// dwords 35	uint32_t o_H_stride;
// dwords 36	uint32_t o_W_stride;

default pipe_depth, 4
default xformx_f_size, 4 // 2..6
default xformy_f_size, 4 // 2..6
default xformx_o_size, 3 // 3
default xformy_o_size, 3 // 3
default elem_size, 4
xformx_d_size = xformx_f_size + xformx_o_size - 1
xformy_d_size = xformy_f_size + xformy_o_size - 1
.if xformx_d_size > xformy_d_size
	tiles_per_wave = wave_size / xformx_d_size
.else
	tiles_per_wave = wave_size / xformy_d_size
.endif
slot_size = xformx_d_size

.text
.p2align 8
.if xform_filter
	.globl gcnAsmWinogradXformFilter
	.type gcnAsmWinogradXformFilter,@function
	.amdgpu_hsa_kernel gcnAsmWinogradXformFilter
.else
	.globl gcnAsmWinogradXformData
	.type gcnAsmWinogradXformData,@function
	.amdgpu_hsa_kernel gcnAsmWinogradXformData
.endif

static_assert(xformx_f_size >=2 && xformx_f_size <= 6)
static_assert(xformy_f_size >=2 && xformy_f_size <= 6)
static_assert(xformx_o_size == 3)
static_assert(xformy_o_size == 3)
static_assert(elem_size == 4)
static_assert(pipe_depth & 1 == 0)

.if xform_filter
	in_tile_width  = xformx_f_size
	in_tile_height = xformy_f_size
	NK = K
	HR = R
	WS = S
.else
	in_tile_width  = xformx_d_size
	in_tile_height = xformy_d_size
	NK = N
	HR = H
	WS = W
.endif


.GPR_ALLOC_BEGIN
// initial state
// s[0:1] - kernarg address
// s2 - wg x (1 wg per CU)
kernarg = 0
gid_x = 2
div_c = 3
.SGPR_ALLOC_FROM 4
// following sgprs should be allocated in strict sequence to follow kernarg layout
.SGPR_ALLOC N
.SGPR_ALLOC C
.SGPR_ALLOC H
.SGPR_ALLOC W

.SGPR_ALLOC K
.SGPR_ALLOC n_groups
.SGPR_ALLOC flags
.SGPR_ALLOC unused1 // reserved

.SGPR_ALLOC d_addr, 2
.SGPR_ALLOC f_addr, 2
.SGPR_ALLOC o_addr, 2
.SGPR_ALLOC dbg_addr, 2

.SGPR_ALLOC R // filter_h
.SGPR_ALLOC S // filter_w
.SGPR_ALLOC pad_h
.SGPR_ALLOC pad_w

.SGPR_ALLOC out_h
.SGPR_ALLOC out_w

.SGPR_ALLOC unused2, 2 // bias_addr
.SGPR_ALLOC unused3 // RELU_alpha

.SGPR_ALLOC d_N_stride
.SGPR_ALLOC d_C_stride
.SGPR_ALLOC d_H_stride
.SGPR_ALLOC d_W_stride

.SGPR_ALLOC f_K_stride
.SGPR_ALLOC f_C_stride
.SGPR_ALLOC f_H_stride
.SGPR_ALLOC f_W_stride

.SGPR_ALLOC o_N_stride
.SGPR_ALLOC o_C_stride
.SGPR_ALLOC o_H_stride
.SGPR_ALLOC o_W_stride

// end of kernarg extent
.if .SGPR_NEXT_FREE % 2
	.SGPR_ALLOC_ONCE chw_step
.endif
.SGPR_ALLOC stmp, 2
.SGPR_ALLOC valid_mask, 2
//.SGPR_ALLOC handler_ptr, 2
//.SGPR_ALLOC ret_ptr, 2
.SGPR_ALLOC_ONCE chw_step
//.SGPR_ALLOC scur_n
.SGPR_ALLOC pipe_cnt
.SGPR_ALLOC frontend_finished

.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
vtmp_size = 4
.VGPR_ALLOC vtmp, vtmp_size
.VGPR_ALLOC voff_d
.VGPR_ALLOC voff_o
.VGPR_ALLOC vlocal_h
.VGPR_ALLOC vcur_w
.VGPR_ALLOC vcur_tw
.VGPR_ALLOC vcur_th
.VGPR_ALLOC vcur_c
.VGPR_ALLOC vcur_n
.VGPR_ALLOC vlds_waddr
.VGPR_ALLOC vlds_raddr
.VGPR_ALLOC rdbuf, slot_size * pipe_depth
.VGPR_ALLOC wrbuf, xformy_d_size
.VGPR_ALLOC oaddrbuf, pipe_depth
.VGPR_ALLOC waddrbuf, pipe_depth


.LDS_ALLOC_FROM 0
lds_hstride = elem_size * (wave_size + tiles_per_wave)
lds_buf_size = xformy_d_size * lds_hstride
.LDS_ALLOC lds_buf_even, lds_buf_size
.LDS_ALLOC lds_buf_odd,  lds_buf_size


.GPR_ALLOC_END


//.text 0
//.p2align 8
.if xform_filter
	gcnAsmWinogradXformFilter:
.else
	gcnAsmWinogradXformData:
.endif


	.amd_kernel_code_t
		enable_sgpr_kernarg_segment_ptr = 1
		compute_pgm_rsrc2_tgid_x_en = 1
		is_ptr64 = 1
		compute_pgm_rsrc1_vgprs = .AUTO_VGPR_GRANULATED_COUNT
		compute_pgm_rsrc1_sgprs = .AUTO_SGPR_GRANULATED_COUNT
		compute_pgm_rsrc2_tidig_comp_cnt = 0
		compute_pgm_rsrc2_user_sgpr = 2
		kernarg_segment_byte_size = 148
		wavefront_sgpr_count = .AUTO_SGPR_COUNT
		workitem_vgpr_count = .AUTO_VGPR_COUNT
		float_mode = 192
		workgroup_group_segment_byte_size = .AUTO_LDS_BYTE_SIZE
	.end_amd_kernel_code_t
	
	

	s_load_dwordx16 s[N:dbg_addr+1], s[kernarg:kernarg+1], 0x0
	s_load_dwordx16 s[R:f_H_stride], s[kernarg:kernarg+1], 0x4 * 16
	s_load_dwordx4 s[f_W_stride:o_H_stride], s[kernarg:kernarg+1], 0x4 * 32
	s_load_dword   s[o_W_stride], s[kernarg:kernarg+1], 0x4 * 36

	s_waitcnt 0

	// compute wino_c and base_tile id
	.GPR_REUSE unused1, base_tile
	.GPR_REUSE unused2, tiles_w
	tiles_h = tiles_w + 1
	.GPR_REUSE unused3, wino_c
	.GPR_REUSE out_h, neg_tiles_h
	.GPR_REUSE out_w, neg_tiles_w
	.GPR_REUSE flags, neg_c
	_s_ceil_u32 s[tiles_w], s[S], %xformx_f_size
	_s_ceil_u32 s[tiles_h], s[R], %xformy_f_size
	s_mul_i32 s[wino_c], s[tiles_w], s[tiles_h]
	s_mul_i32 s[wino_c], s[wino_c], s[C]
	s_mul_i32 s[base_tile], 0 + tiles_per_wave, s[gid_x]
	s_mul_i32 s[chw_step], 0 + tiles_per_wave, s[n_groups]
	s_sub_i32 s[neg_tiles_h], 0, s[tiles_h]
	s_sub_i32 s[neg_tiles_w], 0, s[tiles_w]
	s_sub_i32 s[neg_c],  0, s[C]
	
	// early exit
	err = stmp+1
	u16limit = frontend_finished
	s_mov_b32 s[err], 0
	s_mov_b32 s[u16limit], 1<<16
	s_mul_i32 s[stmp], s[wino_c], s[NK] // total tiles
	s_cmp_ge_u32 s[base_tile], s[stmp]
	s_cmov_b32 s[err], 1
	s_cmp_ge_u32 s[H], s[u16limit]
	s_cmov_b32 s[err], 1
	s_cmp_ge_u32 s[W], s[u16limit]
	s_cmov_b32 s[err], 1
	s_cmp_ge_u32 s[wino_c], 1<<30
	s_cmov_b32 s[err], 1
	s_cmp_ge_u32 s[NK], s[u16limit]
	s_cmov_b32 s[err], 1
	s_cmp_gt_u32 s[pad_h], 1
	s_cmov_b32 s[err], 1
	s_cmp_gt_u32 s[pad_w], 1
	s_cmov_b32 s[err], 1
	s_cmp_ge_u32 s[chw_step], s[u16limit]
	s_cmov_b32 s[err], 1
	s_cmp_eq_u32 s[err], 1
	s_cbranch_scc1 endpgm
	.GPR_INVALIDATE err
	.GPR_INVALIDATE u16limit

	// construct buffer descriptors
	// size covers whole buffer
	.GPR_REUSE d_addr, d_desc
	.GPR_REUSE o_addr, o_desc
	.GPR_INVALIDATE f_addr
	.GPR_INVALIDATE dbg_addr
	s_mov_b32 s[d_desc+3], 0x00020000
	s_mov_b32 s[o_desc+3], 0x00020000
	s_mul_i32 s[d_desc+2], s[HR], s[WS]
	s_mul_i32 s[d_desc+2], s[d_desc+2], s[NK]
	s_mul_i32 s[d_desc+2], s[d_desc+2], s[C]
	s_mulk_i32 s[d_desc+2], 0 + elem_size
	s_mul_i32 s[o_desc+2], s[wino_c], s[NK]
	s_mulk_i32 s[o_desc+2], 0 + elem_size * xformx_d_size * xformy_d_size
	
	// compute divisors
	//.GPR_REUSE f_K_stride, div_n
	.GPR_REUSE f_C_stride, soff_o
	.GPR_REUSE f_H_stride, div_th
	.GPR_REUSE f_W_stride, div_tw
	v_writelane_b32 v[vtmp], s[NK],      0
	v_writelane_b32 v[vtmp], s[C],       1
	v_writelane_b32 v[vtmp], s[tiles_h], 2
	v_writelane_b32 v[vtmp], s[tiles_w], 3
	ceil_2_32_div_u16 v[vtmp], v[vtmp], vtmp+1, stmp
	//v_readlane_b32 s[div_n], v[vtmp],  0
	v_readlane_b32 s[div_c], v[vtmp],  1
	v_readlane_b32 s[div_th], v[vtmp], 2
	v_readlane_b32 s[div_tw], v[vtmp], 3

	// compute indices and address
	_v_div_const_u32_u16 v[vlocal_h], v[tid], %tiles_per_wave, s[stmp]
	v_mul_u32_u24 v[vtmp], 0 + tiles_per_wave, v[vlocal_h]
	v_sub_u32 v[vlds_waddr], v[tid], v[vtmp]
	v_add_u32 v[vcur_tw], s[base_tile], v[vlds_waddr]
	s_mov_b32 s[stmp], 1<<16
	v_cmp_gt_u32 vcc, s[stmp], v[vcur_tw]
	s_cbranch_vccz endpgm
	v_mov_b32 v[vcur_th], 0
	v_mov_b32 v[vcur_c], 0
	v_mov_b32 v[vcur_n], 0

	v_lshlrev_b32 v[vlds_waddr], 2, v[vlds_waddr]
	v_mul_u32_u24 v[vtmp], 0 + lds_hstride, v[vlocal_h]
	v_add_u32 v[vlds_waddr], v[vlds_waddr], v[vtmp]
	v_lshlrev_b32 v[vlds_raddr], 2, v[tid]
	v_mov_b32 v[vtmp], 0x80000000
	v_cmp_ge_u32 vcc, 0 + xformy_d_size, v[vlocal_h]
	v_cndmask_b32 v[vlds_waddr], v[vtmp], v[vlds_waddr], vcc

	// init pipe related variables
	s_mov_b32 s[frontend_finished], 0
	s_sub_i32 s[pipe_cnt], 0, pipe_depth
	
	// enter main loop
	disable_srd o_desc
	s_branch loop_entrance

	.macro winograd_xform o_size, f_size, base_gpr, vtmp
		.irp i,0,1,2,3,4,5,6,7
			.if \i < (\o_size + \f_size - 1)
				d\i = \base_gpr + \i
			.endif
			.if \i < vtmp_size
				t\i = \vtmp + \i
			.endif
		.endr
		.if xform_filter
			.if \o_size == 3 && \f_size == 2
				v_mov_b32 v[d3], v[d1]
				v_sub_f32 v[d2], v[d0], v[d1] div:2
				v_add_f32 v[d1], v[d0], v[d1] div:2

				//{{ 1, 0 },
				// { 0.5, 0.5 },
				// { 0.5,-0.5 },
				// { 0, 1 }},
			.elseif \o_size == 3 && \f_size == 3
				v_mov_b32 v[d4], v[d2]
				v_fma_f32 v[d3], 2.0, v[d1], v[d0]
				v_mac_f32 v[d3], 4.0, v[d2]
				v_mul_f32 v[d3], 0.16666666667, v[d3] // 1/6
				v_add_f32 v[t0], v[d0], v[d2]
				v_mul_f32 v[d0], 0.5, v[d0]
				v_sub_f32 v[d2], v[d1], v[t0]
				v_mul_f32 v[d2], 0.16666666667, v[d2] // 1/6
				v_add_f32 v[d1], v[d1], v[t0]
				v_mul_f32 v[d1], -0.5, v[d1]

				//d[4] = d[2];
				//d[3] = ((d[0] + 2*d[1]) + 4*d[2]) * 0.16666666667f; // 1/6;
				//float f = d[0] + d[2];
				//d[0] = d[0] * 0.5f;
				//d[2] = (d[1] - f) * 0.16666666667f; // 1/6;
				//d[1] = -0.5f * (d[1] + f);

				//{{ 0.5, 0, 0 },
				// {-0.5,-0.5,-0.5 },
				// {-1.0/6, 1.0/6, -1.0/6 },
				// { 1.0/6, 1.0/3, 2.0/3 },
				// { 0, 0, 1 }},
			.elseif \o_size == 3 && \f_size == 4
				v_fma_f32 v[t0], 4.0, v[d2], v[d0]
				v_fma_f32 v[t1], 4.0, v[d3], v[d1]
				v_add_f32 v[d4], v[d0], v[d2]
				v_add_f32 v[d5], v[d1], v[d3]
				v_mul_f32 v[d0], 0.25, v[d0]
				v_add_f32 v[d1], v[d4], v[d5]
				v_mul_f32 v[d1], -0.16666666667, v[d1] // -1/6
				v_sub_f32 v[d2], v[d4], v[d5]
				v_mul_f32 v[d2], -0.16666666667, v[d2] // -1/6
				v_mov_b32 v[d5], v[d3]
				v_fma_f32 v[d3],  2.0, v[t1], v[t0]
				v_fma_f32 v[d4], -2.0, v[t1], v[t0]
				v_mul_f32 v[d3], 0.04166666667, v[d3] // 1/24
				v_mul_f32 v[d4], 0.04166666667, v[d4] // 1/24

				//d[5] = d[3];
				//float f1 = -d[0] - d[2];
				//float f2 = d[1] + d[3];
				//float f3 = d[0] + 4 * d[2];
				//float f4 = 2*d[1] + 8*d[3];
				//d[0] = d[0]*0.25f;
				//d[1] = (f1 - f2) * 0.16666666667f; // 1/6;
				//d[2] = (f1 + f2) * 0.16666666667f; // 1/6;
				//d[3] = (f3 + f4) * 0.04166666667f; // 1/24;
				//d[4] = (f3 - f4) * 0.04166666667f; // 1/24;

				//{{ 0.25, 0, 0, 0 },
				// {-1.0/6, -1.0/6, -1.0/6,-1.0/6 },
				// {-1.0/6,  1.0/6, -1.0/6, 1.0/6 },
				// { 1.0/24, 1.0/12, 1.0/6, 1.0/3 },
				// { 1.0/24,-1.0/12, 1.0/6,-1.0/3 },
				// { 0, 0, 0, 1 }},
			.elseif \o_size == 3 && \f_size == 5
				//d[6] = d[4];
				//d[5] = (32*d[0] + (16*d[1] + (8*d[2] + (4*d[3] + 2*d[4])))) * 0.02222222222f; // 1/45;
				//float f1 = d[0] + d[2] + d[4];
				//float f2 = d[1] + d[3];
				//float f3 = (d[0] + 4*d[2]) + 16*d[4];
				//float f4 = 2*d[1] + 8*d[3];
				//d[0] = 0.5f * d[0];
				//d[1] = (-f1 - f2) * 0.33333333333f; // 1/3;
				//d[2] = (f1 - f2)  * 0.11111111111f; // 1/9;
				//d[3] = (f3 + f4)  * 0.02777777778f; // 1/36;
				//d[4] = (-f3 + f4) * 0.01666666667f; // 1/60;

				//{{ 0.5, 0, 0, 0, 0, 0 },
				// { -1.0/3,  -1.0/3,  -1.0/3, -1.0/3,  -1.0/3 },
				// {  1.0/9,  -1.0/9,   1.0/9, -1.0/9,   1.0/9 },
				// { 1.0/36,  1.0/18,   1.0/9,  2.0/9,   4.0/9 },
				// {-1.0/60,  1.0/30, -1.0/15, 2.0/15, -4.0/15 },
				// {32.0/45, 16.0/45,  8.0/45, 4.0/45,  2.0/45 },
				// { 0, 0, 0, 0, 1 }},
			.elseif \o_size == 3 && \f_size == 6
				//d[0] = d[0];
				//d[7] = d[5];
				//float f1 = -d[0] - d[2] - d[4];
				//float f2 = d[1] + d[3] + d[5];
				//float f3 = (d[0] + 4*d[2]) + 16*d[4];
				//float f4 = (2*d[1] + 8*d[3]) + 32*d[5];
				//float f5 = 32*d[0] + (8*d[2] + 2*d[4]);
				//float f6 = 16*d[1] + (4*d[3] + d[5]);
				//d[1] = (f1 - f2) * 0.22222222222f; // 2/9;
				//d[2] = (f1 + f2) * 0.22222222222f; // 2/9;
				//d[3] = (f3 + f4) * 0.01111111111f; // 1/90;
				//d[4] = (f3 - f4) * 0.01111111111f; // 1/90;
				//d[5] = (f5 + f6) * 0.02222222222f; // 1/45;
				//d[6] = (f5 - f6) * 0.02222222222f; // 1/45;

				//{{ 1, 0, 0, 0, 0, 0 },
				// {  -2.0/9,   -2.0/9, -2.0/9,  -2.0/9, -2.0/9,   -2.0/9 },
				// {  -2.0/9,    2.0/9, -2.0/9,   2.0/9, -2.0/9,    2.0/9 },
				// {  1.0/90,   1.0/45, 2.0/45,  4.0/45, 8.0/45,  16.0/45 },
				// {  1.0/90,  -1.0/45, 2.0/45, -4.0/45, 8.0/45, -16.0/45 },
				// { 32.0/45,  16.0/45, 8.0/45,  4.0/45, 2.0/45,   1.0/45 },
				// { 32.0/45, -16.0/45, 8.0/45, -4.0/45, 2.0/45,  -1.0/45 },
				// { 0, 0, 0, 0, 0, 1 }},
			.else
				static_assert(0)
			.endif
		.else
			.if \o_size == 3 && \f_size == 2
				v_sub_f32 v[d0], v[d0], v[d2]
				v_sub_f32 v[d3], v[d3], v[d1]
				v_sub_f32 v[t0], v[d2], v[d1]
				v_add_f32 v[d1], v[d1], v[d2]
				v_mov_b32 v[d2], v[t0]

				//d[0] = d[0] - d[2];
				//d[3] = d[3] - d[1];
				//float f = d[2] - d[1];
				//d[1] = d[1] + d[2];
				//d[2] = f;

				//{{ 1, 0,-1, 0 },
				// { 0, 1, 1, 0 },
				// { 0,-1, 1, 0 },
				// { 0,-1, 0, 1 }},
			.elseif \o_size == 3 && \f_size == 3
				v_sub_f32 v[d0], v[d0], v[d2] mul:2
				v_sub_f32 v[d4], v[d4], v[d2]
				v_fma_f32 v[t0], -2.0, v[d1], v[d3]
				v_fma_f32 v[t1],  2.0, v[d1], v[d3]
				v_sub_f32 v[d3], v[d3], v[d1]
				v_add_f32 v[d0], v[d0], v[d3]
				v_mac_f32 v[d4], -2.0, v[d3]
				v_sub_f32 v[d1], v[t0], v[d2]
				v_mul_f32 v[t0], -3.0, v[d2]
				v_add_f32 v[d2], v[t0], v[t1]

				//float f = d[3] - d[1];
				//d[0] = 2*(d[0] - d[2]) + f;
				//d[4] = d[4] - d[2] - 2*f;
				//float f2 = -2*d[1] - d[2] + d[3];
				//float f3 = 2*d[1] - 3*d[2] + d[3];
				//d[1] = f2;
				//d[2] = f3;
				//d[3] = f;

				//{{ 2,-1,-2, 1, 0 },
				// { 0,-2,-1, 1, 0 },
				// { 0, 2,-3, 1, 0 },
				// { 0,-1, 0, 1, 0 },
				// { 0, 2,-1,-2, 1 }},
			.elseif \o_size == 3 && \f_size == 4
				v_fma_f32 v[d0], 4.0, v[d0], v[d4]
				v_mac_f32 v[d0], -5.0, v[d2]
				v_fma_f32 v[d5], 4.0, v[d1], v[d5]
				v_mac_f32 v[d5], -5.0, v[d3]

				v_sub_f32 v[t0], v[d3], v[d1] mul:2
				v_sub_f32 v[t1], v[d4], v[d2]
				v_fma_f32 v[t2], -4.0, v[d2], v[d4]
				
				v_fma_f32 v[d1], -4.0, v[d1], v[d3]
				v_sub_f32 v[d2], v[t2], v[d1]
				v_add_f32 v[d1], v[t2], v[d1]
				v_add_f32 v[d3], v[t1], v[t0]
				v_sub_f32 v[d4], v[t1], v[t0]

				//d[0] = (4*d[0] + d[4]) - 5 * d[2];
				//d[5] = (4*d[1] + d[5]) - 5 * d[3];
				//float f1 = d[3] - 4*d[1];
				//float f2 = d[4] - 4*d[2];
				//float f3 = d[3] - d[1];
				//float f4 = d[4] - d[2];
				//d[1] = f2 + f1;
				//d[2] = f2 - f1;
				//d[3] = f4 + 2*f3;
				//d[4] = f4 - 2*f3;

				//{{ 4, 0,-5, 0, 1, 0 },
				// { 0,-4,-4, 1, 1, 0 },
				// { 0, 4,-4,-1, 1, 0 },
				// { 0,-2,-1, 2, 1, 0 },
				// { 0, 2,-1,-2, 1, 0 },
				// { 0, 4, 0,-5, 0, 1 }},
			.elseif \o_size == 3 && \f_size == 5
				//d[0] = ((2*d[0] - 2.5f*d[2]) + (0.5f*d[4] - d[5])) + (5*d[3] - 4*d[1]);
				//d[6] = ((-2*d[1] + 2.5f*d[3]) - (0.5f*d[5] - d[6])) - (5*d[4] - 4*d[2]);
				//float d1 = ((d[5] + 0.5f*d[4]) + 2*d[1]) - (2*d[2] + 4.5f*d[3]);
				//float d2 = ((d[5] - 1.5f*d[4]) - 2*d[1]) + (6*d[2] - 3.5f*d[3]);
				//float f = d[4] - d[2];
				//float d3 = ((d[5] + d[1]) + 1.5f*f) - 2*d[3];
				//float d4 = ((d[5] - d[1]) - 2.5f*f);
				//d[5] += 4*d[1] - 5*d[3];
				//d[1] = d1;
				//d[2] = d2;
				//d[3] = d3;
				//d[4] = d4;

				//{{ 2, -4, -2.5,    5,  0.5,   -1, 0 },
				// { 0,  2,   -2, -4.5,  0.5,    1, 0 },
				// { 0, -2,    6, -3.5, -1.5,    1, 0 },
				// { 0,  1, -1.5,   -2,  1.5,    1, 0 },
				// { 0, -1,  2.5,    0, -2.5,    1, 0 },
				// { 0,  4,    0,   -5,    0,    1, 0 },
				// { 0, -2,    4,  2.5,   -5, -0.5, 1 }},
			.elseif \o_size == 3 && \f_size == 6
				//d[0] = (d[0] - d[6]) + 5.25f*(d[4] - d[2]);
				//d[7] = (d[7] - d[1]) + 5.25f*(d[3] - d[5]);
				//float f1 = (d[1] + d[5]) - 4.25f*d[3];
				//float f2 = (d[2] + d[6]) - 4.25f*d[4];
				//float a = 2.5f * d[3];
				//float f3 = (0.5f*d[1] + 2*d[5]) - a;
				//float f4 = (0.25f*d[2] + d[6]) - 1.25f*d[4];
				//float f5 = (2*d[1] + 0.5f*d[5]) - a;
				//float f6 = (4*d[2] + d[6]) - 5*d[4];
				//d[1] = f2 + f1;
				//d[2] = f2 - f1;
				//d[3] = f4 + f3;
				//d[4] = f4 - f3;
				//d[5] = f6 + f5;
				//d[6] = f6 - f5;

				//{{ 1,    0, -5.25,     0,  5.25,    0, -1, 0 },
				// { 0,    1,     1, -4.25, -4.25,    1,  1, 0 },
				// { 0,   -1,     1,  4.25, -4.25,   -1,  1, 0 },
				// { 0,  0.5,  0.25,  -2.5, -1.25,    2,  1, 0 },
				// { 0, -0.5,  0.25,   2.5, -1.25,   -2,  1, 0 },
				// { 0,    2,     4,  -2.5,    -5,  0.5,  1, 0 },
				// { 0,   -2,     4,   2.5,    -5, -0.5,  1, 0 },
				// { 0,   -1,     0,  5.25,     0,-5.25,  0, 1 }},
			.else
				static_assert(0)
			.endif
		.endif
	.endm



	.macro single_load vgpr, dwords, dwords_left, count, soffset, reads
		.rept \count
			.if \dwords == 1
				buffer_load_dword v[\vgpr], v[voff_d], s[d_desc:d_desc+3], 0, offen offset:0+\soffset
			.else
				buffer_load_dwordx\dwords v[\vgpr:\vgpr+\dwords-1], v[voff_d], s[d_desc:d_desc+3], 0, offen offset:0+\soffset
			.endif
			\dwords_left = \dwords_left - \dwords
			\soffset = \soffset + 4 * \dwords
			\reads = \reads + 1
			\vgpr = \vgpr + \dwords
		.endr
	.endm

	.macro read_data buf, reads
		vgpr = \buf
		soffset = 0
		dwords_left = in_tile_width
		.if !xform_filter
			// first single dword load required because of possible padding
			single_load vgpr, 1, dwords_left, 1, soffset, \reads
		.endif
		
		single_load vgpr, 4, dwords_left, (dwords_left / 4), soffset, \reads
		single_load vgpr, 3, dwords_left, (dwords_left / 3), soffset, \reads
		single_load vgpr, 2, dwords_left, (dwords_left / 2), soffset, \reads
		single_load vgpr, 1, dwords_left, (dwords_left / 1), soffset, \reads
	.endm

	.macro write_data buf, voff
		i\@ = 0
		.rept xformy_d_size
			.if i\@ == 0
				s_mov_b32 s[soff_o], 0
			.else
				s_add_u32 s[soff_o], s[soff_o], s[o_H_stride]
			.endif
			buffer_store_dword v[\buf+i\@], v[\voff], s[o_desc:o_desc+3], s[soff_o], offen
			i\@ = i\@ + 1
		.endr
	.endm

	.macro normalize_nchw_idx_u16 n, c, th, tw, tmp
		v_mul_hi_u32 \tmp, s[div_tw], \tw
		v_cmp_eq_u32 vcc, 1, s[tiles_w]
		v_cndmask_b32 \tmp, \tmp, \tw, vcc
		v_mad_i32_i24 \tw, \tmp, s[neg_tiles_w], \tw

		v_add_u32 \th, \tmp, \th
		v_mul_hi_u32 \tmp, s[div_th], \th
		v_cmp_eq_u32 vcc, 1, s[tiles_h]
		v_cndmask_b32 \tmp, \tmp, \th, vcc
		v_mad_i32_i24 \th, \tmp, s[neg_tiles_h], \th

		v_add_u32 \c, \tmp, \c
		v_mul_hi_u32 \tmp, s[div_c], \c
		v_cmp_eq_u32 vcc, 1, s[C]
		v_cndmask_b32 \tmp, \tmp, \c, vcc
		v_mad_i32_i24 \c, \tmp, s[neg_c], \c

		v_add_u32 \n, \tmp, \n
	.endm
	
	.macro compute_voff off_d, off_o, w, n, c, th, tw, vtmp
		v_mul_lo_u32 \off_o, s[o_W_stride], v[vlocal_h]
		v_mad_u32_u24 \vtmp, s[tiles_h], \c, \th
		v_mad_u32_u24 \vtmp, s[tiles_w], \vtmp, \tw
		v_mul_lo_u32 \vtmp, s[o_C_stride], \vtmp
		v_add_u32 \off_o, \off_o, \vtmp
		v_mul_lo_u32 \vtmp, s[o_N_stride], \n
		v_add_u32 \off_o, \off_o, \vtmp

		v_mov_b32 \vtmp, 0x80000000
		v_cmp_lt_u32 s[valid_mask:valid_mask+1], \n, s[NK]
		v_cndmask_b32 \off_o, \vtmp, \off_o, s[valid_mask:valid_mask+1]
		v_cmp_gt_i32 vcc, 0 + tiles_per_wave * xformy_d_size, v[tid] 
		v_cndmask_b32 \off_o, \vtmp, \off_o, vcc

		v_mul_u32_u24 \w, 0 + xformx_f_size, \tw
		v_mad_u32_u24 \vtmp, 0 + xformy_f_size, \th, v[vlocal_h]
		v_cmp_gt_i32 vcc, 0 + tiles_per_wave * in_tile_height, v[tid] 
		s_and_b64 s[valid_mask:valid_mask+1], vcc, s[valid_mask:valid_mask+1]
		.if !xform_filter
			v_subrev_u32 \w, s[pad_w], \w
			v_subrev_u32 \vtmp, s[pad_h], \vtmp
			v_cmp_ge_i32 vcc, \vtmp, 0
			s_and_b64 s[valid_mask:valid_mask+1], vcc, s[valid_mask:valid_mask+1]
		.endif
		v_cmp_lt_i32 vcc, \vtmp, s[HR]
		s_and_b64 s[valid_mask:valid_mask+1], vcc, s[valid_mask:valid_mask+1]

		v_mul_lo_i32 \off_d, s[d_W_stride], \w
		v_mul_lo_u32 \vtmp, s[d_H_stride], \vtmp
		v_add_u32 \off_d, \off_d, \vtmp
		v_mul_lo_u32 \vtmp, s[d_C_stride], \c
		v_add_u32 \off_d, \off_d, \vtmp
		v_mul_lo_u32 \vtmp, s[d_N_stride], \n
		v_add_u32 \off_d, \off_d, \vtmp
		
		v_mov_b32 \vtmp, 0x80000000
		v_cndmask_b32 \off_d, \vtmp, \off_d, s[valid_mask:valid_mask+1]
	.endm

main_loop:

	slot = 0
	.rept pipe_depth
		.if slot & 1
			lds_roff = lds_buf_even	
			lds_woff = lds_buf_odd
		.else
			lds_roff = lds_buf_odd
			lds_woff = lds_buf_even	
		.endif
		rd_slot = slot
		xf_slot = (rd_slot + 1) % pipe_depth

		rd_base = rdbuf + slot_size * rd_slot
		xf_base = rdbuf + slot_size * xf_slot

		s_add_u32 s[pipe_cnt], s[frontend_finished], s[pipe_cnt]
		s_cbranch_scc1 endpgm
		
		s_cmp_eq_u32 s[frontend_finished], 1
		_s_cbranch scc1, skip_frontend, %slot

		v_add_u32 v[vcur_tw], s[chw_step], v[vcur_tw]
		.if slot == 0
			loop_entrance:
		.endif
		normalize_nchw_idx_u16 v[vcur_n], v[vcur_c], v[vcur_th], v[vcur_tw], v[vtmp]
		v_cmp_lt_u32 vcc, v[vcur_n], s[NK]
		_s_cbranch vccz, set_epilogue_state, %slot
		compute_voff v[voff_d], v[voff_o], v[vcur_w], v[vcur_n], v[vcur_c], v[vcur_th], v[vcur_tw], v[vtmp]
		_s_branch skip_frontend, %slot

label set_epilogue_state, %slot
		disable_srd d_desc
		s_mov_b32 s[frontend_finished], 1

label skip_frontend, %slot

		v_mov_b32 v[waddrbuf+rd_slot], v[vcur_w]

		// read columns from lds
		s_waitcnt lgkmcnt(0)
		i = 0
		.rept in_tile_height
			ds_read_b32 v[wrbuf + i], v[vlds_raddr] offset:0+lds_roff
			lds_roff = lds_roff + lds_hstride
			i = i + 1
		.endr

		reads_per_slot = 0
		read_data rd_base, reads_per_slot

		writes_per_slot = xformy_d_size
		s_waitcnt vmcnt(0+(pipe_depth - 1) * (reads_per_slot+xformy_d_size))
		//s_waitcnt 0

		// zeroing out of bounds vgprs
		i = 0
		.rept in_tile_width
			v_cmp_lt_u32 vcc, v[waddrbuf+xf_slot], s[WS]
			v_cndmask_b32 v[xf_base+i], 0, v[xf_base+i], vcc
			i = i + 1
			.if i != in_tile_height
				v_add_u32 v[waddrbuf+xf_slot], 1, v[waddrbuf+xf_slot]
			.endif
		.endr

		// transform each row
		winograd_xform xformx_o_size, xformx_f_size, xf_base, vtmp

		// write rows to lds
		i = 0
		.rept xformx_d_size
			ds_write_b32 v[vlds_waddr], v[xf_base + i], offset:0+lds_woff
			lds_woff = lds_woff + elem_size * tiles_per_wave
			i = i + 1
		.endr
		
		// transform each column
		s_waitcnt lgkmcnt(0+xformx_d_size)
		winograd_xform xformy_o_size, xformy_f_size, wrbuf, vtmp
		
		// store result
		write_data wrbuf, oaddrbuf+rd_slot
		v_mov_b32 v[oaddrbuf+rd_slot], v[voff_o]

		slot = slot + 1
	.endr

	enable_srd o_desc
	s_branch main_loop
main_loop_end:

	s_mov_b32 s0, s0 //debug

endpgm:

s_endpgm

.Lfunc_end0:
.if xform_filter
	.size gcnAsmWinogradXformFilter, .Lfunc_end0 - gcnAsmWinogradXformFilter
.else
	.size gcnAsmWinogradXformData, .Lfunc_end0 - gcnAsmWinogradXformData
.endif


.ifndef ROCM_METADATA_VERSION
    .error "ROCM_METADATA_VERSION must be defined"
.end
.endif

.macro METADATA wg_x, lds_size
  .if ROCM_METADATA_VERSION == 4
    .if xform_filter
        .amd_amdgpu_hsa_metadata
            { Version: [ 1, 0 ],
                Kernels:
                - { Name: gcnAsmWinogradXformFilter, SymbolName: 'gcnAsmWinogradXformFilter@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
                    Attrs:
                      { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                    CodeProps:
                      { KernargSegmentSize: 148, GroupSegmentFixedSize: \lds_size, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
                    Args:
                    - { Name: N       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: C       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: H       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: W       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: K       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: n_groups, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: flags   , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: unused_1, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: filter_ptr      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: reserved2       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: x_filter_ptr    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: ret_addr        , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*'  , AddrSpaceQual: Global, AccQual: Default }
                    - { Name: R       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: S       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: pad_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: pad_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: out_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: out_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: reserved3       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved4       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default, IsConst: true }
                    - { Name: d_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: reserved5       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved6       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved7       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved8       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                  }
            }
        .end_amd_amdgpu_hsa_metadata
    .else
        .amd_amdgpu_hsa_metadata
            { Version: [ 1, 0 ],
                Kernels:
                - { Name: gcnAsmWinogradXformData, SymbolName: 'gcnAsmWinogradXformData@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
                    Attrs:
                      { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
                    CodeProps:
                      { KernargSegmentSize: 148, GroupSegmentFixedSize: \lds_size, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
                    Args:
                    - { Name: N       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: C       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: H       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: W       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: K       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: n_groups, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: flags   , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: unused_1, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: filter_ptr      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: reserved2       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: x_filter_ptr    , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default}
                    - { Name: ret_addr        , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*'  , AddrSpaceQual: Global, AccQual: Default }
                    - { Name: R       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: S       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: pad_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: pad_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: out_h, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: out_w, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: reserved3       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved4       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: F32, TypeName: 'float', AccQual: Default, IsConst: true }
                    - { Name: d_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: d_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: reserved5       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved6       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved7       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }                    
                    - { Name: reserved8       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_N_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_C_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_H_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                    - { Name: o_W_stride, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
                  }
            }
        .end_amd_amdgpu_hsa_metadata
    .endif
    
  .else
    .error "Unsupported ROCM_METADATA_VERSION"
    .end
  .endif
.endm

.altmacro
.macro METADATA_WRAPPER wg_x, lds_size
    METADATA %\wg_x, %\lds_size
.endm

METADATA_WRAPPER 64, .AUTO_LDS_BYTE_SIZE

