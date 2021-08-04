#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#define UNUSED __attribute__((__unused__))

#ifndef MLO_LARGE_MAP
#define MLO_LARGE_MAP 0
#endif

#ifndef MLO_N_INPUTS_REMAINDER
#define MLO_N_INPUTS_REMAINDER 0
#endif

#ifndef MLO_N_IN_TILES_PERSTACK
#define MLO_N_IN_TILES_PERSTACK 0
#endif

#ifndef MLO_N_IN_CHNLS
#define MLO_N_IN_CHNLS 0
#endif

void calculateXYPos(uint linPos, uint width, uint* __restrict x, uint* __restrict y)
{
    (*y) = (uint)((float)linPos * (1.0f / (float)width) + 0.00001f);
    (*x) = linPos - mul24((*y), width);
}

uint calculateOffset(uint stride, uint x, uint y)
{
    uint ret = y * stride + x;
    return (ret);
}

void readDataElem(uint linPos,
                  __local _FLOAT* lcl_data,
                  uint lcl_base,
                  UNUSED uint lcl_height,
                  uint lcl_width,
                  uint lcl_stride,
                  uint lcl_y,
                  uint lcl_x,
                  const __global _FLOAT* gbl_data,
                  uint gbl_base,
                  uint gbl_height,
                  uint gbl_width,
                  uint gbl_stride,
                  uint gbl_y,
                  uint gbl_x,
                  bool vis,
                  UNUSED bool debug)
{
    uint x, y;
    calculateXYPos(linPos, lcl_width, &x, &y);
    uint g_x      = x + gbl_x;
    uint g_y      = y + gbl_y;
    uint gbl_off0 = calculateOffset(gbl_stride, g_x, g_y);
    uint gbl_off  = gbl_off0 + gbl_base;

#if MLO_LARGE_MAP == 1
    uint lcl_off = lcl_base + linPos;
    (void)lcl_stride;
    (void)lcl_x;
    (void)lcl_y;
#else
    uint l_x     = x + lcl_x;
    uint l_y     = y + lcl_y;
    uint lcl_off = lcl_base + mad24(l_y, lcl_stride, l_x);
#endif

#if MLO_LARGE_MAP == 1
    //	vis &= (g_x >= 0 && g_x < gbl_width && g_y >= 0 && g_y < gbl_height);
    vis &= (g_x < gbl_width && g_y < gbl_height);
#else
    (void)gbl_width;
    (void)gbl_height;
#endif
    gbl_off        = (vis) ? gbl_off : 0;
    _FLOAT gbl_val = gbl_data[gbl_off];
    gbl_val        = (vis) ? gbl_val : 0;

    lcl_data[lcl_off] = gbl_val;
}

void readData(uint lcl_id,
              uint size,
              uint lcl_p_stride,
              __local _FLOAT* lcl_data,
              uint lcl_base,
              uint lcl_height,
              uint lcl_width,
              uint lcl_stride,
              uint lcl_y,
              uint lcl_x,
              const __global _FLOAT* gbl_data,
              uint gbl_base,
              uint gbl_height,
              uint gbl_width,
              uint gbl_stride,
              uint gbl_y,
              uint gbl_x,
              bool vis,
              bool debug)
{

    for(uint i = lcl_id; i < size; i += lcl_p_stride)
    {
        readDataElem(i,
                     lcl_data,
                     lcl_base,
                     lcl_height,
                     lcl_width,
                     lcl_stride,
                     lcl_y,
                     lcl_x,
                     gbl_data,
                     gbl_base,
                     gbl_height,
                     gbl_width,
                     gbl_stride,
                     gbl_y,
                     gbl_x,
                     vis,
                     debug);
    }
}

void readDataVec2(uint lcl_id,
                  uint size,
                  uint lcl_p_stride,
                  __local _FLOAT2* lcl_data,
                  uint lcl_base,
                  UNUSED uint lcl_height,
                  uint lcl_width,
#if MLO_LARGE_MAP != 1
                  uint lcl_stride,
                  uint lcl_y,
                  uint lcl_x,
#endif
                  const __global _FLOAT* gbl_data,
                  uint2 gbl_base,
#if MLO_LARGE_MAP == 1
                  uint gbl_height,
                  uint gbl_width,
#endif
                  uint gbl_stride,
                  uint gbl_y,
                  uint gbl_x,
                  bool visX,
                  bool visY,
#if MLO_N_INPUTS_REMAINDER <= MLO_N_IN_TILES_PERSTACK
                  bool IsLast,
#endif
                  UNUSED bool debug)
{

    uint x, y;
    for(uint i = lcl_id; i < size; i += lcl_p_stride)
    {
        bool lvisX = visX, lvisY = visY;
        calculateXYPos(i, lcl_width, &x, &y);
        uint g_x         = x + gbl_x;
        uint g_y         = y + gbl_y;
        uint gbl_off0    = calculateOffset(gbl_stride, g_x, g_y);
        uint2 gbl_off_v2 = (uint2)(gbl_off0) + gbl_base;

#if MLO_LARGE_MAP == 1
        uint lcl_off = lcl_base + i;
        lvisX &= (g_x < gbl_width && g_y < gbl_height);
        lvisY &= (g_x < gbl_width && g_y < gbl_height);
#else
        uint l_x            = x + lcl_x;
        uint l_y            = y + lcl_y;
        uint lcl_off        = lcl_base + mad24(l_y, lcl_stride, l_x);
#endif
        lcl_data[lcl_off].x = (lvisX) ? gbl_data[gbl_off_v2.x] : (_FLOAT)0;
#if MLO_N_INPUTS_REMAINDER <= MLO_N_IN_TILES_PERSTACK
        lcl_data[lcl_off].y = (IsLast) ? (_FLOAT)0 : ((lvisY) ? gbl_data[gbl_off_v2.y] : (_FLOAT)0);
#else
        lcl_data[lcl_off].y = (lvisY) ? gbl_data[gbl_off_v2.y] : (_FLOAT)0;
#endif
    }
}

void readDataTile(__local _FLOAT* lcl_data,
                  const __global _FLOAT* gbl_data,
                  int tile_y,
                  int tile_x,
                  uint gbl_stride,
                  uint gbl_base,
                  uint lcl_stride,
                  uint lcl_base,
                  uint gbl_height,
                  uint gbl_width,
                  uint lcl_height,
                  uint lcl_width,
                  uint lcl_id1,
                  uint lcl_id0,
                  uint lcl_grp_sz1,
                  uint lcl_grp_sz0,
                  uint fltr_pad1,
                  uint fltr_pad0,
                  _FLOAT padding_val)
{
    for(uint j = lcl_id1; j < lcl_height; j += lcl_grp_sz1)
    {
        int y_act       = (j - fltr_pad1);
        bool invisibleY = (tile_y + y_act < 0) || (tile_y + y_act >= gbl_height);

        uint y_gbl_off = y_act * gbl_stride + gbl_base;

        uint y_lcl_off = j * lcl_stride + lcl_base;

        for(uint i = lcl_id0; i < lcl_width; i += lcl_grp_sz0)
        {
            int x_act       = (i - fltr_pad0);
            bool invisibleX = (tile_x + x_act < 0) || (tile_x + x_act >= gbl_width);

            bool invis = invisibleX || invisibleY;

            uint g_off = (invis) ? 0 : y_gbl_off + x_act;

            _FLOAT val = gbl_data[g_off];

            val = (invis) ? padding_val : val;

            lcl_data[y_lcl_off + i] = val;
        }
    }
}

void readDataTileVec2(__local _FLOAT2* lcl_data,
                      const __global _FLOAT* gbl_data,
                      int tile_y,
                      int tile_x,
                      uint gbl_stride,
                      uint2 gbl_base,
                      uint lcl_stride,
                      uint lcl_base,
                      uint gbl_height,
                      uint gbl_width,
                      uint lcl_height,
                      uint lcl_width,
                      uint lcl_id1,
                      uint lcl_id0,
                      uint lcl_grp_sz1,
                      uint lcl_grp_sz0,
                      uint fltr_pad1,
                      uint fltr_pad0,
#if MLO_N_IN_CHNLS % 2 == 1
                      bool IsLast,
#endif
                      _FLOAT padding_val)
{
    for(uint j = lcl_id1; j < lcl_height; j += lcl_grp_sz1)
    {
        int y_act          = (j - fltr_pad1);
        bool invisibleY    = (tile_y + y_act < 0) || (tile_y + y_act >= gbl_height);
        uint2 y_gbl_off_v2 = (uint2)(y_act * gbl_stride) + gbl_base;
        uint y_lcl_off     = j * lcl_stride + lcl_base;
        for(uint i = lcl_id0; i < lcl_width; i += lcl_grp_sz0)
        {
            int x_act       = (i - fltr_pad0);
            bool invisibleX = (tile_x + x_act < 0) || (tile_x + x_act >= gbl_width);
            bool invis      = invisibleX || invisibleY;
            uint2 g_off     = (invis) ? (uint2)(0) : y_gbl_off_v2 + (uint2)(x_act);
#if MLO_N_IN_CHNLS % 2 == 0
            lcl_data[y_lcl_off + i] =
                (invis) ? (_FLOAT2)(padding_val) : (_FLOAT2)(gbl_data[g_off.x], gbl_data[g_off.y]);
#else
            lcl_data[y_lcl_off + i].x = (invis) ? padding_val : gbl_data[g_off.x];
            lcl_data[y_lcl_off + i].y =
                (IsLast) ? (_FLOAT)0 : (invis) ? padding_val : gbl_data[g_off.y];
#endif
        }
    }
}

void loadData(uint lcl_id,
              uint lcl_p_stride,
              __local _FLOAT* lcl_data,
              uint lcl_off,
              uint lcl_size,
              uint lcl_height,
              uint lcl_width,
              uint lcl_stride,
              uint lcl_bot_y,
              uint lcl_bot_x,
              const __global _FLOAT* gbl_data,
              uint gbl_off,
              uint gbl_size,
              uint gbl_height,
              uint glb_width,
              uint gbl_stride,
              uint gbl_bot_y,
              uint gbl_bot_x,
              uint buf_block_ind,
              uint max_n_bufs,
              uint lcl_n_bufs,
              bool debug)
{

    for(uint c = 0; c < lcl_n_bufs; ++c, lcl_off += lcl_size, gbl_off += gbl_size)
    {
        bool vis = (buf_block_ind + c < max_n_bufs);
        readData(lcl_id,
                 lcl_size,
                 lcl_p_stride,
                 lcl_data,
                 lcl_off,
                 lcl_height,
                 lcl_width,
                 lcl_stride,
                 lcl_bot_y,
                 lcl_bot_x,
                 gbl_data,
                 gbl_off,
                 gbl_height,
                 glb_width,
                 gbl_stride,
                 gbl_bot_y,
                 gbl_bot_x,
                 vis,
                 (debug));
    }
}

#pragma GCC diagnostic pop
