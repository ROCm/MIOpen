
#ifndef MIOPEN_ACT_OP_ID
#define MIOPEN_ACT_OP_ID 0
#endif

// NOTE: only defining the ones we need to support for batch norm fused activation
#define MIOPEN_ACT_PASTHRU 0      // x
#define MIOPEN_ACT_RELU 3         // max(0, x)
#define MIOPEN_ACT_CLIPPED_RELU 7 // min(alpha, max(0, x))

// NOTE: all these defines need to be crafted to work with data that is both singular
// like float and half but also vector types like float4.

#if MIOPEN_ACT_OP_ID == MIOPEN_ACT_PASTHRU
#define FORWARD_ACTIVATION(res, data, alpha, beta, gamma) \
{ \
    res = data; \
}

#define BACKWARD_ACTIVATION(bot_diff, top_diff, bot_data, top_data, alpha, beta, gamma) \
{ \
    bot_diff = top_diff; \
}

#elif MIOPEN_ACT_OP_ID == MIOPEN_ACT_RELU
#define FORWARD_ACTIVATION(res, data, alpha, beta, gamma) \
{ \
    res = fmax(data, 0); \
}

#define BACKWARD_ACTIVATION(bot_diff, top_diff, bot_data, top_data, alpha, beta, gamma) \
{ \
    bot_diff = top_diff * (bot_data > 0); \
}

#elif MIOPEN_ACT_OP_ID == MIOPEN_ACT_CLIPPED_RELU
#define FORWARD_ACTIVATION(res, data, alpha, beta, gamma) \
{ \
    res = fmin(alpha, fmax(data, 0)); \
}

#define BACKWARD_ACTIVATION(bot_diff, top_diff, bot_data, top_data, alpha, beta, gamma) \
{ \
    bot_diff = top_diff * ((bot_data > 0 && bot_data <= alpha) ? 1.f : 0.f; \
}
#endif
