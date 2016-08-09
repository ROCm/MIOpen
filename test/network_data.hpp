
#include <mlopen/tensor.hpp>

template<class T, class F>
void visit_network(F f)
{
    mlopenDataType_t t = mlopenFloat; // TODO: Compute this from T
    f(mlopen::TensorDescriptor(t, { 100, 3,    32,  32  }), mlopen::TensorDescriptor(t, { 3,    32,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 100, 32,   16,  16  }), mlopen::TensorDescriptor(t, { 32,   32,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 100, 32,   8,   8   }), mlopen::TensorDescriptor(t, { 32,   64,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 256, 3,    227, 227 }), mlopen::TensorDescriptor(t, { 3,    96,   11,  11 }));
    f(mlopen::TensorDescriptor(t, { 256, 96,   27,  27  }), mlopen::TensorDescriptor(t, { 96,   256,  5,   5  }));
    f(mlopen::TensorDescriptor(t, { 256, 256,  13,  13  }), mlopen::TensorDescriptor(t, { 256,  384,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 256, 384,  13,  13  }), mlopen::TensorDescriptor(t, { 384,  384,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 256, 384,  13,  13  }), mlopen::TensorDescriptor(t, { 384,  256,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  3,    224, 224 }), mlopen::TensorDescriptor(t, { 3,    64,   7,   7  }));
    f(mlopen::TensorDescriptor(t, { 32,  64,   56,  56  }), mlopen::TensorDescriptor(t, { 64,   64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  64,   56,  56  }), mlopen::TensorDescriptor(t, { 64,   192,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  192,  28,  28  }), mlopen::TensorDescriptor(t, { 192,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  192,  28,  28  }), mlopen::TensorDescriptor(t, { 192,  96,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  96,   28,  28  }), mlopen::TensorDescriptor(t, { 96,   128,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  192,  28,  28  }), mlopen::TensorDescriptor(t, { 192,  16,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  16,   28,  28  }), mlopen::TensorDescriptor(t, { 16,   32,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  192,  28,  28  }), mlopen::TensorDescriptor(t, { 192,  32,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  256,  28,  28  }), mlopen::TensorDescriptor(t, { 256,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  256,  28,  28  }), mlopen::TensorDescriptor(t, { 256,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  128,  28,  28  }), mlopen::TensorDescriptor(t, { 128,  192,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  256,  28,  28  }), mlopen::TensorDescriptor(t, { 256,  32,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  32,   28,  28  }), mlopen::TensorDescriptor(t, { 32,   96,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  256,  28,  28  }), mlopen::TensorDescriptor(t, { 256,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  480,  14,  14  }), mlopen::TensorDescriptor(t, { 480,  192,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  480,  14,  14  }), mlopen::TensorDescriptor(t, { 480,  96,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  96,   14,  14  }), mlopen::TensorDescriptor(t, { 96,   208,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  480,  14,  14  }), mlopen::TensorDescriptor(t, { 480,  16,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  16,   14,  14  }), mlopen::TensorDescriptor(t, { 16,   48,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  480,  14,  14  }), mlopen::TensorDescriptor(t, { 480,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  4,   4   }), mlopen::TensorDescriptor(t, { 512,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  160,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  112,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  112,  14,  14  }), mlopen::TensorDescriptor(t, { 112,  224,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  24,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  24,   14,  14  }), mlopen::TensorDescriptor(t, { 24,   64,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  128,  14,  14  }), mlopen::TensorDescriptor(t, { 128,  256,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  24,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  24,   14,  14  }), mlopen::TensorDescriptor(t, { 24,   64,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  112,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  144,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  144,  14,  14  }), mlopen::TensorDescriptor(t, { 144,  288,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  32,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  32,   14,  14  }), mlopen::TensorDescriptor(t, { 32,   64,   5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  64,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  528,  4,   4   }), mlopen::TensorDescriptor(t, { 528,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  528,  14,  14  }), mlopen::TensorDescriptor(t, { 528,  256,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  528,  14,  14  }), mlopen::TensorDescriptor(t, { 528,  160,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  160,  14,  14  }), mlopen::TensorDescriptor(t, { 160,  320,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  528,  14,  14  }), mlopen::TensorDescriptor(t, { 528,  32,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  32,   14,  14  }), mlopen::TensorDescriptor(t, { 32,   128,  5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  528,  14,  14  }), mlopen::TensorDescriptor(t, { 528,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  256,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  160,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  160,  7,   7   }), mlopen::TensorDescriptor(t, { 160,  320,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  32,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  32,   7,   7   }), mlopen::TensorDescriptor(t, { 32,   128,  5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  384,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  192,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  192,  7,   7   }), mlopen::TensorDescriptor(t, { 192,  384,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  48,   1,   1  }));
    f(mlopen::TensorDescriptor(t, { 32,  48,   7,   7   }), mlopen::TensorDescriptor(t, { 48,   128,  5,   5  }));
    f(mlopen::TensorDescriptor(t, { 32,  832,  7,   7   }), mlopen::TensorDescriptor(t, { 832,  128,  1,   1  }));
    f(mlopen::TensorDescriptor(t, { 128, 3,    231, 231 }), mlopen::TensorDescriptor(t, { 3,    96,   11,  11 }));
    f(mlopen::TensorDescriptor(t, { 128, 96,   28,  28  }), mlopen::TensorDescriptor(t, { 96,   256,  5,   5  }));
    f(mlopen::TensorDescriptor(t, { 128, 256,  12,  12  }), mlopen::TensorDescriptor(t, { 256,  512,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 128, 512,  12,  12  }), mlopen::TensorDescriptor(t, { 512,  1024, 3,   3  }));
    f(mlopen::TensorDescriptor(t, { 128, 1024, 12,  12  }), mlopen::TensorDescriptor(t, { 1024, 1024, 3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  3,    224, 224 }), mlopen::TensorDescriptor(t, { 3,    64,   3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  64,   112, 112 }), mlopen::TensorDescriptor(t, { 64,   128,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  128,  56,  56  }), mlopen::TensorDescriptor(t, { 128,  256,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  256,  56,  56  }), mlopen::TensorDescriptor(t, { 256,  256,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  256,  28,  28  }), mlopen::TensorDescriptor(t, { 256,  512,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  512,  28,  28  }), mlopen::TensorDescriptor(t, { 512,  512,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  512,  3,   3  }));
    f(mlopen::TensorDescriptor(t, { 64,  512,  14,  14  }), mlopen::TensorDescriptor(t, { 512,  512,  3,   3  }));
}