#ifndef GUARD_CALC_ERR_
#define GUARD_CALC_ERR_

template <typename T>
double CalcErr(T c_val, T g_val)
{
    double err = -1.0;
    if(sizeof(T) == 2)
    {
        int16_t* c_uval = reinterpret_cast<int16_t*>(&c_val);
        int16_t* g_uval = reinterpret_cast<int16_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }
    else if(sizeof(T) == 4)
    {
        int32_t* c_uval = reinterpret_cast<int32_t*>(&c_val);
        int32_t* g_uval = reinterpret_cast<int32_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }
    else if(sizeof(T) == 8)
    {
        int64_t* c_uval = reinterpret_cast<int64_t*>(&c_val);
        int64_t* g_uval = reinterpret_cast<int64_t*>(&g_val);
        err             = static_cast<double>(std::abs(*c_uval - *g_uval));
    }

    //		double delta = abs(c_val - g_val);
    //	double nextafter_delta = nextafterf(min(abs(c_val), abs(g_val)), (T)INFINITY) -
    // min(abs(c_val), abs(g_val));
    //		err = delta / nextafter_delta;
    return err;
}

#endif // GUARD_GUARD_CALC_ERR_
