uint iDiv_legacy(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return (r);
}

uint iDiv(uint v, uint d)
{
    uint r = v / d;
    return (r);
}

uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24((uint)u, (uint)d);
    return (r);
}
