#ifndef __PRECISION_H
#define __PRECISION_H
#ifdef __cplusplus
extern "C"
{
#endif
#ifdef USE_DOUBLE
    typedef double real_t;
#else
    typedef float real_t;
#endif
#ifdef __cplusplus
}
#endif
#endif