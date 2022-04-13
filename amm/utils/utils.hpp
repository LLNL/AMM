/**
// =============================================================================
// Adaptive Multilinear Meshes
//
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and
// AMM Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3)
//
// =============================================================================
**/

#pragma once
#ifndef AMM_UTILS_H
#define AMM_UTILS_H
//! ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <climits>

#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include "amm/macros.hpp"
#include "amm/utils/exceptions.hpp"
#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"


//! ----------------------------------------------------------------------------
//! macros to handle validity of defined values
//! ----------------------------------------------------------------------------

#define AMM_is_valid_nbits(_)       ((_) > 0 && (_) <= AMM_MAX_NBITS)
#define AMM_is_valid_nbytes(_)      ((_) > 0 && (_) <= AMM_MAX_NBYTES)
#define AMM_is_valid_precision(_)   ((_) > 0 && (_) <= AMM_MAX_PRECISION)

//! ----------------------------------------------------------------------------
//! macros to handle zeros and nans
//! ----------------------------------------------------------------------------

#define AMM_nan(T)          std::numeric_limits<T>::signaling_NaN()
#define AMM_is_nan(_)       std::isnan(_)

#define AMM_zero(T)         T(0)
#define AMM_is_zero(_)      fabs(_) < amm::traits_nbytes<sizeof(_)>::epsilon

#define AMM_if_val(a,_)     (_?a:0.0)

//! ----------------------------------------------------------------------------
//! macros to handle powers of 2
//! ----------------------------------------------------------------------------

#define AMM_pow2(_)             (1u << (_))
#define AMM_log2(_)             (int(CHAR_BIT*sizeof(_) - amm::utils::bcount_lz(_))-1)

#define AMM_is_pow2(_)          ((_) && (!((_) & ((_)-1))))
#define AMM_is_pow2k(_,k)       (0 == ((_) & (AMM_pow2(k)-1)))

// dimensions and strides at level
#define AMM_ldim(_)             AMM_pow2(_+1)
#define AMM_ldx(_,L)            AMM_pow2(L-_)
#define AMM_ldim_half(_,l)      (((_-1)>>(l))+1)
#define AMM_lsize_dbl(_)        ((_ << 1) - TypeCoord(1))
#define AMM_lsize_half(_)       ((_ + TypeCoord(1)) >> 1)
#define AMM_lsize_rad(_)        ((_ - TypeCoord(1)) >> 1)
#define AMM_lsize_up(_,s)       (_ - s + TypeCoord(1))
#define AMM_lsize_dwn(_,s)      (_ + s - TypeCoord(1))

//! ----------------------------------------------------------------------------
//! computation of index using the x,y,z coordinates
//! ----------------------------------------------------------------------------

#define AMM_xy2idx(x,y,nx,ny)       ((x) + (nx)*(y))
#define AMM_xyz2idx(x,y,z,nx,ny,nz) ((x) + (nx)*((y) + (ny)*(z)))

#define AMM_xy2grid(x,y,d)           AMM_xy2idx(x,y,d[0],d[1])
#define AMM_xyz2grid(x,y,z,d)        AMM_xyz2idx(x,y,z,d[0],d[1],d[2])

//! ----------------------------------------------------------------------------
//! macros to manipulate bits
//! ----------------------------------------------------------------------------

#define AMM_bit32(b)            (1ul<<(b))
#define AMM_set_bit32(_,b)      (_ |= AMM_bit32(b))
#define AMM_clear_bit32(_,b)    (_ &= ~AMM_bit32(b))
#define AMM_is_set_bit32(_,b)   ((_ & AMM_bit32(b)) != 0)

//#define AMM_bit64(b)            (1ull<<(b))
//#define AMM_set_bit64(_,b)      (_ |= AMM_bit64(b))
//#define AMM_clear_bit64(_,b)    (_ &= ~AMM_bit64(b))
//#define AMM_is_set_bit64(_,b)   ((_ & AMM_bit64(b)) != 0)

//! ----------------------------------------------------------------------------
//! macros to compare strings
//! ----------------------------------------------------------------------------
#define AMM_is_same(a,b)     (a.compare(b) == 0)
#define AMM_starts_with(a,b) (a.rfind(b, 0) == 0)

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------

#ifdef AMM_USE_NAN_FOR_MISSING_VERTEX

// use nan to represent a missing value in the mesh (i.e, a vertex that doesnt exist)
#define AMM_missing_vertex(T)                   AMM_nan(T)
#define AMM_is_missing_vertex(_)                AMM_is_nan(_)
#define AMM_replace_missing_with_zeros(T,_)     std::for_each(_.begin(),_.end(),[](T &v){if(AMM_is_missing_vertex(v)) v=T(0)})
#define AMM_add_potential_missing(a,b)          (AMM_is_missing_vertex(a)?0.0:(a)) + (AMM_is_missing_vertex(b)?0.0:(b))

#else

// use 0 to represent a missing value in the mesh (i.e, a vertex that doesnt exist)
#define AMM_missing_vertex(T)                   AMM_zero(T)
#define AMM_is_missing_vertex(_)                AMM_is_zero(_)
#define AMM_replace_missing_with_zeros(T,_)     (_)
#define AMM_add_potential_missing(a,b)          a+b

#endif
//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
namespace amm {
namespace utils {

//! ----------------------------------------------------------------------------
//! bit counters
//! ----------------------------------------------------------------------------

#if defined(__GNUC__) || defined(__MINGW32__) || defined(__clang__)

template<typename T>
inline static
size_t
bcount_ones(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    return sizeof(T) <= 4 ? __builtin_popcount(_) : __builtin_popcountll(_);
}


template <typename T>
inline static
size_t
bcount_lz(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    static_assert(sizeof(T) <= 8, "T can have a max size of 8 bytes!");

    if (_ == 0)
        return sizeof (T)*CHAR_BIT;

    switch (sizeof(T)) {
        case 1: return __builtin_clz(_) - 24;
        case 2: return __builtin_clz(_) - 16;
        case 3: return __builtin_clz(_) -  8;
        case 4: return __builtin_clz(_);
        case 5: return __builtin_clzll(_) - 24;
        case 6: return __builtin_clzll(_) - 16;
        case 7: return __builtin_clzll(_) -  8;
        case 8: return __builtin_clzll(_);
    }
}


template <typename T>
inline static
int
bcount_tz(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    return sizeof(T) <= 4 ? __builtin_ctz(_) : __builtin_ctzll(_);
}



#elif defined(_MSC_VER)
#include <intrin.h>

#pragma intrinsic(_BitScanReverse, _BitScanReverse64, _BitScanForward, _BitScanForward64)
inline int nlz32(unsigned x) {
  int n;

  if (x == 0) return(32);
  n = 0;
  if (x <= 0x0000FFFF) { n = n + 16; x = x << 16; }
  if (x <= 0x00FFFFFF) { n = n + 8; x = x << 8; }
  if (x <= 0x0FFFFFFF) { n = n + 4; x = x << 4; }
  if (x <= 0x3FFFFFFF) { n = n + 2; x = x << 2; }
  if (x <= 0x7FFFFFFF) { n = n + 1; }
  return n;
}

inline int nlz64(unsigned long long x) {
  int n;

  if (x == 0) return(64);
  n = 0;
  if (x <= 0x00000000FFFFFFFFul) { n = n + 32; x = x << 32; }
  if (x <= 0x0000FFFFFFFFFFFFul) { n = n + 16; x = x << 16; }
  if (x <= 0x00FFFFFFFFFFFFFFul) { n = n + 8; x = x << 8; }
  if (x <= 0x0FFFFFFFFFFFFFFFul) { n = n + 4; x = x << 4; }
  if (x <= 0x3FFFFFFFFFFFFFFFul) { n = n + 2; x = x << 2; }
  if (x <= 0x7FFFFFFFFFFFFFFFul) { n = n + 1; }
  return n;
}

//! --------------------------------------------------------------------------------
//! return the position of the least significant bit (from the right)
//! if x = 0, return 32
//! --------------------------------------------------------------------------------
inline int ntz32(unsigned x) {
  int n;

  if (x == 0) return(32);
  n = 1;
  if ((x & 0x0000FFFF) == 0) { n = n + 16; x = x >> 16; }
  if ((x & 0x000000FF) == 0) { n = n + 8; x = x >> 8; }
  if ((x & 0x0000000F) == 0) { n = n + 4; x = x >> 4; }
  if ((x & 0x00000003) == 0) { n = n + 2; x = x >> 2; }
  return n - (x & 1);
}

//! --------------------------------------------------------------------------------
//! return the position of the least significant bit (from the right)
//! if x = 0, return 64
//! --------------------------------------------------------------------------------
inline int ntz64(unsigned long long x) {
  int n;

  if (x == 0) return(64);
  n = 1;
  if ((x & 0x00000000FFFFFFFFul) == 0) { n = n + 32; x = x >> 32; }
  if ((x & 0x000000000000FFFFul) == 0) { n = n + 16; x = x >> 16; }
  if ((x & 0x00000000000000FFul) == 0) { n = n + 8; x = x >> 8; }
  if ((x & 0x000000000000000Ful) == 0) { n = n + 4; x = x >> 4; }
  if ((x & 0x0000000000000003ul) == 0) { n = n + 2; x = x >> 2; }
  return n - (x & 1);
}

template<typename T>
inline static int bcount_ones(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    //! based on https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    //if (_ == 0) return 0;
    T v = _;
    int c = 0;
    for (; v; c++) v &= v-1;
    return c;
}
template <typename T>
inline static int bcount_lz(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    return //_ == 0 ? CHAR_BIT*sizeof(T) :
           sizeof(T) <= 4 ? nlz32(unsigned(_)) : nlz64(unsigned long long(_));
}
template <typename T>
inline int bcount_tz(const T _) {
    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    return //_ == 0 ? CHAR_BIT*sizeof(T) :
           sizeof(T) <= 4 ? ntz32(unsigned(_)) : ntz64(unsigned long long(_));
}
#endif



//! ----------------------------------------------------------------------------
//! error and norm computations
//! ----------------------------------------------------------------------------
/* Compute the l2 norm of a function */
template<typename T>
inline static
double
l2norm(const T *data, const size_t sz) {
    double l2 = 0;
    for(size_t i=0; i < sz; ++i) {
        const double v = static_cast<double>(data[i]);
        l2 += (v * v);
    }
    return std::sqrt(l2);
}


/* Compute the mean squared error between two functions */
template <typename T>
inline static
double
mse(const T* u, const T* v, const size_t bounds[3]) {

    double err = 0;
    const size_t sz = bounds[0]*bounds[1]*bounds[2];
    for (size_t i = 0; i < sz; ++i) {
        const double diff = static_cast<double>(v[i]-u[i]);
        err += (diff * diff);
    }
    return err / double(sz);
}


/* Compute the root mean squared error between two functions */
template <typename T>
inline static
double
rmse(const T* u, const T* v, const size_t bounds[3]) {
    return std::sqrt(mse(u, v, bounds));
}


/* Compute the peak signal to noise ratio between two functions */
template <typename T>
inline static
double
psnr(const T* u, const T* v, const size_t bounds[3]) {


    T _min = u[0];
    T _max = u[0];

    const size_t sz = bounds[0]*bounds[1]*bounds[2];
    for (size_t i = 0; i < sz; ++i) {
        _min = (u[i] < _min) ? u[i] : _min;
        _max = (u[i] > _max) ? u[i] : _max;
    }
    const double d = 0.5*(_max-_min);
    return 20.0*std::log10(d) - 10.0*std::log10(mse(u, v, bounds));
}


//! ----------------------------------------------------------------------------
//! function range computation
//! ----------------------------------------------------------------------------
/* Compute the range of a function given as an array */
template <typename T>
inline static
std::pair<T,T>
frange(const T *f, const size_t sz) {

    if(f == nullptr)    return std::make_pair(T(0), T(0));
    auto res = std::minmax_element(f, f+sz);
    return std::make_pair(*res.first, *res.second);
}

/* Compute the range of a function given as a vector */
template <typename T>
inline static
std::pair<T,T>
frange(const std::vector<T> &f) {

    if(f.empty())       return std::make_pair(T(0), T(0));
    auto res = std::minmax_element(f.begin(), f.end());
    return std::make_pair(*res.first, *res.second);
}


/* Compute the range of a function given as a map */
template <typename Idx, typename T>
inline static
std::pair<T,T>
frange(const std::unordered_map<Idx, T> &f) {

    if(f.empty())   return std::make_pair(T(0), T(0));

    auto iter = f.begin();
    T _min = iter->second;
    T _max = iter->second;

    for(++iter; iter != f.end(); ++iter) {
        if (iter->second > _max) _max = iter->second;
        if (iter->second < _min) _min = iter->second;
    }
    return std::make_pair(_min, _max);
}



//! ----------------------------------------------------------------------------------
inline static
TypeIndex
next_pow2_plus_1(const TypeIndex n) {
    // return the same value,
    //      if n = 0 or n == 1 or n = 2^k + 1
    return (n < 2) || AMM_is_pow2(n-1) ? n : AMM_pow2(AMM_log2(n-1)+1) + 1;
}

//! ----------------------------------------------------------------------------------

/*
// -----------------------------------------------------------------------------------
// not used Dec 2021
//------------------------------------------------------------------------------------
//! ----------------------------------------------------------------------------------
//! is a given number a multiple of 2?
//!     type = 0:   check for any multiple
//!     type = 1:   check for odd multiple
//!     type = 2:   check for even multiple
inline bool is_multiple_of_pow2(TypeCoord p, TypeScale l, uint8_t type = 0) {

    // everything is a multiple pf 2^0 = 1
    if(l == 0)
        return true;

    // remainder should be zero
    TypeCoord r = p & (AMM_pow2(l)-1);
    if (r != 0)
        return false;

    // divisor
    TypeCoord d = p >> l;
    switch(type) {
        case 1:    return (d&1) == 1;       // odd multiple
        case 2:    return (d&1) == 0;       // even multiple
    }
    return true;
}

//! ----------------------------------------------------------------------------------
//! snap a number to the nearest power of 2
inline TypeCoord snap_to_pow2(TypeCoord p, TypeScale l, uint8_t type = 0) {

    // divisor
    TypeCoord d = p >> l;

    switch(type) {

        // if d is even, round up to the next odd
        case 1:     if ((d&1) == 0)     d = d+1;    break;

        // if d is odd, round up to the next even
        case 2:     if ((d&1) == 1)     d = d+1;    break;
    }
    return (d * AMM_pow2(l));
}

//! ----------------------------------------------------------------------------------
//! get the next 2^l + 1 (l <= L)
inline static TypeIndex next_pow2_plus_1(TypeIndex n, TypeScale L) {

    // 0 or 1, return the same value
    if (n < 2)
        return n;

    // catch edge case where n is a power of 2 + 1
    if (AMM_is_pow2(n-1))
        return n;

    // search for next power of 2
    for(TypeScale i = 1; i <= L; i++) {
        TypeIndex p = AMM_pow2(i);
        if (p >= n)
            return p+1;
    }
    return AMM_pow2(L)+1;
}


//! find largest p, such that if x = k * 2^p
template<typename T>
inline void factorize_powerof2(const T x, T &p, T &k) {

    p = 0;  k = 0;
    if (x == 0)
        return;

    for(T y = x; !(y & 1); y >>= 1, p++ ){}
    k = x / AMM_pow2(p);
}

//! snap a float/double to the nearest (odd / even / any) integer
template<typename T>
inline int snap2nearestInteger(T x, uint8_t type = 0) {

    switch (type) {

        // nearest odd
        case 1:     return std::nearbyint( 0.5f*x + 0.5f ) * 2 - 1;

        // nearest even
        case 2:     return std::nearbyint( 0.5f*x ) * 2;
    }

    // nearest integer
    return std::nearbyint(x);
}

//! find if a is a (odd / even / any) multiple of b
template<typename T>
inline bool is_multiple(T a, T b, uint8_t type = 0) {

    if (b == 1)         return true;
    if (b == 0)         return false;
    if (a%b != 0)       return false;

    switch (type) {

        // odd multiple
        case 1:     return a/b % 2 == 1;

        // even multiple
        case 2:     return a/b % 2 == 0;
    }

    // any multiple
    return true;
}

//! compute the next power of 2
template<typename T>
inline T next_pow2(T n) {

    for(uint8_t i = 0; true; i++) {
        T p = AMM_pow2(i);
        if (p >= n)
            return p;
    }
    return 0;
}

//! test for existence in a vector
template<typename T>
inline bool contains(const std::vector<T> &list, const T &val) {
    return std::find(list.begin(), list.end(), val) != list.end();
}
// -----------------------------------------------------------------------------------
// not used Dec 2021
//------------------------------------------------------------------------------------
*/

//! ----------------------------------------------------------------------------
//! float comparisons
//! ----------------------------------------------------------------------------
template <typename T>
inline bool
approximatelyZero(T a, T epsilon = amm::traits_nbytes<sizeof(T)>::tolerance) {
    return std::abs(a) <= epsilon;
}

template <typename T>
inline bool
essentiallyZero(T a, T epsilon = amm::traits_nbytes<sizeof(T)>::tolerance) {
    return std::abs(a) <= epsilon;
}

template <typename T>
inline bool
approximatelyEqual(T a, T b, T epsilon = amm::traits_nbytes<sizeof(T)>::tolerance) {
    T fa = std::abs(a);
    T fb = std::abs(b);
    return std::abs(a - b) <= ((fa < fb ? fb : fa) * epsilon);
}

template <typename T>
inline bool
essentiallyEqual(T a, T b, T epsilon = amm::traits_nbytes<sizeof(T)>::tolerance) {
    T fa = std::abs(a);
    T fb = std::abs(b);
    return std::abs(a - b) <= ((fa < fb ? fa : fb) * epsilon);
}
//! ----------------------------------------------------------------------------

}}   // end of namespace
#endif
