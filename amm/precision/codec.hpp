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
#ifndef AMM_CODEC_H
#define AMM_CODEC_H

//! ----------------------------------------------------------------------------
#include <math.h>
#include <limits.h>
#include <algorithm>
#include <type_traits>
#include <unordered_map>

#include "macros.hpp"
#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"
#include "utils/utils.hpp"
#include "utils/logger.hpp"


//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! functions to extract exponents from a floating-point value
//! ----------------------------------------------------------------------------
template <typename _Float>
inline
int
exp(const _Float v) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    //static constexpr int ebias = (1 << (CHAR_BIT*sizeof(int)-1)) - 1;
    const int ebias = (1 << (CHAR_BIT*sizeof(int)-1)) - 1;

    if (fabs(v) > 0) {
        static int e;
        frexp(v, &e);
        return int(std::max(e, 1-ebias));
    }

    // @TODO: this doesn't actually seem to be an issue now, might be ok to just remove the warning message
    AMM_log_error << "\nDetected extraction of exponent from 0 value scalar: may cause incorrect behavior\n";
    return int(-ebias);
}


template <typename _Float>
inline
int
exp(const std::vector<_Float> &vals) {
    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");

    int e = 0;
    std::for_each(vals.begin(), vals.end(),
                  [&e](const _Float &v){
                        e = std::max(e, exp<_Float>(fabs(v)));
                  });
    return e;
}


template <typename _Idx, typename _Float>
inline
int
exp(const std::unordered_map<_Idx, _Float> &vmap) {
    static_assert(std::is_unsigned<_Idx>::value, "_Idx is required to be an unsigned type!");
    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");

    int e = 0;
    std::for_each(vmap.begin(), vmap.end(),
                  [&e](const std::pair<_Idx, _Float> &v){
                        e = std::max(e, exp<_Float>(fabs(v.second)));
                  });
    return e;
}


//! ----------------------------------------------------------------------------
//! functions to encode/decode a floating point value as unsigned integers
//! ----------------------------------------------------------------------------

// encode/decode based on the data type
template <typename _Float, typename _Quant>
inline
_Quant
encode(const _Float v, const int e) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");

    using QTraits = amm::traits_nbytes<sizeof(_Quant)>;

    static constexpr int nbits = CHAR_BIT*sizeof(_Float);

    typename QTraits::signed_t ival = ldexp(double(v), (nbits - 2) - e);
    return _Quant((ival ^ QTraits::negabinary_mask) - QTraits::negabinary_mask);
}


template <typename _Float, typename _Quant>
inline
_Float
decode(const _Quant v, const int e)  {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");

    using QTraits = amm::traits_nbytes<sizeof(_Quant)>;

    static constexpr int nbits = CHAR_BIT*sizeof(_Float);

    typename QTraits::signed_t ival = ((v + QTraits::negabinary_mask) ^ QTraits::negabinary_mask);
    return _Float(ldexp(double(ival), e - (nbits - 2)));
}


// encode/decode based on the the number of bytes
template <typename _Float, typename _Quant, TypePrecision N>
inline
_Quant
encoden(const _Float v, const int e) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(N), "Invalid N!");

    static constexpr size_t shft = CHAR_BIT*(sizeof(_Quant) - N);

    return encode<_Float, _Quant>(v, e) >> shft;
}


template <typename _Float, typename _Quant, TypePrecision N>
inline
_Float
decoden(_Quant v, const int e) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(N), "Invalid N!");

    static constexpr size_t shft = CHAR_BIT*(sizeof(_Quant) - N);

    return decode<_Float, _Quant>(v << shft, e);
}

//! ----------------------------------------------------------------------------
} // end of namespace amm
#endif
