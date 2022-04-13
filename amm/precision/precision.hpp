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
#ifndef AMM_PRECISION_H
#define AMM_PRECISION_H

//! ----------------------------------------------------------------------------
#include <limits.h>
#include <type_traits>
#include <vector>

#include "macros.hpp"
#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"
#include "precision/codec.hpp"
#include "utils/utils.hpp"

namespace amm {

//! ----------------------------------------------------------------------------
//! functions to change the precision of an unsigned data type
//! ----------------------------------------------------------------------------

template <TypePrecision Nin, TypePrecision Nout>
inline
typename amm::traits_nbytes<Nout>::unsigned_t
inflate(typename amm::traits_nbytes<Nin>::unsigned_t _) {

    static_assert(AMM_is_valid_precision(Nin), "Invalid Nin!");
    static_assert(AMM_is_valid_precision(Nout), "Invalid Nout!");
    static_assert(Nin <= Nout, "Nin should be smaller than Nout!");

    static constexpr size_t shft = CHAR_BIT*(Nout - Nin);

    using Tout = typename amm::traits_nbytes<Nout>::unsigned_t;
    return (Tout(_) << shft);
}


template <TypePrecision Nin, TypePrecision Nout>
inline
typename amm::traits_nbytes<Nout>::unsigned_t
deflate(typename amm::traits_nbytes<Nin>::unsigned_t _) {

    static_assert(AMM_is_valid_precision(Nin), "Invalid Nin!");
    static_assert(AMM_is_valid_precision(Nout), "Invalid Nout!");
    static_assert(Nin >= Nout, "Nin should be larger than Nout!");

    static constexpr size_t shft = CHAR_BIT*(Nin - Nout);

    using Tout = typename amm::traits_nbytes<Nout>::unsigned_t;
    return Tout(_ >> shft);
}


template <typename _Quant, TypePrecision N>
inline
_Quant
reduce(const _Quant _) {

    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(sizeof(_Quant)), "Invalid _Quant precision!");
    static_assert(AMM_is_valid_precision(N), "Invalid N!");

    if (sizeof(_Quant) <= N)
        return _;

    static constexpr size_t shft = CHAR_BIT*(sizeof(_Quant)-N);
    return ((_ >> shft) << shft);
}


template <typename _Quant>
inline
_Quant
reduce(const _Quant _, const TypePrecision &n) {

    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(sizeof(_Quant)), "Invalid _Quant precision!");

    if (!AMM_is_valid_precision(n)) {
        std::cerr << " reduce(_, "<<int(n)<<" got invalid precision!\n";
        exit(1);
    }
    if (sizeof(_Quant) <= n)
        return _;

    const size_t shft = CHAR_BIT*(sizeof(_Quant)-n);
    return ((_ >> shft) << shft);
}


//! ----------------------------------------------------------------------------
//! functions to test the suitability of precision
//! ----------------------------------------------------------------------------

//! get a suitable precision for a single unsigned value
template <typename _Quant>
inline
TypePrecision
precision(const _Quant _) {
    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(sizeof(_Quant)), "Invalid _Quant precision!");

    return sizeof(_Quant) - utils::bcount_tz<_Quant>(_)/CHAR_BIT;
}


template <typename _Float>
inline
TypePrecision
precisionf(const _Float _) {
    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");

    using _Quant = typename amm::traits_nbytes<sizeof(_Float)>::unsigned_t;
    return precision(encode<_Float, _Quant>(_, exp<_Float> (_)));
}


template <typename _Quant>
inline
TypePrecision
precision(const std::vector<_Quant> &vals) {

    static_assert(std::is_unsigned<_Quant>::value, "_Quant is required to be an unsigned type!");
    static_assert(AMM_is_valid_precision(sizeof(_Quant)), "Invalid _Quant precision!");

    TypePrecision prec = 1;

    const size_t &sz = vals.size();
    for(size_t i = 0; i < sz; i++){
        prec = std::max(prec, precision(vals[i]));
    }

    if (!AMM_is_valid_precision(prec)) {
        std::cerr << " precision computed invalid precision " << int(prec) << "!\n";
        exit(1);
    }
    return prec;
}



template <typename _Float>
inline TypePrecision
precisionf(const std::vector<_Float> &vals, const int e) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");
    using _Quant = typename amm::traits_nbytes<sizeof(_Float)>::unsigned_t;

    TypePrecision prec = 1;
    std::for_each(vals.begin(), vals.end(),
                  [&prec, e](const _Float &v){
                        prec = std::max(prec, precision(encode<_Float, _Quant>(v, e)));
                  });

    if (!AMM_is_valid_precision(prec)) {
        std::cerr << " precisionf :: invalid precision " << int(prec) << "!\n";
        exit(1);
    }
    return prec;
}


//! ----------------------------------------------------------------------------
//! reduce the precision of a float
//! ----------------------------------------------------------------------------

template <typename _Float>
inline
_Float
reduce_precision(const _Float _, const TypePrecision &n) {

    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");

    if (n >= sizeof(_Float))
        return _;

    using _Quant = typename amm::traits_nbytes<sizeof(_Float)>::unsigned_t;

    const int e = exp<_Float>(_);
    return decode<_Float, _Quant>(reduce<_Quant>(encode<_Float, _Quant>(_, e), n), e);
}

//! ----------------------------------------------------------------------------
} // end of namespace amm
#endif
