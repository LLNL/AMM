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
#ifndef AMM_BYTE_TRAITS_H
#define AMM_BYTE_TRAITS_H

#include <cstdint>
#include <limits>
#include <climits>

namespace amm {
//! ----------------------------------------------------------------------------
//! traits of data types based on number bytes
//! ----------------------------------------------------------------------------

template <uint8_t nbytes>
struct traits_nbytes;

template <>
struct traits_nbytes<1> {
    using signed_t = int8_t;
    using unsigned_t = uint8_t;
    using float_t = float;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-1);
    static constexpr unsigned_t negabinary_mask = 0xaa;
};

template <>
struct traits_nbytes<2> {
    using signed_t = int16_t;
    using unsigned_t = uint16_t;
    using float_t = float;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-3);
    static constexpr unsigned_t negabinary_mask = 0xaaaa;
};

template <>
struct traits_nbytes<3> {
    using signed_t = int32_t;
    using unsigned_t = uint32_t;
    using float_t = float;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-4);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaa;
};

template <>
struct traits_nbytes<4> {
    using signed_t = int32_t;
    using unsigned_t = uint32_t;
    using float_t = float;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-5);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaa;

    static constexpr int exponent_bits = 8;
    static constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
};

template <>
struct traits_nbytes<5> {
    using signed_t = int64_t;
    using unsigned_t = uint64_t;
    using float_t = double;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-6);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaaaaaaaaaaULL;
};

template <>
struct traits_nbytes<6> {
    using signed_t = int64_t;
    using unsigned_t = uint64_t;
    using float_t = double;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-7);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaaaaaaaaaaULL;
};

template <>
struct traits_nbytes<7> {
    using signed_t = int64_t;
    using unsigned_t = uint64_t;
    using float_t = double;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-8);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaaaaaaaaaaULL;
};

template <>
struct traits_nbytes<8> {
    using signed_t = int64_t;
    using unsigned_t = uint64_t;
    using float_t = double;

    static constexpr float_t epsilon = std::numeric_limits<float_t>::epsilon();
    static constexpr float_t tolerance = static_cast<float_t>(1e-12);
    static constexpr unsigned_t negabinary_mask = 0xaaaaaaaaaaaaaaaaULL;

    static constexpr int exponent_bits = 11;
    static constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
};


//! ----------------------------------------------------------------------------
//! traits of data types based on number bits
//! ----------------------------------------------------------------------------
template <uint8_t nbits>
struct traits_nbits : public traits_nbytes<(nbits+CHAR_BIT-1)/CHAR_BIT> {};

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------

}   // end of namespace
#endif
