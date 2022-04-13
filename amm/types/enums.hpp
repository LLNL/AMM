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
#ifndef AMM_ENUMS_H
#define AMM_ENUMS_H

//! ----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <type_traits>

// @TODO: at the very end, move these types of the namespace
//namespace amm {

//! ----------------------------------------------------------------------------
//! enumerations for different constructs
//! ----------------------------------------------------------------------------

//! type of wavelet coefficient
enum struct EnumWCoefficient: std::uint8_t {

    None = 8,

    // ordering: zyx
      S = 0,   W = 1,                       // 1D
     SS = 0,  SW = 1,  WS = 2,  WW = 3,     // 2D
    SSS = 0, SSW = 1, SWS = 2, SWW = 3,     // 3D (z = scaling)
    WSS = 4, WSW = 5, WWS = 6, WWW = 7      // 3D (z = wavelet)
};

//! type of data extrapolation
enum struct EnumExtrapolation: std::uint8_t {
    None = 0,
    Zero = 1,
    Linear = 2,
    LinearLifting = 3
};

//! enumeration of edges and faces
enum struct EnumCell: std::uint8_t {
    None = 255,
    MnX = 0, MxX = 1, MnY = 2, MxY = 3, MnZ = 4, MxZ = 5,
    MnXMnZ = 0, MxXMnZ = 1, MnYMnZ = 2, MxYMnZ = 3,     // 3D Front
    MnXMnY = 4, MxXMnY = 5, MnXMxY = 6, MxXMxY = 7,     // 3D Side
    MnXMxZ = 8, MxXMxZ = 9, MnYMxZ = 10, MxYMxZ = 11    // 3D Back
};

//! enumeration of the three axes
enum struct EnumAxes: std::uint8_t {
    None = 0,
    X = 1, Y = 2, Z = 4,
    XY = 3, XZ = 5, YZ = 6,
    XYZ = 7
};

//! enumeration of axes directions
enum struct EnumDimension: std::uint8_t {
    None = 255,
    X = 0, Y = 1, Z = 2
};


//! ----------------------------------------------------------------------------
//! convenience functions for enums
//! ----------------------------------------------------------------------------

template <class Enum>
inline bool operator == (const Enum &e, const int &v) { return static_cast<int>(e) == v;    }
template <class Enum>
inline bool operator >= (const Enum &e, const int &v) { return static_cast<int>(e) >= v;    }
template <class Enum>
inline bool operator >  (const Enum &e, const int &v) { return static_cast<int>(e) > v;     }
template <class Enum>
inline bool operator <= (const Enum &e, const int &v) { return static_cast<int>(e) <= v;    }
template <class Enum>
inline bool operator <  (const Enum &e, const int &v) { return static_cast<int>(e) < v;     }


template<class Enum>
constexpr auto as_utype(const Enum value) {
    using _tp = typename std::underlying_type<Enum>::type;
    static_assert(std::is_unsigned<_tp>::value, "as_utype is designed to work with enums of unsigned types!");
    return static_cast<_tp>(value);
}

template<class Enum>
constexpr auto as_enum(const typename std::underlying_type<Enum>::type value) {
    return static_cast<Enum>(value);
}

template<class Enum, typename T>
constexpr auto as_enum(const T value) {

    static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
    using _tp = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<_tp>(value));
}

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
//}   // end of namespace
#endif
