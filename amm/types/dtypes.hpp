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
#ifndef AMM_DTYPES_H
#define AMM_DTYPES_H

//! ----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <ostream>

// TODO: at the very end, move these types of the namespace
//namespace amm {

//! ----------------------------------------------------------------------------
//! data types used in the code
//! ----------------------------------------------------------------------------
using TypeDim      = std::uint8_t;      // type for dimensions  : need values in 0,1,2,3
                                            // also used for num_bits_per_level: 2,3 (basic octree) or 3,5 (advanced octree)
                                            // also used for sizeflags (three bits for long-ness in each dimension)
using TypeScale    = std::uint8_t;      // type for levels      : need values 0 to MAX_DATA_DEPTH
using TypeChildId  = std::uint8_t;      // type for child id    : need values 0 to 25
using TypeCornerId = std::uint8_t;      // type for corner id   : need values 0 to 7
                                            // also used for split ids : 5 in 2D and 19 in 3D
                                            // also used for anchor ids : need values up to 26
                                            // also used for idx in iterators : need values up to 27
using TypePrecision = std::uint8_t;
using TypeExponent  = std::int16_t;

using TypeIndex    = std::uint64_t;     // type for grid index
using TypeCoord    = std::uint64_t;     // type for x,y,z coordinates
//using TypeCoord    = std::uint32_t;     // type for x,y,z coordinates
// TODO: change TypeCoord back to u32

using TypeLocalIdx = uint8_t;       // uint8 is sufficient to store 64 indices

//}   // end of namespace


//! ----------------------------------------------------------------------------
//! generic ostream methods
//! ----------------------------------------------------------------------------


template <typename T>
inline
std::ostream&
operator<<(std::ostream &os, const std::pair<T,T> &p) {
    return os << "("<<p.first<<", "<<p.second<<")";
}


/*
inline std::ostream &operator<<(std::ostream &os, unsigned char c) {
    std::cerr << " couting uint << \n";
    exit (1);
    return os << static_cast<int>(c);
}
inline std::ostream &operator<<(std::ostream &os, signed char c) {
    std::cerr << " couting int << \n";
    exit (1);
    return os << static_cast<int>(c);
}


//! to apply ostream on arrays
template <typename T, unsigned int N>
typename std::enable_if<!std::is_same<T, char>::value, std::ostream &>::type
operator<<(std::ostream &os, const T (&arr)[N])
{
    int i;
    for(i = 0; i < N; i++)
        os << arr[i] << " ";
    os << std::endl;
    return os;
}
*/

//! ----------------------------------------------------------------------------

#endif
