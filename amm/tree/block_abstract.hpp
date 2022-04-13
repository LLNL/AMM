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
#ifndef AMM_BLOCK_ABSTRACT_H
#define AMM_BLOCK_ABSTRACT_H

//! ----------------------------------------------------------------------------
#include <string>
#include <type_traits>
#include <unordered_map>

#include "macros.hpp"
#include "utils/exceptions.hpp"
#include "types/dtypes.hpp"

//#define AMM_DEBUG_BLOCK_ACCESS

//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! the base class for all types of blocks
template <typename TypeValue>
class block {

    static_assert(std::is_floating_point<TypeValue>::value, "block requires a floating-point type!");
    static_assert(sizeof(TypeValue) == 4 || sizeof(TypeValue) == 8, "block requires a float32 or a float64!");

public:
    // common function to check if the vertex is valid
    static inline void validate(const TypeLocalIdx _, const std::string &caller) {
#ifdef AMM_DEBUG_BLOCK_ACCESS
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS,
                          "(%s) tried to invalid vertex %d\n", caller.c_str(), _);
#endif
    }

public:
    // abstract API
    inline size_t size() const ;
    inline void clear() ;

    inline bool contains(const TypeLocalIdx) const ;
    inline TypeValue get(const TypeLocalIdx) const ;
    inline TypeValue get(const TypeLocalIdx, bool &) const ;
    inline void set(const TypeLocalIdx, const TypeValue) ;
    inline void add(const TypeLocalIdx, const TypeValue) ;
    inline uint8_t precision() const ;

    inline void print(const int = -1) const ;
    inline void update(const std::unordered_map<TypeLocalIdx, TypeValue> &, bool do_add) ;

    inline float memory_in_kb() const;
};

//! ----------------------------------------------------------------------------
}   // end of namespace
#endif
