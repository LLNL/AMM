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
#ifndef AMM_LCODE_H
#define AMM_LCODE_H
//! ----------------------------------------------------------------------------

#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"
#include "containers/bitmask.hpp"
#include "utils/utils.hpp"
#include "utils/exceptions.hpp"

namespace amm {

//! ----------------------------------------------------------------------------
//! centralized functonality to manipulate location codes
//! stores BPL (bits per level) for L levels
//! and an extra bit for the root
//! ----------------------------------------------------------------------------

template<TypeScale L, TypeDim BPL>
struct location_code {

    static_assert (BPL*L+1 <= 64, "AMM_MAX_DATA_DEPTH should be <= 12; using datatype upto 64 bits!");

    using dtype = typename amm::traits_nbits<BPL*L+1>::unsigned_t;
    using btype = typename amm::bitmask<CHAR_BIT*sizeof(dtype)>;

    //! bitmap to extract the child id
    static constexpr dtype sChildMask = (1 << BPL)-1;
    static constexpr dtype sDimMaskX = 1;
    static constexpr dtype sDimMaskY = 2;
    static constexpr dtype sDimMaskZ = 4;

    //! compute the level of a location code
    static inline
    TypeScale
    level(const dtype _) {
        return (CHAR_BIT*sizeof(dtype)-amm::utils::bcount_lz(_)-1) / BPL;
    }

    //! location code of the parent node
    static inline
    dtype
    parent(const dtype _) {
        return _ >> BPL;
    }

    //! location code of the child node
    static inline
    dtype
    child(const dtype _, const TypeChildId cid) {
        return _ << BPL | cid;
    }

    //! location code of n'th ancestor
    static inline
    dtype
    parentn(const dtype _, const uint8_t n) {
        return _ >> (BPL*n);
    }

    //! location code of n'th descendent (child id = 0)
    static inline
    dtype
    childn(const dtype _, const uint8_t n) {
        return _ << (BPL*n);
    }

    //! given the location code of a node, find what child it is
    static inline
    TypeChildId
    childId(const dtype _) {
        return (_ & sChildMask);
    }

    //! remove the root bit from a location code
    static inline
    dtype
    remove_root(const dtype _) {
        return (_ & ~(1 << (level(_)*BPL)));
    }

    //! extract coords of the node center from the location code
    static inline
    void
    to_coords(dtype _, const TypeScale lroot, TypeCoord &x, TypeCoord &y){

        const dtype maskl = 1 << (lroot-1);
        x = maskl;
        y = maskl;

        while (_ > 1) {
            x >>= 1;
            y >>= 1;
            if (_ & sDimMaskX)      x |= maskl;
            if (_ & sDimMaskY)      y |= maskl;
            _ >>= BPL;
        }
    }
    static inline
    void
    to_coords(dtype _, const TypeScale lroot, TypeCoord &x, TypeCoord &y, TypeCoord&z){

        static constexpr dtype maskx = 1;
        static constexpr dtype masky = 2;
        static constexpr dtype maskz = 4;

        const dtype maskl = 1 << (lroot-1);
        x = maskl;
        y = maskl;
        z = maskl;

        while (_ > 1) {
            x >>= 1;
            y >>= 1;
            z >>= 1;
            if (_ & sDimMaskX)      x |= maskl;
            if (_ & sDimMaskY)      y |= maskl;
            if (_ & sDimMaskZ)      z |= maskl;
            _ >>= BPL;
        }
    }

    //! extract location code from coords of the node center
    static inline
    dtype
    from_coords(TypeCoord x, TypeCoord y, const TypeScale lroot) {

        // find the level at which this coordinate exists!
        const TypeScale tx = amm::utils::bcount_tz(x);
        const TypeScale ty = amm::utils::bcount_tz(y);

        AMM_error_invalid_arg(tx != ty,
                              "LocCode.from_coords(%u,%u) for invalid vertex (lvls = %d,%d)!", x,y,tx,ty);

        const dtype maskl = 1 << (lroot-1);
        dtype lcode = 1;

        for (TypeScale l = 0; l < lroot-tx-1; l++) {
            lcode <<= BPL;
            if (x & maskl)      lcode |= sDimMaskX;
            if (y & maskl)      lcode |= sDimMaskY;
            x <<= 1;
            y <<= 1;
        }
        return lcode;
    }

    static inline
    dtype
    from_coords(TypeCoord x, TypeCoord y, TypeCoord z, const TypeScale lroot) {

        // find the level at which this coordinate exists!
        const TypeScale tx = amm::utils::bcount_tz(x);
        const TypeScale ty = amm::utils::bcount_tz(y);
        const TypeScale tz = amm::utils::bcount_tz(z);

        AMM_error_invalid_arg(tx != ty || tx != tz,
                              "LocCode.from_coords(%u,%u,%u) for invalid vertex (lvls = %d,%d,%d)!", x,y,z,tx,ty,tz);

        const dtype maskl = 1 << (lroot-1);
        dtype lcode = 1;

        for (TypeScale l = 0; l < lroot-tx-1; l++) {
            lcode <<= BPL;
            if (x & maskl)      lcode |= sDimMaskX;
            if (y & maskl)      lcode |= sDimMaskY;
            if (z & maskl)      lcode |= sDimMaskZ;
            x <<= 1;
            y <<= 1;
            z <<= 1;
        }
        return lcode;
    }
};

}   // end of namespacce
#endif
