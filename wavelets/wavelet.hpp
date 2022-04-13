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
#ifndef WAVELET_H
#define WAVELET_H

#include <cstdlib>
#include <vector>

#include "wavelets/dtypes.hpp"

/// --------------------------------------------------------------------------------
//! Abstract base class for wavelet computation
/// --------------------------------------------------------------------------------
namespace wavelets {

template <typename dtype>
class Wavelets {

protected:
    const ltype mMaxLevels;         // Max levels of wavelet transform supported
    const bool mNormalize;          // normalized or not
    const bool mInPlace;            // transform the values in place
                                        // i.e., wavelet coefficients are still stored in spatial domain

protected:
    // any instantiation of wavelets needs to define the forward and inverse lift
    virtual bool flift_x(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;
    virtual bool flift_y(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;
    virtual bool flift_z(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;

    virtual bool ilift_x(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;
    virtual bool ilift_y(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;
    virtual bool ilift_z(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) = 0;

public:

    Wavelets(const ltype &maxLevels, const bool &normalize, const bool &inplace) :
             mMaxLevels(maxLevels), mNormalize(normalize), mInPlace(inplace) {}

    /* forward wavelet transform */
    bool forward_transform(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype levels = std::max(lx, std::max(ly, lz));
        itype l;

        if (levels > mMaxLevels)
            return false;

        for (l = 0; l < levels; l++) {
            if (l < lx && !flift_x(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
            if (l < ly && !flift_y(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
            if (l < lz && !flift_z(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
        }
        return true;
    }

    /* inverse wavelet transform */
    bool inverse_transform(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype levels = std::max(lx, std::max(ly, lz));
        itype l;

        if (levels > mMaxLevels)
            return false;

        for (l = levels; l--;) {
            if (l < lz && !ilift_z(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
            if (l < ly && !ilift_y(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
            if (l < lx && !ilift_x(f, nx, ny, nz, std::min(l, lx), std::min(l, ly), std::min(l, lz)))
                return false;
        }
        return true;
    }

    /* convenient wrappers for 1D and 2D functions */
    inline bool forward_transform(dtype* f, itype nx, itype ny, itype lx, itype ly) {
        return forward_transform(f, nx, ny, 1, lx, ly, 0);
    }
    inline bool forward_transform(dtype* f, itype nx, itype lx) {
        return forward_transform(f, nx, 1, 1, lx, 0, 0);
    }
    inline bool inverse_transform(dtype* f, itype nx, itype ny, itype lx, itype ly) {
        return inverse_transform(f, nx, ny, 1, lx, ly, 0);
    }
    inline bool inverse_transform(dtype* f, itype nx, itype lx) {
        return inverse_transform(f, nx, 1, 1, lx, 0, 0);
    }
};
}
#endif
