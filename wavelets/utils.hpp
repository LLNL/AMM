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
#ifndef WAVELETS_UTILS_H
#define WAVELETS_UTILS_H

//! ----------------------------------------------------------------------------
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>
#include <numeric>

#include "utils/logger.hpp"
#include "utils/timer.hpp"
#include "wavelets/dtypes.hpp"
#include "amm/utils/utils.hpp"
#include "creator/stencil_wcdf53.hpp"


//! ----------------------------------------------------------------------------
namespace wavelets {

//! ---------------------------------------------------------------------------------
std::array<itype, 6> bound_level(itype nx, itype ny, itype nz, int l);

/* Compute the dimensions for lifting steps */
void lifting_dims(const itype nx, const itype ny, const itype nz,   // data dimensions
                  const itype lx, const itype ly, const itype lz,   // current level for transform
                  const bool in_place,                              // in-place transform?
                  itype &mx, itype &my, itype &mz,                  // dimension at current level
                  itype &dx, itype &dy, itype &dz);                 // stride at current level


//! ---------------------------------------------------------------------------------
//! Compute the type and level of a wavelet coefficient (for in-place transform)
//! ---------------------------------------------------------------------------------

// explicit computation for a single vertex
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const ltype maxl, ltype &l);

// cached computation for complete data
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const stype sz, const ltype maxl, ltype &l);

// 2D and 3D that use 1D internally
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const itype y, const stype sz, const ltype maxl, ltype &l);
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const itype y, const itype z, const stype sz, const ltype maxl, ltype &l);

//! ---------------------------------------------------------------------------------
//! Unpack wavelet coefficients from wavelet to spatial coordinates
//! compute the type, level, and spatial location of a wavelet coefficient (for standard transform)
//!     also, sort them by subband (decreasing level)
//! ---------------------------------------------------------------------------------

/* Compute unpacking indices */
void wavelet2spatial(const itype nx, const itype ny, const ltype L,
                     std::vector<std::tuple<size_t, size_t, uint8_t, EnumWCoefficient> > &uindices);

void wavelet2spatial(const itype nx, const itype ny, const itype nz, const ltype L,
                     std::vector<std::tuple<size_t, size_t, uint8_t, EnumWCoefficient> > &uindices);


//! ---------------------------------------------------------------------------------
//! Miscellaneous
//! ---------------------------------------------------------------------------------

/* reduce resolution of a function by setting its wavelet coefficieints finer than the given level to zero */
template <typename T>
void reduce_resolution(const T* f, itype nx, itype ny, itype nz, itype l, T* g) {

    itype mx = AMM_ldim_half(nx, l);
    itype my = AMM_ldim_half(ny, l);
    itype mz = AMM_ldim_half(nz, l);

    for (itype z = 0; z < nz; ++z) {
    for (itype y = 0; y < ny; ++y) {
    for (itype x = 0; x < nx; ++x) {
        itype i = AMM_xyz2idx(x,y,z,nx,ny,nz);
        g[i] = (z<mz && y<my && x<mx) ? f[i] : 0;
    }}}
}

/* Convolve two 1D signals. The size of the output signal should be nf+ng-1. */
template <typename T>
void convolve(const T* RESTRICT f, int nf, const T* RESTRICT g, int ng, OUT T* RESTRICT y) {
    int ny = nf + ng - 1;
    // y[n] = \sum f[k] * g[n-k]
    PARALLEL_LOOP
    for (int n = 0; n < ny; ++n) {
        y[n] = 0;
        for (int k = 0; k < nf; ++k) {
            if (n>=k && n<ng+k) {
                y[n] += f[k] * g[n-k];
            }
        }
    }
}

/* Convenient wrapper around the above convolve. */
template <typename T>
std::vector<T> convolve(const T* f, int nf, const T* g, int ng) {
    int ny = nf + ng - 1;
    std::vector<T> y(ny);
    convolve(f, nf, g, ng, y.data());
    return y;
}

/* For a 1D signal, insert a zero (odd) sample between every two adjacent even samples.
The size of the output signal should be nf*2-1. */
template <typename T>
void upsample_zeros(const T* RESTRICT f, int nf, OUT T* RESTRICT g) {
    int ng = nf*2 - 1;
    for (int i = 0; i < ng; i += 2) {
        g[i] = f[i/2];
        if (i+1 < ng) {
            g[i+1] = 0;
        }
    }
}

/* Convenient wrapper around the above upsample_zeros. */
template <typename T>
std::vector<T> upsample_zeros(const T* f, int nf) {
    int ng = nf*2 - 1;
    std::vector<T> g(ng);
    upsample_zeros(f, nf, g.data());
    return g;
}


//! ---------------------------------------------------------------------------------
//! Filter wavelets whose stencils do not intersect with the boundary
//! ---------------------------------------------------------------------------------
template <typename T>
void filter_external_wavelets(std::vector<T> &data,
                              const size_t m_inDims[3], const size_t m_dims[3],
                              const uint8_t &wmaxlevels, const EnumWavelet &wtype) {

    amm::timer t;
    AMM_log_info << "Filtering external coefficients...";
    fflush(stdout);

    // filter out external wavelets by replacing them with zero!
    //static constexpr T EXTERNAL_VAL = std::numeric_limits<T>::signaling_NaN();
    static constexpr T EXTERNAL_VAL = T(0);
    static const TypeDim nstypes = (m_dims[2] == 1) ? 4 : 8;

    std::vector<size_t> nfiltered(nstypes, TypeDim(0));

    //TODO: use openmp parallelism for these loops
    if (1 == m_dims[2]) {

        // create an instance of 2D stencils that we will query
        using TypeStencil = typename amm::stencil_cdf53<2>;
        using TypeOffset  = typename TypeStencil::St::TypeOffset;

        auto tcreator = TypeStencil::init(wtype);

        // for each type of stencil, use its span (radius) to determine which stencils won't intersect with the domain
        for(uint8_t stype = 0; stype < nstypes; stype++) {

            const bool wavelet_x = (stype & 1);
            const bool wavelet_y = (stype & 2);
            const TypeOffset &sspan = tcreator[stype].span();   // use span here to allow flexbility in wavelet type
            //std::cout << "\n\n stype = " << int(stype) << " : " << wavelet_x << ", " << wavelet_y << ": span = " << sspan <<"\n";

            const TypeScale wminlevels = (stype == 0) ? wmaxlevels : 1;
            for(TypeScale wlvl = wmaxlevels; wlvl >= wminlevels; --wlvl) {

                //std::cout << "\t wlvl = " << int(wlvl) << "\n";

                // for this level, find the stride, spacing and offset of coefficients
                const TypeIndex stride  = AMM_pow2(wlvl);   // stride = strides between adjacent coefficients of same type
                const TypeIndex spacing = stride >> 1;      // spacing = spacing between adjacent points within a stencil

                const TypeIndex &offset_y = wavelet_y ? spacing : 0;
                const TypeIndex &offset_x = wavelet_x ? spacing : 0;
                //std::cout << "\t : stride = " << stride << ", offsets = " << offset_x << ", " << offset_y << ", spacing =" << spacing << "\n";

                // find the location of the first stencil that is outside the data domain
                // i.e., find (offset + k*stride), such that (offset + k*stride) > (data_dims - spacing)
                const TypeIndex minx = offset_x + stride*(1+(m_inDims[0]+spacing*sspan[0]-offset_x)/stride);
                const TypeIndex miny = offset_y + stride*(1+(m_inDims[1]+spacing*sspan[1]-offset_y)/stride);
                //std::cout << "\t : minx = " << minx << ", miny = " << miny << "\n";

                for(TypeIndex y = offset_y; y < m_dims[1]; y+=stride){
                for(TypeIndex x = offset_x; x < m_dims[0]; x+=stride){

                    // if this stencil does not affect the mesh
                    if (x >= minx || y >= miny) {
                        //std::cout << "\t --> " << int(x) << ", " << int(y) << "\n";
                        const TypeIndex idx = y*m_dims[0] + x;
                        data[idx] = EXTERNAL_VAL;
                        nfiltered[stype] ++;
                    }
                }}
            }
        }
    }

    else {
        // create an instance of 3D stencils that we will query
        using TypeStencil = typename amm::stencil_cdf53<3>;
        using TypeOffset  = typename TypeStencil::St::TypeOffset;

        auto tcreator = TypeStencil::init(wtype);

        // for each type of stencil, use it span (radius) to determine which stencils won't intersect with the domain
        for(uint8_t stype = 0; stype < nstypes; stype++) {

            bool wavelet_x = (stype & 1);
            bool wavelet_y = (stype & 2);
            bool wavelet_z = (stype & 4);
            const TypeOffset &sspan = tcreator[stype].span();
            //std::cout << "\n\n stype = " << int(stype) << " : " << wavelet_x << ", " << wavelet_y << ": span = " << sspan <<"\n";

            const TypeScale wminlevels = (stype == 0) ? wmaxlevels : 1;
            for(TypeScale wlvl = wmaxlevels; wlvl >= wminlevels; --wlvl) {

                //std::cout << "\t wlvl = " << int(wlvl) << "\n";

                // for this level, find the stride, spacing and offset of coefficients
                const TypeIndex stride  = AMM_pow2(wlvl);   // stride = strides between adjacent coefficients of same type
                const TypeIndex spacing = stride >> 1;      // spacing = spacing between adjacent points within a stencil

                const TypeIndex &offset_z = wavelet_z ? spacing : 0;
                const TypeIndex &offset_y = wavelet_y ? spacing : 0;
                const TypeIndex &offset_x = wavelet_x ? spacing : 0;
                //std::cout << "\t : stride = " << stride << ", offsets = " << offset_x << ", " << offset_y << ", spacing =" << spacing << "\n";

                // find the location of the first stencil that is outside the data domain
                // i.e., find (offset + k*stride), such that (offset + k*stride) > (data_dims - spacing)
                const TypeIndex minx = offset_x + stride*(1+(m_inDims[0]+spacing*sspan[0]-offset_x)/stride);
                const TypeIndex miny = offset_y + stride*(1+(m_inDims[1]+spacing*sspan[1]-offset_y)/stride);
                const TypeIndex minz = offset_z + stride*(1+(m_inDims[2]+spacing*sspan[2]-offset_z)/stride);
                //std::cout << "\t : minx = " << minx << ", miny = " << miny << ", minz = " << minz << "\n";

                for(TypeIndex z = offset_z; z < m_dims[2]; z+=stride){
                for(TypeIndex y = offset_y; y < m_dims[1]; y+=stride){
                for(TypeIndex x = offset_x; x < m_dims[0]; x+=stride){

                    // if this stencil does not affect the mesh
                    if (x >= minx || y >= miny || z >= minz) {
                        //std::cout << "\t --> " << int(x) << ", " << int(y) << "\n";
                        const TypeIndex idx = z*m_dims[0]*m_dims[1]+y*m_dims[0]+x;
                        data[idx] = EXTERNAL_VAL;
                        nfiltered[stype] ++;
                    }
                }}}
            }
        }
    }

    const size_t ntotal_coefficients = m_dims[0]*m_dims[1]*m_dims[2];
    const size_t ntotal_filtered = std::accumulate(nfiltered.begin(), nfiltered.end(), size_t(0));

    AMM_log_info << " done!" << t << " filtered (";
    for(TypeDim k = 0; k < nstypes; k++)    AMM_log_info << nfiltered[k] << " ";
    AMM_log_info << ") stencils! total " << ntotal_filtered << " out of " << (ntotal_coefficients)
                     << " (= " << 100.0*double(ntotal_filtered)/double(ntotal_coefficients) << " %)\n";
}

//! ---------------------------------------------------------------------------------

}   // end of namespace
#endif
