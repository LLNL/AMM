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

#include <array>
#include <bitset>
#include <algorithm>

#include "types/dtypes.hpp"
#include "amm/utils/utils.hpp"
#include "amm/utils/exceptions.hpp"
#include "wavelets/utils.hpp"

// BE CAREFUL to use this macro with data dimensions (nx, ny, nz) available
#define IDX(x,y,z)              AMM_xyz2idx(x,y,z,nx,ny,nz)

namespace wavelets {

//! ----------------------------------------------------------------------------
//! wavelet transform requires computation of dimensions for each data
//! depending upon whether we want normal (coefficients stored in wavelet domain)
//! or
//! in-place (coefficients stored in spatial domain).
//! these dimensions are different
//! ----------------------------------------------------------------------------
void lifting_dims(const itype nx, const itype ny, const itype nz,   // data dimensions
                  const itype lx, const itype ly, const itype lz,   // current level for transform
                  const bool in_place,                              // in-place transform?
                  itype &mx, itype &my, itype &mz,                  // dimension at current level
                  itype &dx, itype &dy, itype &dz) {                // stride at current level

    if (in_place) {
        mx = nx;            my = ny;            mz = nz;
        dx = AMM_pow2(lx);  dy = AMM_pow2(ly);  dz = AMM_pow2(lz);
    }
    else {
        dx = 1;                     dy = 1;                     dz = 1;
        mx = AMM_ldim_half(nx, lx); my = AMM_ldim_half(ny, ly); mz = AMM_ldim_half(nz, lz);
    }
}

std::array<itype, 6> bound_level(itype nx, itype ny, itype nz, int l) {
    for (int i = 0; i < l; ++i) {
        nx = ((nx >> 1) << 1) + 1;
        ny = ((ny >> 1) << 1) + 1;
        nz = ((nz >> 1) << 1) + 1;
        nx = (nx + 1) >> 1;
        ny = (ny + 1) >> 1;
        nz = (nz + 1) >> 1;
    }
    itype mx = nx, my = ny, mz = nz;
    nx = ((nx >> 1) << 1) + 1;
    ny = ((ny >> 1) << 1) + 1;
    nz = ((nz >> 1) << 1) + 1;
    return std::array<itype, 6>{ mx, my, mz, nx, ny, nz };
}


//! ----------------------------------------------------------------------------
//! Compute the type and level of a wavelet coefficient (for in-place transform)
//! ----------------------------------------------------------------------------
// explicit computation for a single vertex
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const ltype maxl, ltype &l) {

   // a scaling coefficient at level l exists at           k.2^l
   // a wavelet coefficient at level l exists at 2^(l-1) + k.2^l
   for(l = maxl; l > 0; l--) {

       itype d = AMM_pow2(l);        // stride at level l
       itype r = x & (d-1);      // remainder!

       if (r == 0)     return EnumWCoefficient::S;
       if (2*r == d)   return EnumWCoefficient::W;
   }

   AMM_error_logic(true, "failed to detect vertex type!");
}

// cached computation for a single vertex
EnumWCoefficient get_wcoeff_lvl_type(const itype x, const stype sz, const ltype maxl, ltype &l) {

    using mtype = uint8_t;  // use a bitmap of 8 bits to store the required map

    static std::vector<mtype> ltmap;
    static ltype smaxl = 0;

    const bool is_invalid = (smaxl > 0 && smaxl != maxl) || (ltmap.size() > 0 && ltmap.size() != sz);

    AMM_error_invalid_arg(is_invalid,
                          "vertex -> (level, type) map was initialized with sz = %lu and maxl = %d\nnow, you are trying to use this for sz = %lu and maxl = %d\n",
                          ltmap.size(), smaxl, sz, maxl);

    // initialize the map
    if (ltmap.empty()) {

        // a scaling coefficient at level l exists at           k.2^l
        // a wavelet coefficient at level l exists at 2^(l-1) + k.2^l

        // initialize the map with verything at the scaling coefficient
        // and then, overwrite the values for successive levels

        uint8_t sval = maxl << 1;  // initial value (for scaling vertices at highest level)
        smaxl = maxl;
        ltmap.resize(sz, sval);

        for(ltype ll = maxl; ll > 0; ll--) {

            const itype d = AMM_pow2(ll);           // stride at level l
            const itype o = AMM_pow2(ll-1);         // offset at level l
            const itype v = (ll << 1) + 1;      // the bitmap has last bit for wavelets (always 1)

            for(itype lx = o; lx < sz; lx+=d) {
                ltmap[lx] = v;
            }
        }
    }

    l = ltmap[x] >> 1;
    return EnumWCoefficient(ltmap[x] & 1);
}

EnumWCoefficient get_wcoeff_lvl_type(const itype x, const itype y,
                                     const stype sz, const ltype maxl, ltype &l) {

   ltype lx,ly;
   EnumWCoefficient tx = get_wcoeff_lvl_type(x,sz,maxl,lx);
   EnumWCoefficient ty = get_wcoeff_lvl_type(y,sz,maxl,ly);

   l = std::min(lx,ly);

   // if at least one of the two is a scaling coefficient,
   // the final coefficient will be have a scaling piece in it
   // remember, scaling coefficients exist only at the highest level
   EnumWCoefficient r;
   r = (ty == EnumWCoefficient::S && tx == EnumWCoefficient::S) ? EnumWCoefficient::SS :
       (ty == EnumWCoefficient::S && tx == EnumWCoefficient::W) ? EnumWCoefficient::SW :
       (ty == EnumWCoefficient::W && tx == EnumWCoefficient::S) ? EnumWCoefficient::WS :

       // if both are wavelet coefficients, then i treat them as scaling for the higher level
       (ly > lx) ? EnumWCoefficient::SW :
       (ly < lx) ? EnumWCoefficient::WS :
                   EnumWCoefficient::WW;
   return r;
}

EnumWCoefficient get_wcoeff_lvl_type(const itype x, const itype y, const itype z,
                                     const stype sz, const ltype maxl, ltype &l) {

    ltype lx,ly,lz;
    EnumWCoefficient tx = get_wcoeff_lvl_type(x,sz,maxl,lx);
    EnumWCoefficient ty = get_wcoeff_lvl_type(y,sz,maxl,ly);
    EnumWCoefficient tz = get_wcoeff_lvl_type(z,sz,maxl,lz);

    l = std::min({lx,ly,lz});
    EnumWCoefficient t = static_cast<EnumWCoefficient>((static_cast<int>(tz) << 2) + (static_cast<int>(ty) << 1) + static_cast<int>(tx));

    // if at most one coefficient is a wavelet coefficient
    EnumWCoefficient r;
    r = (t == EnumWCoefficient::SSS || t == EnumWCoefficient::SSW || t == EnumWCoefficient::SWS || t == EnumWCoefficient::WSS) ? t :    // at most one wavelet
        (lx == ly && lx == lz) ? t :                                                                    // if all are on same level
        (lx == ly && lx != lz) ? ((l==lx) ? EnumWCoefficient::SWW : EnumWCoefficient::WSS) :     // 3 or 4
        (ly == lz && ly != lx) ? ((l==ly) ? EnumWCoefficient::WWS : EnumWCoefficient::SSW) :     // 6 or 1
        (lz == lx && lz != ly) ? ((l==lx) ? EnumWCoefficient::WSW : EnumWCoefficient::SWS) :     // 5 or 2
        (l == lx) ? EnumWCoefficient::SSW :                                              // 1
        (l == ly) ? EnumWCoefficient::SWS :                                              // 2
        (l == lz) ? EnumWCoefficient::WSS :                                              // 4
                    EnumWCoefficient::None;

    if (EnumWCoefficient::None == r) {
        printf(" get_wcoeff_lvl_type() failed to detect vertex type!\n");
        exit(1);
    }
    return r;
}

//! ---------------------------------------------------------------------------------
//! Unpack wavelet coefficients from wavelet to spatial coordinates
//! compute the type, level, and spatial location of a wavelet coefficient (for standard transform)
//!     also, sort them by subband (decreasing level)
//! ---------------------------------------------------------------------------------
void wavelet2spatial(const itype nx, const itype ny, const ltype L,
                     std::vector<std::tuple<size_t, size_t, uint8_t, EnumWCoefficient>> &uindices) {

    uint32_t nz = 1;
    uindices.reserve(nx*ny*nz);

    // scaling
    {
    uint32_t sx = AMM_ldim_half(nx, L);
    uint32_t sy = AMM_ldim_half(ny, L);

    uint32_t px = 1<<L;
    uint32_t py = 1<<L;

    for(uint32_t y = 0; y < sy; y++) {
    for(uint32_t x = 0; x < sx; x++) {
        size_t widx = IDX(x,y,0);
        size_t uidx = IDX(x*px,y*py,0);                             // unpack
        uindices.push_back(std::make_tuple(widx, uidx, L, EnumWCoefficient::SS));
    }
    }
    }

    for(uint8_t l = L; l > 0; l--) {

        uint32_t px = 1<<l;
        uint32_t py = 1<<l;

        uint32_t sx = AMM_ldim_half(nx, l);
        uint32_t wx = AMM_ldim_half(nx, l-1);

        uint32_t sy = AMM_ldim_half(ny, l);
        uint32_t wy = AMM_ldim_half(ny, l-1);

        //printf(LOG_PRNT_STD, " l = %d :: [%d %d]\n", l, sx, wx);

        // dy
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = 0; x < sx; x++) {
            size_t widx = IDX(x,y,0);
            size_t uidx = IDX(x*px,(y-sy)*py+py/2,0);               // unpack
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WS));
        }
        }

        // dx
        for(uint32_t y = 0; y < sy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x,y,0);
            size_t uidx = IDX((x-sx)*px+px/2,y*py,0);               // unpack
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::SW));
        }
        }

        // dxdy
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x,y,0);
            size_t uidx = IDX((x-sx)*px+px/2,(y-sy)*py+py/2,0);     // unpack
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WW));
        }
        }
    }
}

void wavelet2spatial(const itype nx, const itype ny, const itype nz, const ltype L,
                     std::vector<std::tuple<size_t, size_t, uint8_t, EnumWCoefficient>> &uindices) {

    uindices.reserve(nx*ny*nz);

    // scaling
    {
    uint32_t sx = AMM_ldim_half(nx, L);
    uint32_t sy = AMM_ldim_half(ny, L);
    uint32_t sz = AMM_ldim_half(nz, L);

    uint32_t px = 1<<L;
    uint32_t py = 1<<L;
    uint32_t pz = 1<<L;

    for(uint32_t z = 0; z < sz; z++) {
    for(uint32_t y = 0; y < sy; y++) {
    for(uint32_t x = 0; x < sx; x++) {
        size_t widx = IDX(x,y,z);
        size_t uidx = IDX(x*px, y*py, z*pz);
        uindices.push_back(std::make_tuple(widx, uidx, L, EnumWCoefficient::SSS));
    }
    }
    }

    }

    // wavelet at all other levels
    for(uint8_t l = L; l > 0; l--) {

        uint32_t px = 1<<l;
        uint32_t py = 1<<l;
        uint32_t pz = 1<<l;

        uint32_t sx = AMM_ldim_half(nx, l);
        uint32_t wx = AMM_ldim_half(nx, l-1);

        uint32_t sy = AMM_ldim_half(ny, l);
        uint32_t wy = AMM_ldim_half(ny, l-1);

        uint32_t sz = AMM_ldim_half(nz, l);
        uint32_t wz = AMM_ldim_half(nz, l-1);

        // dz
        for(uint32_t z = sz; z < wz; z++) {
        for(uint32_t y = 0; y < sy; y++) {
        for(uint32_t x = 0; x < sx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX(x*px, y*py, (z-sz)*pz+pz/2);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WSS));
        }
        }
        }

        // dy
        for(uint32_t z = 0; z < sz; z++) {
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = 0; x < sx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX(x*px, (y-sy)*py+py/2, z*pz);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::SWS));
        }
        }
        }

        // dx
        for(uint32_t z = 0; z < sz; z++) {
        for(uint32_t y = 0; y < sy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x,y,z);
            size_t uidx = IDX((x-sx)*px+px/2, y*py, z*pz);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::SSW));
        }
        }
        }

        // dxdy
        for(uint32_t z = 0; z < sz; z++) {
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX((x-sx)*px+px/2, (y-sy)*py+py/2, z*pz);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::SWW));
        }
        }
        }

        // dydz
        for(uint32_t z = sz; z < wz; z++) {
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = 0; x < sx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX(x*px, (y-sy)*py+py/2, (z-sz)*pz+pz/2);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WWS));
        }
        }
        }

        // dxdz
        for(uint32_t z = sz; z < wz; z++) {
        for(uint32_t y = 0; y < sy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX((x-sx)*px+px/2, y*py, (z-sz)*pz+pz/2);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WSW));
        }
        }
        }

        // dxdydz
        for(uint32_t z = sz; z < wz; z++) {
        for(uint32_t y = sy; y < wy; y++) {
        for(uint32_t x = sx; x < wx; x++) {
            size_t widx = IDX(x, y, z);
            size_t uidx = IDX((x-sx)*px+px/2, (y-sy)*py+py/2, (z-sz)*pz+pz/2);
            uindices.push_back(std::make_tuple(widx, uidx, l, EnumWCoefficient::WWW));
        }
        }
        }

    }
}

//! ---------------------------------------------------------------------------------
}


