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
#ifndef CDF53_WAVELETS_H
#define CDF53_WAVELETS_H

//! ----------------------------------------------------------------------------
#include <array>
#include <vector>
#include <cassert>

#include "wavelets/dtypes.hpp"
#include "wavelets/wavelet.hpp"
#include "wavelets/utils.hpp"
#include "amm/utils/utils.hpp"

// BE CAREFUL to use this macro with data (f) and dimensions (nx, ny, nz) available
#define FUNC(x,y,z)             f[AMM_xyz2idx(x,y,z,nx,ny,nz)]

//! ----------------------------------------------------------------------------
namespace wavelets {

//! ----------------------------------------------------------------------------
//! CDF53 / Approximating Linear B-Spline
//! ----------------------------------------------------------------------------
template <typename dtype>
class Wavelets_CDF53 : public Wavelets<dtype> {

private:
    const EnumWavelet mType;                                // wavelet type (Approximating or Interpolating)
    std::vector<double> wavelet_weights, scaling_weights;   // weights of scaling and wavelet functions
    std::vector<double> wavelet_norms, scaling_norms;       // norms and lengths of the same at various levels

    // ----------------------------------------------------------------------------------
    /* Compute the squared norms of the CDF5/3 scaling and wavelet functions.
    mMaxLevels is the number of times the wavelet transform is done (it is the number of levels
    minus one).*/
    void init() {

        wavelet_norms.resize(this->mMaxLevels+1);
        scaling_norms.resize(this->mMaxLevels+1);

        std::vector<double> scaling_lengths(this->mMaxLevels+1), wavelet_lengths(this->mMaxLevels+1);

        // for each level, compute these
        std::vector<double> scaling_weights_at_l = this->scaling_weights;
        std::vector<double> scaling_func_at_l    = this->scaling_weights;
        std::vector<double> wavelet_weights_at_l = this->wavelet_weights;
        std::vector<double> wavelet_func_at_l    = this->wavelet_weights;

        /* compute the squared norms of each scaling and wavelet function */
        for (ltype l = 0; l < this->mMaxLevels+1; ++l) {

            scaling_norms[l] = amm::utils::l2norm(scaling_func_at_l.data(), scaling_func_at_l.size());
            wavelet_norms[l] = amm::utils::l2norm(wavelet_func_at_l.data(), wavelet_func_at_l.size());

            scaling_lengths[l] = static_cast<int>(scaling_func_at_l.size());
            wavelet_lengths[l] = static_cast<int>(wavelet_func_at_l.size());

            // upsample the wavelet weights
            wavelet_weights_at_l = upsample_zeros(wavelet_weights_at_l.data(), static_cast<int>(wavelet_weights_at_l.size()));

            // compute the new wavelet function
            wavelet_func_at_l = convolve(wavelet_weights_at_l.data(), static_cast<int>(wavelet_weights_at_l.size()), scaling_func_at_l.data(), static_cast<int>(scaling_func_at_l.size()));

            // upsample the scaling weights
            scaling_weights_at_l = upsample_zeros(scaling_weights_at_l.data(), static_cast<int>(scaling_weights_at_l.size()));

            // compute the new scaling function
            scaling_func_at_l = convolve(scaling_weights_at_l.data(), static_cast<int>(scaling_weights_at_l.size()), scaling_func_at_l.data(), static_cast<int>(scaling_func_at_l.size()));
        }
    }

    // ----------------------------------------------------------------------------------
    /* Forward lifting in X Y and Z */
    bool flift_x(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (mx <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[lx] / (lx ? scaling_norms[lx - 1] : 1);
            w = wavelet_norms[lx] / (lx ? scaling_norms[lx - 1] : 1);
        }

        stype dx2 = 2*dx;
        stype x, y, z;

        /* w-lift */
        for (z = 0;  z < mz; z+=dz)
        for (y = 0;  y < my; y+=dy)
        for (x = dx; x < mx; x+=dx2) {
            stype xmin = x - dx;
            stype xmax = (x == mx - dx ? x - dx : x + dx);
            FUNC(x,y,z) -= (FUNC(xmin,y,z) + FUNC(xmax,y,z)) / 2;
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (z = 0;  z < mz; z+=dz)
        for (y = 0;  y < my; y+=dy)
        for (x = dx; x < mx; x+=dx2) {
            stype xmin = x - dx;
            stype xmax = (x == mx - dx ? x - dx : x + dx);
            FUNC(xmin,y,z) += FUNC(x,y,z) / 4;
            FUNC(xmax,y,z) += FUNC(x,y,z) / 4;
        }
        }

        // nothing more to be done for inplace transform if not normalize
        if (this->mInPlace && !this->mNormalize)
            return true;

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {

            for (z = 0; z < mz; z+=dz)
            for (y = 0; y < my; y+=dy)
            for (x = 0; x < mx; x+=dx)
                FUNC(x,y,z) = FUNC(x,y,z) * (x & dx ? w : s);
        }

        else {
            std::vector<dtype> t (mx);
            for (z = 0; z < mz; z++)
            for (y = 0; y < my; y++) {

                for (x = 0; x < mx; x++){
                    t[((x & 1 ? mx : 0) + x) / 2] = FUNC(x,y,z) * (x & 1 ? w : s);
                }

                for (x = 0; x < mx; x++)
                    FUNC(x,y,z) = t[x];
            }
        }

        return true;
    }

    bool flift_y(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (my <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[ly] / (ly ? scaling_norms[ly - 1] : 1);
            w = wavelet_norms[ly] / (ly ? scaling_norms[ly - 1] : 1);
        }

        stype dy2 = 2*dy;
        stype x, y, z;

        /* w-lift */
        for (z = 0;  z < mz; z+=dz)
        for (x = 0;  x < mx; x+=dx)
        for (y = dy; y < my; y+=dy2) {
            stype ymin = y - dy;
            stype ymax = (y == my - dy ? y - dy : y + dy);
            FUNC(x,y,z) -= ( FUNC(x,ymin,z) + FUNC(x,ymax,z) ) / 2;
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (z = 0;  z < mz; z+=dz)
        for (x = 0;  x < mx; x+=dx)
        for (y = dy; y < my; y+=dy2) {
            stype ymin = y - dy;
            stype ymax = (y == my - dy ? y - dy : y + dy);
            FUNC(x,ymin,z) += FUNC(x,y,z) / 4;
            FUNC(x,ymax,z) += FUNC(x,y,z) / 4;
        }
        }

        // nothing more to be done if not this->normalize and inplace
        if (!this->mNormalize && this->mInPlace){
            return true;
        }

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {

            for (z = 0; z < mz; z+=dz)
            for (x = 0; x < mx; x+=dx)
            for (y = 0; y < my; y+=dy)
                FUNC(x,y,z) = FUNC(x,y,z) * (y & dy ? w : s);
        }

        else {

            std::vector<dtype> t (my);
            for (z = 0; z < mz; z++)
            for (x = 0; x < mx; x++) {

                for (y = 0; y < my; y++){

                    t[((y & 1 ? my : 0) + y) / 2] = FUNC(x,y,z) * (y & 1 ? w : s);
                }

                for (y = 0; y < my; y++)
                    FUNC(x,y,z) = t[y];
            }
        }

        return true;
    }

    bool flift_z(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (mz <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[lz] / (lz ? scaling_norms[lz - 1] : 1);
            w = wavelet_norms[lz] / (lz ? scaling_norms[lz - 1] : 1);
        }

        stype dz2 = 2*dz;
        stype x, y, z;

        /* w-lift */
        for (y = 0;  y < my; y+=dy)
        for (x = 0;  x < mx; x+=dx)
        for (z = dz; z < mz; z+=dz2) {
            stype zmin = z - dz;
            stype zmax = (z == mz - dz ? z - dz : z + dz);
            FUNC(x,y,z) -= (FUNC(x,y,zmin) + FUNC(x,y,zmax)) / 2;
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (y = 0;  y < my; y+=dy)
        for (x = 0;  x < mx; x+=dx)
        for (z = dz; z < mz; z+=dz2) {
            stype zmin = z - dz;
            stype zmax = (z == mz - dz ? z - dz : z + dz);
            FUNC(x,y,zmin) += FUNC(x,y,z) / 4;
            FUNC(x,y,zmax) += FUNC(x,y,z) / 4;
        }
        }

        // nothing more to be done if not this->normalize and inplace
        if (!this->mNormalize && this->mInPlace)
            return true;

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {

            for (y = 0; y < my; y+=dy)
            for (x = 0; x < mx; x+=dz)
            for (z = 0; z < mz; z+=dz)
                FUNC(x,y,z) = FUNC(x,y,z) * (z & dz ? w : s);
        }

        else {

            std::vector<dtype> t (mz);
            for (y = 0; y < my; y++)
            for (x = 0; x < mx; x++) {

                for (z = 0; z < mz; z++)
                    t[((z & 1 ? mz : 0) + z) / 2] = FUNC(x,y,z) * (z & 1 ? w : s);

                for (z = 0; z < mz; z++)
                    FUNC(x,y,z) = t[z];
            }
        }
        return true;
    }

    // ----------------------------------------------------------------------------------
    /* Inverse lifting in X Y and Z */
    bool ilift_x(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (mx <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[lx] / (lx ? scaling_norms[lx - 1] : 1);
            w = wavelet_norms[lx] / (lx ? scaling_norms[lx - 1] : 1);
        }

        stype dx2 = 2*dx;
        stype x, y, z;

        // handle odd number of values
        stype oddm = (mx&1);

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {
            for (z = 0; z < mz; z+=dz)
            for (y = 0; y < my; y+=dy)
            for (x = 0; x < mx; x+=dx)
                FUNC(x,y,z) = FUNC(x,y,z) / (x & dx ? w : s);
        }

        else if (!this->mInPlace) {

            std::vector<dtype> t (mx);
            for (z = 0; z < mz; z++)
            for (y = 0; y < my; y++) {
                for (x = 0; x < mx; x++)
                    t[2 * x - (2 * x < mx+oddm ? 0 : mx - 1 + oddm)] = FUNC(x,y,z) / (2 * x < mx ? s : w);

                for (x = 0; x < mx; x++)
                    FUNC(x,y,z) = t[x];
            }
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (z = 0;  z < mz; z+=dz)
        for (y = 0;  y < my; y+=dy)
        for (x = dx; x < mx; x+=dx2) {
            stype xmin = x - dx;
            stype xmax = (x == mx - dx ? x - dx : x + dx);
            FUNC(xmin,y,z) -= FUNC(x,y,z) / 4;
            FUNC(xmax,y,z) -= FUNC(x,y,z) / 4;
        }
        }

        /* w-lift */
        for (z = 0;  z < mz; z+=dz)
        for (y = 0;  y < my; y+=dy)
        for (x = dx; x < mx; x+=dx2) {
            stype xmin = x - dx;
            stype xmax = (x == mx - dx ? x - dx : x + dx);
            FUNC(x,y,z) += (FUNC(xmin,y,z)+FUNC(xmax,y,z)) / 2;
        }

        return true;
    }

    bool ilift_y(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (my <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[ly] / (ly ? scaling_norms[ly - 1] : 1);
            w = wavelet_norms[ly] / (ly ? scaling_norms[ly - 1] : 1);
        }

        stype dy2 = 2*dy;
        stype x, y, z;

        // handle odd number of values
        stype oddm = (my&1);

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {
            for (z = 0; z < mz; z+=dz)
            for (x = 0; x < mx; x+=dx)
            for (y = 0; y < my; y+=dy)
                FUNC(x,y,z) = FUNC(x,y,z) / (y & dy ? w : s);
        }

        else if (!this->mInPlace) {

            std::vector<dtype> t (my);
            for (z = 0; z < mz; z++)
            for (x = 0; x < mx; x++) {
                for (y = 0; y < my; y++)
                    t[2 * y - (2 * y < my+oddm ? 0 : my - 1 + oddm)] = FUNC(x,y,z) / (2 * y < my ? s : w);

                for (y = 0; y < my; y++)
                    FUNC(x,y,z) = t[y];
            }
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (z = 0;  z < mz; z+=dz)
        for (x = 0;  x < mx; x+=dx)
        for (y = dy; y < my; y+=dy2) {
            stype ymin = y - dy;
            stype ymax = (y == my - dy ? y - dy : y + dy);
            FUNC(x,ymin,z) -= FUNC(x,y,z) / 4;
            FUNC(x,ymax,z) -= FUNC(x,y,z) / 4;
        }
        }

        /* w-lift */
        for (z = 0;  z < mz; z+=dz)
        for (x = 0;  x < mx; x+=dx)
        for (y = dy; y < my; y+=dy2) {
            stype ymin = y - dy;
            stype ymax = (y == my - dy ? y - dy : y + dy);
            FUNC(x,y,z) += (FUNC(x,ymin,z)+FUNC(x,ymax,z)) / 2;
        }

        return true;
    }

    bool ilift_z(dtype* f, itype nx, itype ny, itype nz, itype lx, itype ly, itype lz) {

        itype mx,my,mz,dx,dy,dz;
        lifting_dims(nx,ny,nz,lx,ly,lz,this->mInPlace,mx,my,mz,dx,dy,dz);
        if (mz <= 1)
            return false;

        double s = 1, w = 1;
        if (this->mNormalize) {
            s = scaling_norms[lz] / (lz ? scaling_norms[lz - 1] : 1);
            w = wavelet_norms[lz] / (lz ? scaling_norms[lz - 1] : 1);
        }

        stype dz2 = 2*dz;
        stype x, y, z;

        // to handle odd number of values
        stype oddm = (mz&1);

        /* scale and reorder */
        if (this->mInPlace && this->mNormalize) {
            for (y = 0; y < my; y+=dx)
            for (x = 0; x < mx; x+=dy)
            for (z = 0; z < mz; z+=dz)
                FUNC(x,y,z) = FUNC(x,y,z) / (z & dz ? w : s);
        }
        else if (!this->mInPlace) {

            std::vector<dtype> t(mz);
            for (y = 0; y < my; y++)
            for (x = 0; x < mx; x++) {

                for (z = 0; z < mz; z++)
                    t[2 * z - (2 * z < mz+oddm ? 0 : mz - 1 + oddm)] = FUNC(x,y,z) / (2 * z < mz ? s : w);

                for (z = 0; z < mz; z++)
                    FUNC(x,y,z) = t[z];
            }
        }

        /* s-lift */
        if (this->mType == EnumWavelet::Approximating) {
        for (y = 0;  y < my; y+=dy)
        for (x = 0;  x < mx; x+=dx)
        for (z = dz; z < mz; z+=dz2) {
            stype zmin = z - dz;
            stype zmax = (z == mz - dz ? z - dz : z + dz);
            FUNC(x,y,zmin) -= FUNC(x,y,z) / 4;
            FUNC(x,y,zmax) -= FUNC(x,y,z) / 4;
        }
        }

        /* w-lift */
        for (y = 0;  y < my; y+=dy)
        for (x = 0;  x < mx; x+=dx)
        for (z = dz; z < mz; z+=dz2) {
            stype zmin = z - dz;
            stype zmax = (z == mz - dz ? z - dz : z + dz);
            FUNC(x,y,z) += (FUNC(x,y,zmin)+FUNC(x,y,zmax)) / 2;
        }
        return true;
    }

public:
    //! ----------------------------------------------------------------------------------
    Wavelets_CDF53(const EnumWavelet &type, const ltype &maxLevels, const bool &normalize, const bool &inplace) :
        mType(type), Wavelets<dtype>(maxLevels, normalize, inplace) {

        AMM_error_invalid_arg(mType != EnumWavelet::Approximating && mType != EnumWavelet::Interpolating,
                               "Wavelets_CDF53(%d): invalid wavelet type!",
                               int(type));

        this->scaling_weights = {0.5, 1.0, 0.5};
        this->wavelet_weights = {-0.125, -0.25, 0.75, -0.25, -0.125};
        init();
    }

    //! ----------------------------------------------------------------------------------
    static void lift_x_boundary(dtype* f, int Nx, int Ny, int Nz, int nx, int ny, int nz, int l) {
        assert(AMM_is_pow2(Nx-1));
        assert(Ny==1 || AMM_is_pow2(Ny-1));
        assert(Nz==1 || AMM_is_pow2(Nz-1));
        assert(nx<=Nx); assert(ny<=Ny); assert(nz<=Nz);
        assert(l >= 0);

        int L = 1 << l;
        int Mx = (Nx+L-1)/ L, My=(Ny+L-1)/L, Mz=(Nz+L-1)/L;
        if (Mx <= 1) return;

        auto bl = bound_level(nx, ny, nz, l);
        int mx = bl[0], my = bl[1], mz = bl[2], px = bl[3], py = bl[4], pz = bl[5];
        /* linear extrapolate */
        if (mx < px) {
            assert(mx + 1 == px);
            for (int z = 0; z < pz; ++z) {
                for (int y = 0; y < py; ++y) {
                    dtype a = f[AMM_xyz2idx(mx - 2, y, z, Nx, Ny, Nz)];
                    dtype b = f[AMM_xyz2idx(mx - 1, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(mx, y, z, Nx, Ny, Nz)] = 2 * b - a;
                }
            }
        }
        /* odd lift */
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 1; x < px; x += 2) {
                    dtype fl = f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)];
                    dtype fr = f[AMM_xyz2idx(x + 1, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] -= 0.5 * (fl + fr);
                }
            }
        }
        /* even lift */
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 1; x < px; x += 2) {
                    dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)] += 0.25 * fm;
                    f[AMM_xyz2idx(x + 1, y, z, Nx, Ny, Nz)] += 0.25 * fm;
                }
            }
        }
        /* shuffle */
        std::vector<dtype> t(Mx>>1);
        int s = (Mx + 1) >> 1;
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 1; x < Mx; x += 2) {
                    t[x >> 1] = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x >> 1, y, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)];
                }
                if ((Mx & 1) != 0)
                    f[AMM_xyz2idx(Mx >> 1, y, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(Mx - 1, y, z, Nx, Ny, Nz)];
                for (int x = 0; x < (Mx >> 1); ++x)
                    f[AMM_xyz2idx(s + x, y, z, Nx, Ny, Nz)] = t[x];
            }
        }
    }

    static void ilift_x_boundary(dtype* f, int Nx, int Ny, int Nz, int l) {
        assert(AMM_is_pow2(Nx-1));
        assert(Ny==1 || AMM_is_pow2(Ny-1));
        assert(Nz==1 || AMM_is_pow2(Nz-1));
        assert(l >= 0);

        int L = 1 << l;
        int Mx = (Nx + L - 1) / L, My = (Ny + L - 1) / L, Mz = (Nz + L - 1) / L;
        if (Mx <= 1) return;

        std::vector<dtype> t(Mx>>1);
        int s = (Mx + 1) >> 1;
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 0; x < (Mx >> 1); ++x)
                    t[x] = f[AMM_xyz2idx(s + x, y, z, Nx, Ny, Nz)];
                if ((Mx & 1) != 0)
                    f[AMM_xyz2idx(Mx - 1, y, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(Mx >> 1, y, z, Nx, Ny, Nz)];
                for (int x = ((Mx >> 1) << 1) - 1; x >= 1; x -= 2) {
                    f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x >> 1, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] = t[x >> 1];
                }
            }
        }
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 1; x < Mx; x += 2) {
                    dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)] -= 0.25*fm;
                    f[AMM_xyz2idx(x + 1, y, z, Nx, Ny, Nz)] -= 0.25*fm;
                }
            }
        }
        for (int z = 0; z < Mz; ++z) {
            for (int y = 0; y < My; ++y) {
                for (int x = 1; x < Mx; x += 2) {
                    dtype fl = f[AMM_xyz2idx(x - 1, y, z, Nx, Ny, Nz)];
                    dtype fr = f[AMM_xyz2idx(x + 1, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] += (fl + fr) * 0.5;
                }
            }
        }
    }

    static void lift_y_boundary(dtype* f, int Nx, int Ny, int Nz, int nx, int ny, int nz, int l) {
        assert(AMM_is_pow2(Nx - 1));
        assert(Ny==1 || AMM_is_pow2(Ny - 1));
        assert(Nz==1 || AMM_is_pow2(Nz - 1));
        assert(nx <= Nx); assert(ny <= Ny); assert(nz <= Nz);
        assert(l >= 0);

        int L = 1 << l;
        int Mx = (Nx + L - 1) / L, My = (Ny + L - 1) / L, Mz = (Nz + L - 1) / L;
        if (My <= 1) return;

        auto bl = bound_level(nx, ny, nz, l);
        int mx = bl[0], my = bl[1], mz = bl[2], px = bl[3], py = bl[4], pz = bl[5];
        /* linear extrapolate */
        if (my < py) {
            assert(my + 1 == py);
            for (int z = 0; z < pz; ++z) {
                for (int x = 0; x < px; ++x) {
                    dtype a = f[AMM_xyz2idx(x, my - 2, z, Nx, Ny, Nz)];
                    dtype b = f[AMM_xyz2idx(x, my - 1, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, my, z, Nx, Ny, Nz)] = 2 * b - a;
                }
            }
        }
        /* odd lift */
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 1; y < py; y += 2) {
                    dtype fl = f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)];
                    dtype fr = f[AMM_xyz2idx(x, y + 1, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] -= 0.5 * (fl + fr);
                }
            }
        }
        /* even lift */
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 1; y < py; y += 2) {
                    dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)] += 0.25 * fm;
                    f[AMM_xyz2idx(x, y + 1, z, Nx, Ny, Nz)] += 0.25 * fm;
                }
            }
        }
        /* shuffle */
        std::vector<dtype> t(My>>1);
        int s = (My + 1) >> 1;
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 1; y < My; y += 2) {
                    t[y >> 1] = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y >> 1, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)];
                }
                if ((My & 1) != 0)
                    f[AMM_xyz2idx(x, My >> 1, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, My - 1, z, Nx, Ny, Nz)];
                for (int y = 0; y < (My >> 1); ++y)
                    f[AMM_xyz2idx(x, s + y, z, Nx, Ny, Nz)] = t[y];
            }
        }
    }

    static void ilift_y_boundary(dtype* f, int Nx, int Ny, int Nz, int l) {
        assert(AMM_is_pow2(Nx-1));
        assert(Ny==1 || AMM_is_pow2(Ny-1));
        assert(Nz==1 || AMM_is_pow2(Nz-1));
        assert(l >= 0);

        int L = 1 << l;
        int Mx = (Nx + L - 1) / L, My = (Ny + L - 1) / L, Mz = (Nz + L - 1) / L;
        if (My <= 1) return;

        std::vector<dtype> t(My>>1);
        int s = (My + 1) >> 1;
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 0; y < (My >> 1); ++y)
                    t[y] = f[AMM_xyz2idx(x, s + y, z, Nx, Ny, Nz)];
                if ((My & 1) != 0)
                    f[AMM_xyz2idx(x, My - 1, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, My >> 1, z, Nx, Ny, Nz)];
                for (int y = ((My >> 1) << 1) - 1; y >= 1; y -= 2) {
                    f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y >> 1, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] = t[y >> 1];
                }
            }
        }
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 1; y < My; y += 2) {
                    dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)] -= 0.25*fm;
                    f[AMM_xyz2idx(x, y + 1, z, Nx, Ny, Nz)] -= 0.25*fm;
                }
            }
        }
        for (int z = 0; z < Mz; ++z) {
            for (int x = 0; x < Mx; ++x) {
                for (int y = 1; y < My; y += 2) {
                    dtype fl = f[AMM_xyz2idx(x, y - 1, z, Nx, Ny, Nz)];
                    dtype fr = f[AMM_xyz2idx(x, y + 1, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] += (fl + fr) * 0.5;
                }
            }
        }
    }

    static void lift_z_boundary(dtype* f, int Nx, int Ny, int Nz, int nx, int ny, int nz, int l) {
      assert(AMM_is_pow2(Nx - 1));
      assert(Ny==1 || AMM_is_pow2(Ny - 1));
      assert(Nz==1 || AMM_is_pow2(Nz - 1));
      assert(nx <= Nx); assert(ny <= Ny); assert(nz <= Nz);
      assert(l >= 0);

      int L = 1 << l;
      int Mx = (Nx + L - 1) / L, My = (Ny + L - 1) / L, Mz = (Nz + L - 1) / L;
      if (Mz <= 1) return;

      auto bl = bound_level(nx, ny, nz, l);
      int mx = bl[0], my = bl[1], mz = bl[2], px = bl[3], py = bl[4], pz = bl[5];
      /* linear extrapolate */
      if (mz < pz) {
          assert(mz + 1 == pz);
          for (int y = 0; y < py; ++y) {
              for (int x = 0; x < px; ++x) {
                  dtype a = f[AMM_xyz2idx(x, y, mz - 2, Nx, Ny, Nz)];
                  dtype b = f[AMM_xyz2idx(x, y, mz - 1, Nx, Ny, Nz)];
                  f[AMM_xyz2idx(x, y, mz, Nx, Ny, Nz)] = 2 * b - a;
              }
          }
      }
      /* odd lift */
      for (int y = 0; y < py; ++y) {
          for (int x = 0; x < px; ++x) {
              for (int z = 1; z < pz; z += 2) {
                  dtype fl = f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)];
                  dtype fr = f[AMM_xyz2idx(x, y, z + 1, Nx, Ny, Nz)];
                  f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] -= 0.5 * (fl + fr);
              }
          }
      }
      /* even lift */
      for (int y = 0; y < py; ++y) {
          for (int x = 0; x < px; ++x) {
              for (int z = 1; z < pz; z += 2) {
                  dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                  f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)] += 0.25 * fm;
                  f[AMM_xyz2idx(x, y, z + 1, Nx, Ny, Nz)] += 0.25 * fm;
              }
          }
      }
      /* shuffle */
      std::vector<dtype> t(Mz>>1);
      int s = (Mz + 1) >> 1;
      for (int y = 0; y < My; ++y) {
          for (int x = 0; x < Mx; ++x) {
              for (int z = 1; z < Mz; z += 2) {
                  t[z >> 1] = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                  f[AMM_xyz2idx(x, y, z >> 1, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)];
              }
              if ((Mz & 1) != 0)
                  f[AMM_xyz2idx(x, y, Mz >> 1, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y, Mz - 1, Nx, Ny, Nz)];
              for (int z = 0; z < (Mz >> 1); ++z)
                  f[AMM_xyz2idx(x, y, s + z, Nx, Ny, Nz)] = t[z];
          }
      }
    }

    static void ilift_z_boundary(dtype* f, int Nx, int Ny, int Nz, int l) {
        assert(AMM_is_pow2(Nx - 1));
        assert(Ny==1 || AMM_is_pow2(Ny - 1));
        assert(Nz==1 || AMM_is_pow2(Nz - 1));
        assert(l >= 0);

        int L = 1 << l;
        int Mx = (Nx + L - 1) / L, My = (Ny + L - 1) / L, Mz = (Nz + L - 1) / L;
        if (Mz <= 1) return;

        std::vector<dtype> t(Mz>>1);
        int s = (Mz + 1) >> 1;
        for (int y = 0; y < My; ++y) {
            for (int x = 0; x < Mx; ++x) {
                for (int z = 0; z < (Mz >> 1); ++z)
                    t[z] = f[AMM_xyz2idx(x, y, s + z, Nx, Ny, Nz)];
                if ((Mz & 1) != 0)
                    f[AMM_xyz2idx(x, y, Mz - 1, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y, Mz >> 1, Nx, Ny, Nz)];
                for (int z = ((Mz >> 1) << 1) - 1; z >= 1; z -= 2) {
                    f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)] = f[AMM_xyz2idx(x, y, z >> 1, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] = t[z >> 1];
                }
            }
        }
        for (int y = 0; y < My; ++y) {
            for (int x = 0; x < Mx; ++x) {
                for (int z = 1; z < Mz; z += 2) {
                    dtype fm = f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)] -= 0.25*fm;
                    f[AMM_xyz2idx(x, y, z + 1, Nx, Ny, Nz)] -= 0.25*fm;
                }
            }
        }
        for (int y = 0; y < My; ++y) {
            for (int x = 0; x < Mx; ++x) {
                for (int z = 1; z < Mz; z += 2) {
                    dtype fl = f[AMM_xyz2idx(x, y, z - 1, Nx, Ny, Nz)];
                    dtype fr = f[AMM_xyz2idx(x, y, z + 1, Nx, Ny, Nz)];
                    f[AMM_xyz2idx(x, y, z, Nx, Ny, Nz)] += (fl + fr) * 0.5;
                }
            }
        }
    }

    //! ----------------------------------------------------------------------------------
    static void extrapolate(const size_t in_dims[3], const size_t out_dims[3], std::vector<dtype> &out_data) {

        // is extrapolation needed?
        bool extrapolation_needed = false;
        for(size_t d = 0; d < 3; d++) {
            extrapolation_needed |= (out_dims[d] > in_dims[d]);
        }

        if (!extrapolation_needed)
            return;

        AMM_log_info << "Linear-Lifting [" << in_dims[0] << " x " << in_dims[1] << " x " << in_dims[2] << "] data "
                     << "into [" << out_dims[0] << " x " << out_dims[1] << " x " << out_dims[2] << "]... ";
        fflush(stdout);
        amm::timer t;

        int l = 0;
        int nx = (int)in_dims[0],  ny = (int)in_dims[1],  nz = (int)in_dims[2];
        int Nx = (int)out_dims[0], Ny = (int)out_dims[1], Nz = (int)out_dims[2];
        while (true) {
            auto bl = bound_level(nx, ny, nz, l);
            if (bl[0] <= 2 || (ny>1 && bl[1] <= 2) || (nz>1 && bl[2] <= 2)) {
                break;
            }
            Wavelets_CDF53::lift_x_boundary(out_data.data(), Nx, Ny, Nz, nx, ny, nz, l);
            Wavelets_CDF53::lift_y_boundary(out_data.data(), Nx, Ny, Nz, nx, ny, nz, l);
            Wavelets_CDF53::lift_z_boundary(out_data.data(), Nx, Ny, Nz, nx, ny, nz, l);
            ++l;
        }
        while (l > 0) {
            --l;
            Wavelets_CDF53::ilift_z_boundary(out_data.data(), Nx, Ny, Nz, l);
            Wavelets_CDF53::ilift_y_boundary(out_data.data(), Nx, Ny, Nz, l);
            Wavelets_CDF53::ilift_x_boundary(out_data.data(), Nx, Ny, Nz, l);
        }
        t.stop();
        AMM_logc_info << " done!" << t << std::endl;
    }

    //! ----------------------------------------------------------------------------------
};
}
#endif

