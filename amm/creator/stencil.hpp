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
#ifndef AMM_STENCIL_H
#define AMM_STENCIL_H

//! ----------------------------------------------------------------------------
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include "macros.hpp"
#include "amm/types/dtypes.hpp"
#include "amm/types/enums.hpp"
#include "amm/containers/vec.hpp"
#include "utils/utils.hpp"
#include "utils/logger.hpp"
#include "utils/exceptions.hpp"

namespace amm {

//! ----------------------------------------------------------------------------
//! A class to represent a single stencil
//! ----------------------------------------------------------------------------
template <TypeDim Dim>
class stencil {

    static_assert((Dim == 1 || Dim == 2 || Dim == 3), "Stencil works for 1D, 2D, and 3D only!");

    using dtype = float;
    using otype = int;
    using stype = TypeIndex;

public:
    using TypeOffset = Vec<Dim, otype>;
    using TypeVecs = Vec<Dim, stype>;

private:
    //! dimensionality of the stencil anchor (= num of wavelet coefficients)
    TypeDim m_dimAnchor;

    //! span of the stencil (width from the center)
    TypeOffset m_span;

    //! all points and all weights in the stencil
    std::vector<TypeOffset> m_points;
    std::vector<dtype> m_weights;

    //! ------------------------------------------------------------------------

public:
    const TypeDim& dim_anchor() const {             return m_dimAnchor; }
    const TypeOffset& span() const {                return m_span; }

    const TypeOffset& point(const stype i) const {  return m_points[i]; }
    const dtype& weight(const stype i) const {      return m_weights[i];}
    stype size() const {                            return m_points.size(); }

    stencil() {}

    //! construct a 1D stencil
    stencil(EnumWCoefficient _stype, const std::vector<dtype> &_weights) {

        AMM_error_invalid_arg(Dim != 1, "Stencil(_stype, _weights, _epoints) works for 1D only!\n");

        // create weights
        const stype wsz = _weights.size();
        AMM_error_invalid_arg(wsz%2 != 1, "Stencil creation needs odd number of weights\n");

        // dimensinality of this stencil
        m_dimAnchor = TypeDim(_stype);

        // span of the stencil
        const int p = int((wsz-1)/2);
        m_span[0] = p;

        m_weights = _weights;

        // create points
        for(int i = -p; i <= p; i++) {
            m_points.push_back(TypeOffset(otype(i)));
        }

        AMM_error_logic(m_weights.size() != m_points.size(),
                               "Stencil creation failed. Mismatch in numbers of points (%d) and weights (%d)!\n",
                               m_weights.size(), m_points.size());
    }

    //! create a 2D stencil as a tensor product
    stencil(const stencil<1> &y, const stencil<1> &x) {

        AMM_error_invalid_arg(Dim != 2, "Stencil(y, x) works for 2D only!\n");

        // determine the type
        m_dimAnchor = x.dim_anchor() + y.dim_anchor();

        m_span[0] = x.span()[0];
        m_span[1] = y.span()[0];

        // tensor product the points and weights
        const stype nx = x.size();
        const stype ny = y.size();

        m_points.resize(nx*ny);
        m_weights.resize(nx*ny);

        stype i = 0;
        for(stype iy = 0; iy < ny; iy++) {
        for(stype ix = 0; ix < nx; ix++) {

            m_points[i] = TypeOffset(x.point(ix)[0], y.point(iy)[0]);
            m_weights[i] = x.weight(ix) * y.weight(iy);
            i++;
        }
        }
    }

    stencil(const stencil<1> &z, const stencil<2> &yx) {

        AMM_error_invalid_arg(Dim != 3, "Stencil(z, yx) works for 3D only!\n");

        // determine the type
        m_dimAnchor = yx.dim_anchor() + z.dim_anchor();

        m_span[0] = yx.span()[0];
        m_span[1] = yx.span()[1];
        m_span[2] = z.span()[0];

        // tensor product the points and weights
        const stype nxy = yx.size();
        const stype nz = z.size();

        m_points.resize(nxy*nz);
        m_weights.resize(nxy*nz);

        stype i = 0;
        for(stype iz = 0; iz < nz; iz++) {
        for(stype ixy = 0; ixy < nxy; ixy++) {

            m_points[i] = TypeOffset(yx.point(ixy)[0], yx.point(ixy)[1], z.point(iz)[0]);
            m_weights[i] = yx.weight(ixy) * z.weight(iz);
            i++;
        }
        }
    }

    void print() const {
        AMM_log_info << int(Dim) << "D Stencil. dim_anchor = " << int(m_dimAnchor) << "\n";
        AMM_log_info << " span = " << m_span << "\n";
        AMM_log_info << " points = " << m_points.size() << "\n";
    }
};

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
}   // end of namespace
#endif
