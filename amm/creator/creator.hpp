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
#ifndef AMM_CREATOR_WSTENCILS_H
#define AMM_CREATOR_WSTENCILS_H

//! ----------------------------------------------------------------------------
#include "types/dtypes.hpp"
#include "creator/stencil.hpp"
#include "amm.hpp"

namespace amm {

//! ----------------------------------------------------------------------------
//! an abstract mesh creator using wavelet stencils
//! ----------------------------------------------------------------------------
template <typename TypeValue,               // data type for function values
          TypeDim   Dim,                    // dimensionality of the tree
          TypeScale MaxLevels>              // max level of the tree to create
class amm_creator_wstencils {

    using TypeAMM     = amm::AMM<TypeValue, Dim, MaxLevels>;
    using TypeStencil = amm::stencil<Dim>;

    using TypeVertex  = typename TypeAMM::TypeVertex;
    using TypeOffset  = typename TypeStencil::TypeOffset;

    //! ------------------------------------------------------------------------
    //! the mesh that is created
    //! ------------------------------------------------------------------------
protected:

    std::vector<TypeStencil> m_stencils;
    TypeAMM *m_mesh = nullptr;

    //! ------------------------------------------------------------------------
    amm_creator_wstencils(const std::vector<TypeStencil> &nstencils,
                          TypeAMM *mesh) :
        m_stencils(nstencils), m_mesh(mesh) {}

    //! ------------------------------------------------------------------------
    //! create a stencil point
    static bool
    create_stencil_point(const TypeVertex &p,      // center point of the stencil
                         const TypeOffset &o,      // offset of the point wrt the center
                         const TypeCoord s,        // scale of the stencil
                         const TypeVertex &size,   // size of the domain
                         TypeVertex &_) {          // output point

        _ = TypeVertex(TypeCoord(0));
        for(TypeDim d = 0; d < Dim; d++){
            int v = int(p[d]) + int(s)*int(o[d]);
            if (v < 0 || v >= int(size[d]))
                return false;
            _[d] = TypeCoord(v);
        }
        return true;
    }

public:
    //! ------------------------------------------------------------------------
    //! public interface of the mesh creator
    //! ------------------------------------------------------------------------
    const TypeAMM* output() const{    return this->m_mesh;  }

    //! ------------------------------------------------------------------------
    virtual void create_stencil(const TypeVertex &p, const TypeValue val,
                                const TypeScale plevel, const EnumWCoefficient stype) = 0;

    //! ------------------------------------------------------------------------
};
}   // end of namespace
#endif
