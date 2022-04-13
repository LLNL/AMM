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
#ifndef AMM_CREATOR_WCDF53_H
#define AMM_CREATOR_WCDF53_H

//! ----------------------------------------------------------------------------
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "macros.hpp"
#include "containers/vec.hpp"
#include "containers/unordered_map.hpp"
#include "types/dtypes.hpp"
#include "tree/octree_cell_iterators.hpp"
#include "tree/amtree.hpp"
#include "utils/exceptions.hpp"

#include "amm.hpp"
#include "creator/creator.hpp"
#include "creator/stencil_wcdf53.hpp"


//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! a mesh creator for cdf53 wavelet stencils
//! ----------------------------------------------------------------------------
template <typename TypeValue,               // data type for function values
          TypeDim   Dim,                    // dimensionality of the tree
          TypeScale MaxLevels>
class amm_creator_wcdf53 : public amm::amm_creator_wstencils<TypeValue, Dim, MaxLevels> {

    using TypeAMM       = amm::AMM<TypeValue, Dim, MaxLevels>;
    using BaseCreator   = amm::amm_creator_wstencils<TypeValue, Dim, MaxLevels>;

    using TypeLocCode         = typename TypeAMM::TypeLocCode;
    using TypeVertex          = typename TypeAMM::TypeVertex;

    using DualIterator        = typename TypeAMM::DualIterator;
    using PrimalIterator      = typename TypeAMM::PrimalIterator;
    using PrimalDualIterator  = typename TypeAMM::PrimalDualIterator;
    using NeighborAllIterator = typename TypeAMM::NeighborAllIterator;


    //! ------------------------------------------------------------------------
    //! structure to cache wavelet coefficients!
    //! ------------------------------------------------------------------------
    struct WCoeff {

        TypeIndex mIndex;           // std::sort swaps in place so i cant use const
        TypeScale mLevel;
        EnumWCoefficient mStype;
        TypeValue mValue;           // but this is the only thing that changes!

        WCoeff(const TypeIndex &_index, const TypeScale &_level, const EnumWCoefficient &_stype, const TypeValue &_val) :
            mIndex(_index), mLevel(_level), mStype(_stype), mValue(_val) {}


        inline bool
        is_same(const TypeIndex &_index, const TypeScale &_level, const EnumWCoefficient &_stype) const {
            return mIndex == _index && mLevel == _level && mStype == _stype;
        }

        inline bool
        operator< (const WCoeff& rhs) const {

            // first sort on level (coarser first)
            // then on stencil type (vertex, edge, face, cube)
            if (mLevel != rhs.mLevel) return mLevel < rhs.mLevel;
            if (mStype != rhs.mStype) return mStype < rhs.mStype;

            // default to just the rowmajor index
            return mIndex < rhs.mIndex;
        }
    };

    //! --------------------------------------------------------------------------------
    //! this mesh creator depends on the type of the wavelet
    //! (interpolating or apporximating)
    //! --------------------------------------------------------------------------------
    const EnumWavelet m_wtype;
    amm::unordered_map<TypeIndex, WCoeff> m_wcoeffs;

    //! --------------------------------------------------------------------------------
    //! scaling in all dimensions
    //! --------------------------------------------------------------------------------
    void
    createStencil_k0(const TypeVertex &p, const TypeScale &plevel) {

        TypeDim vdim = TypeAMM::OctUtils::get_vertexDimension(p, this->m_mesh->depth()-plevel);
        AMM_error_logic(0 != vdim,
                        "AMMTreeCreator.createStencil_k0() : %s is not a valid point of k0 cell at level %d; vdim = %d\n",
                        p.c_str(), plevel, int(vdim));

        // nodes centered at p
        DualIterator diter (p, 0, this->m_mesh->depth(), plevel);
        for(diter.begin(); !diter.end(); diter.next()) {
            const TypeLocCode ncode = this->m_mesh->create_node_at(diter.val());
        }
    }

    //! --------------------------------------------------------------------------------
    //! wavelet in all dimensions
    //! --------------------------------------------------------------------------------
    void
    createStencil_kd(const TypeVertex &p, const TypeScale &plevel) {

        TypeDim vdim = TypeAMM::OctUtils::get_vertexDimension(p, this->m_mesh->depth()-plevel);
        AMM_error_logic(Dim != vdim,
                        "AMMTreeCreator.createStencil_d() : %s is not a valid point of kd cell at level %d; vdim = %d\n",
                        p.c_str(), plevel, int(vdim));

        // --------------------------------------------------------------------------------
        // phase 1: nodes in the central cell
        const EnumAxes splitAxes_all = (2==Dim) ? EnumAxes::XY : EnumAxes::XYZ;

        TypeLocCode ncode = this->m_mesh->create_node_at(p);
        this->m_mesh->split_node(ncode, splitAxes_all);

        // --------------------------------------------------------------------------------
        // phase 2: nodes in neighboring cells
        if (this->m_wtype != EnumWavelet::Interpolating) {

            NeighborAllIterator niter (p, this->m_mesh->depth(), plevel);
            for(niter.begin(); !niter.end(); niter.next()) {

                const TypeVertex &nbr = niter.val();

                const TypeLocCode ncode = this->m_mesh->create_node_at(nbr);
                const EnumAxes split_axes = TypeAMM::OctUtils::matching_axes(p, nbr);

                this->m_mesh->split_node(ncode, split_axes);
            }
        }
     }

    //! --------------------------------------------------------------------------------
    //! wavelet in at least one and at most D-1 dimension
    //! --------------------------------------------------------------------------------
    void
    createStencil_k(const TypeVertex &p, const TypeScale &plevel) {

        TypeDim vdim = TypeAMM::OctUtils::get_vertexDimension(p, this->m_mesh->depth()-plevel);

        if (Dim == 2 && 1 == vdim) {}
        else if (Dim == 3 && (1 == vdim || 2 == vdim)) {}
        else {
            AMM_error_logic(true,
                            "AMMTreeCreator.createStencil_k() : %s is not a valid point of k cell at level %d; vdim = %d\n",
                            p.c_str(), plevel, int(vdim));
        }

        // --------------------------------------------------------------------------------
        // phase 1: dual of the point
        DualIterator diter (p, vdim, this->m_mesh->depth(), 0);

        for(diter.begin(); !diter.end(); diter.next()) {

            const TypeVertex &dv = diter.val();
            const TypeLocCode ncode = this->m_mesh->create_node_at(dv);

            EnumAxes split_axes = TypeAMM::OctUtils::matching_axes(p, dv);
            this->m_mesh->split_node(ncode, split_axes);
        }

        // --------------------------------------------------------------------------------
        // phase 2: nodes in neighboring cells
        if (this->m_wtype != EnumWavelet::Interpolating) {

            PrimalDualIterator pdnewiter (p, vdim, this->m_mesh->depth());

            if (2 == Dim) {

                for(pdnewiter.begin(); !pdnewiter.end(); pdnewiter.next()) {

                    const TypeVertex &nbr = pdnewiter.val();
                    const TypeLocCode ncode = this->m_mesh->create_node_at(nbr);
                }
            }
            else {
                for(pdnewiter.begin(); !pdnewiter.end(); pdnewiter.next()) {

                    const TypeVertex &nbr = pdnewiter.val();

                    const TypeLocCode ncode = this->m_mesh->create_node_at(nbr);

                    EnumAxes split_axes = TypeAMM::OctUtils::matching_axes(p, nbr);
                    this->m_mesh->split_node(ncode, split_axes);
                }
            }
        }
    }

public:
    //! --------------------------------------------------------------------------------
    amm_creator_wcdf53(const EnumWavelet wtype, const size_t insize[],
                       const bool allow_rectangular, const bool allow_vacuum) :
                  BaseCreator(stencil_cdf53<Dim>::init(wtype),
                              new TypeAMM(insize, allow_rectangular, allow_vacuum)),
                  m_wtype(wtype)
    {}


    void
    add_stencil(const TypeVertex &p, const TypeValue val,
                const TypeScale plevel, const EnumWCoefficient stype) {

        static const TypeDim mdim = AMM_pow2(Dim)-1;
        AMM_error_logic(as_utype(stype) > mdim,
                        "AMMTreeCreator.create_stencil(): Invalid stencil type %d for %dD Tree!\n",
                        int(stype), int(Dim));

        const TypeIndex pidx = this->m_mesh->p2idx(p);

        auto iter = this->m_wcoeffs.find(pidx);
        if (iter == this->m_wcoeffs.end()) {

            // TODO: i think this is inefficient.
            // no need to store all this data into a map (slow)
            // if we can compute them on the fly very fast!
            this->m_wcoeffs.insert(pidx, WCoeff(pidx, plevel, stype, val));
            //this->m_wcoeffs.emplace(pidx, WCoeff(pidx, plevel, stype, val));
        }

        else {
            WCoeff &wc = iter->second;
            AMM_error_logic(!wc.is_same(pidx, plevel, stype),
                            "AMMTreeCreator.add_stencil(): Mismatch in coefficient! earlier=(%lu,%d,%d) != now=(%lu,%d,%d)!\n",
                            wc.mIndex, wc.mLevel, wc.mStype, pidx, plevel, stype);
            wc.mValue += val;
        }
    }

    void
    update() {
        amm::timer t;

        std::vector<WCoeff> sorted_coeffs;
        sorted_coeffs.reserve(m_wcoeffs.size());
        std::for_each(m_wcoeffs.begin(), m_wcoeffs.end(),
                      [&sorted_coeffs](auto iter){sorted_coeffs.emplace_back(iter.second);});

        AMM_logc_info << " (inserting" << t << ")";
        fflush(stdout);

        t.start();
        std::sort(sorted_coeffs.begin(), sorted_coeffs.end());
        AMM_logc_info << " (sorting" << t << ")";
        fflush(stdout);

        t.start();
        m_wcoeffs.clear();
        AMM_logc_info << " (clearing" << t << ")";
        fflush(stdout);

        // now actual processing
        for(auto iter = sorted_coeffs.begin(); iter != sorted_coeffs.end(); iter++) {
            const WCoeff &wc = *iter;
            this->create_stencil(this->m_mesh->idx2p(wc.mIndex), wc.mValue, wc.mLevel, wc.mStype);
        }
    }

    void
    create_stencil(const TypeVertex &p, const TypeValue val,
                   const TypeScale plevel, const EnumWCoefficient stype) {

        static const TypeDim mdim = AMM_pow2(Dim)-1;
        AMM_error_logic(as_utype(stype) > mdim,
                        "AMMTreeCreator.create_stencil(): Invalid stencil type %d for %dD Tree!\n",
                        int(stype), int(Dim));

        // --------------------------------------------------------------------------------
        // first, we create all the nodes (split points are added along the way)
        switch (as_utype(stype)) {
        case 0:
                    this->createStencil_k0(p, plevel);
                    break;
        case mdim:
                    this->createStencil_kd(p, plevel);
                    break;
        default:
                    this->createStencil_k(p, plevel);
                    break;
        }


        // --------------------------------------------------------------------------------
        // now, we update all the vertices of the stencil
        // simple solution for scaling stencil
        if (stype == EnumWCoefficient::S ||
            stype == EnumWCoefficient::SS ||
            stype == EnumWCoefficient::SSS) {
            this->m_mesh->stage_vertex(p, plevel, val);
            return;
        }

        // place the correct stencil on the mesh
        static TypeVertex sp;
        auto stenc = this->m_stencils[as_utype(stype)];

        // vertices need to be added to the next level
        const TypeScale vlevel = plevel+1;
        const TypeCoord dx = AMM_ldx(vlevel, this->m_mesh->depth());

        const size_t npoints = stenc.size();
        for(size_t i = 0; i < npoints; i++) {

            auto si = stenc.point(i);
            auto sw = stenc.weight(i);

            // TODO: could we not add all the vertices?
            // compute the point on the grid
            if (this->create_stencil_point(p, si, dx, this->m_mesh->size(), sp)){
                this->m_mesh->stage_vertex(sp, vlevel, val*sw);
            }
        }

        // -----------------------------------------------------------------
        // set the timestamp of mesh update
        this->m_mesh->set_utime();
    }
};

}   // end of namespace
/// --------------------------------------------------------------------------------
/// --------------------------------------------------------------------------------
#endif
