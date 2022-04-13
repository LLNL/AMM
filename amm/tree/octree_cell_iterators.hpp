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
#ifndef AMM_OCTREE_CELL_ITERATORS_H
#define AMM_OCTREE_CELL_ITERATORS_H

#include <cstdio>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "types/dtypes.hpp"
#include "containers/vec.hpp"
#include "utils/utils.hpp"
#include "octree_utils.hpp"

namespace amm {
namespace octree {

/// ----------------------------------------------------------------------------
/// Base class for a cell iterator
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class octree_cell_iterator {

    static_assert((Dim == 2 || Dim == 3), "octree_cell_iterator works for 2D and 3D only!");

protected:
    using TypeVec = Vec<Dim, TypeCoord>;
    using TypeDir = Vec<Dim, int8_t>;
    using octutils = amm::octree_utils<Dim>;

private:
    using TypeIter = typename std::vector<TypeVec>::const_iterator;
    TypeIter mIdx;                  //! current index of the iterator

protected:
    //! input
    const TypeVec mp;               //! input point
    const TypeDim mk;               //! dimensionality of input point
    const TypeScale mScale;         //! scale of input point (needed only for primal iterator of vertices)
    const TypeScale mMaxLevels;     //! max levels in the tree

    //! output
    std::vector<TypeVec> mOutput;   //! complete output

    octree_cell_iterator(const TypeVec &p, const TypeDim k, const TypeScale maxLevels, const TypeScale scale = 0) :
        mp(p), mk(k), mScale(scale), mMaxLevels(maxLevels) {
        mIdx = mOutput.end();
    }

    //! add a point to iterator output
    void add_point(const TypeVec &p, const TypeCoord M, const TypeCoord dx, const TypeDir &dir) {

        TypeVec o;
        for(TypeDim d = 0; d < Dim; d++) {
            TypeCoord ddx = abs(dir[d])*dx;

            if (dir[d] < 0 && p[d] < ddx)      return; // need to subtract dx from p, so check lower bound
            if (dir[d] > 0 && p[d] >= M-ddx)   return; // need to add dx to p, so check upper bound

            o[d] = p[d] + dir[d]*dx;
        }
        this->mOutput.push_back(o);
    }

    /// --------------------------------------------------------------------------------------
    /// directly add neighboring points
    /// --------------------------------------------------------------------------------------

    //! add axis-aligned neighbors (only one coordinate changes)
    void add_nbrs_axis_aligned(const TypeVec &p, const TypeCoord M, const TypeCoord dx) {

        this->mOutput.reserve(2*Dim);
        if (2 == Dim) {
            this->add_point(p, M, dx, TypeDir(-1, 0));
            this->add_point(p, M, dx, TypeDir( 0,-1));
            this->add_point(p, M, dx, TypeDir(+1, 0));
            this->add_point(p, M, dx, TypeDir( 0,+1));
        }
        else {
            this->add_point(p, M, dx, TypeDir(-1, 0, 0));
            this->add_point(p, M, dx, TypeDir( 0,-1, 0));
            this->add_point(p, M, dx, TypeDir(+1, 0, 0));
            this->add_point(p, M, dx, TypeDir( 0, 0,-1));
            this->add_point(p, M, dx, TypeDir( 0,+1, 0));
            this->add_point(p, M, dx, TypeDir( 0, 0,+1));
        }
    }

    //! add diagonal neighbors
    void add_nbrs_diagonal(const TypeVec &p, const TypeCoord M, const TypeCoord dx) {

        static constexpr uint8_t nnbrs = AMM_pow2(Dim);
        this->mOutput.reserve(nnbrs);
        if (2 == Dim) {
            this->add_point(p, M, dx, TypeDir(-1, -1));
            this->add_point(p, M, dx, TypeDir(+1, -1));
            this->add_point(p, M, dx, TypeDir(+1, +1));
            this->add_point(p, M, dx, TypeDir(-1, +1));
        }
        else {
            this->add_point(p, M, dx, TypeDir(-1, -1, -1));
            this->add_point(p, M, dx, TypeDir(+1, -1, -1));
            this->add_point(p, M, dx, TypeDir(+1, +1, -1));
            this->add_point(p, M, dx, TypeDir(-1, +1, -1));
            this->add_point(p, M, dx, TypeDir(-1, +1, +1));
            this->add_point(p, M, dx, TypeDir(-1, -1, +1));
            this->add_point(p, M, dx, TypeDir(+1, -1, +1));
            this->add_point(p, M, dx, TypeDir(+1, +1, +1));
        }
    }

    //! add the two points lying on the edge
    void add_corners_edge(const TypeVec &p, const TypeCoord M, const TypeCoord dx, const TypeDim edge_orient) {

        this->mOutput.reserve(2);
        if (2 == Dim) {
            switch (edge_orient) {
                case 0: // x-edge
                    this->add_point(p, M, dx, TypeDir(-1, 0));
                    this->add_point(p, M, dx, TypeDir(+1, 0));
                    return;

                case 1: // y-edge
                    this->add_point(p, M, dx, TypeDir( 0, -1));
                    this->add_point(p, M, dx, TypeDir( 0, +1));
                    return;
            }
        }
        else {

            switch (edge_orient) {
                case 0: // x-edge
                    this->add_point(p, M, dx, TypeDir(-1, 0, 0));
                    this->add_point(p, M, dx, TypeDir(+1, 0, 0));
                    return;

                case 1: // y-edge
                    this->add_point(p, M, dx, TypeDir( 0, -1, 0));
                    this->add_point(p, M, dx, TypeDir( 0, +1, 0));
                    return;

                case 2: // z-edge
                    this->add_point(p, M, dx, TypeDir( 0, 0, -1));
                    this->add_point(p, M, dx, TypeDir( 0, 0, +1));
                    return;
            }
        }

        AMM_error_logic(true, " OctreeCellIterator<D=%d>::add_corners_edge(%d) failed!\n", int(Dim), int(edge_orient));
    }

    //! add the four points lying on the face
    void add_corners_face(const TypeVec &p, const TypeCoord M, const TypeCoord dx, const TypeDim face_orient) {

        this->mOutput.reserve(4);
        if (2 == Dim) {
            this->add_point(p, M, dx, TypeDir(-1, -1));
            this->add_point(p, M, dx, TypeDir(+1, -1));
            this->add_point(p, M, dx, TypeDir(+1, +1));
            this->add_point(p, M, dx, TypeDir(-1, +1));
            return;
        }
        else {
            switch(face_orient) {
                case 0: // p is a yz-face
                        this->add_point(p, M, dx, TypeDir(0, -1, -1));
                        this->add_point(p, M, dx, TypeDir(0, +1, -1));
                        this->add_point(p, M, dx, TypeDir(0, +1, +1));
                        this->add_point(p, M, dx, TypeDir(0, -1, +1));
                        return;

                case 1: // p is a xz-face
                        this->add_point(p, M, dx, TypeDir(-1, 0, -1));
                        this->add_point(p, M, dx, TypeDir(+1, 0, -1));
                        this->add_point(p, M, dx, TypeDir(+1, 0, +1));
                        this->add_point(p, M, dx, TypeDir(-1, 0, +1));
                        return;

                case 2: // p is a xy-face
                        this->add_point(p, M, dx, TypeDir(-1, -1, 0));
                        this->add_point(p, M, dx, TypeDir(+1, -1, 0));
                        this->add_point(p, M, dx, TypeDir(+1, +1, 0));
                        this->add_point(p, M, dx, TypeDir(-1, +1, 0));
                        return;
            }
        }

        AMM_error_logic(true, " OctreeCellIterator<D=%d>::add_corners_face(%d) failed!\n", int(Dim), int(face_orient));
    }

    //! add the eight corners of a hex (3D only)
    void add_corners_hex(const TypeVec &p, const TypeCoord M, const TypeCoord dx) {

        AMM_error_logic(2 == Dim, "OctreeCellIterator<Dim=%d>::add_corners_hex() is defined for 3D only!\n", int(Dim));

        this->mOutput.reserve(8);
        this->add_point(p, M, dx, TypeDir(-1, -1, -1));
        this->add_point(p, M, dx, TypeDir(+1, -1, -1));
        this->add_point(p, M, dx, TypeDir(+1, +1, -1));
        this->add_point(p, M, dx, TypeDir(-1, +1, -1));
        this->add_point(p, M, dx, TypeDir(-1, +1, +1));
        this->add_point(p, M, dx, TypeDir(-1, -1, +1));
        this->add_point(p, M, dx, TypeDir(+1, -1, +1));
        this->add_point(p, M, dx, TypeDir(+1, +1, +1));
    }

    //! add the nbrs of a 3D edge (3D only)
    void add_nbrs_edge(const TypeVec &p, const TypeCoord M, const TypeCoord dx, const TypeDim edge_orient) {

        AMM_error_logic(2 == Dim, "OctreeCellIterator<Dim=%d>::add_nbrs_edge(%d) works only for 3D!\n", int(Dim), int(edge_orient));

        this->mOutput.reserve(4);
        switch(edge_orient) {

            case 0:     // p is a x-edge: dual will be a quad in y-z plane
                this->add_point(p, M, dx, TypeDir(0, -1,-1));
                this->add_point(p, M, dx, TypeDir(0, +1,-1));
                this->add_point(p, M, dx, TypeDir(0, +1,+1));
                this->add_point(p, M, dx, TypeDir(0, -1,+1));
                return;

            case 1:     // p is a y-edge: dual will be a quad in x-z plane
                this->add_point(p, M, dx, TypeDir(-1, 0,-1));
                this->add_point(p, M, dx, TypeDir(+1, 0,-1));
                this->add_point(p, M, dx, TypeDir(+1, 0,+1));
                this->add_point(p, M, dx, TypeDir(-1, 0,+1));
                return;

            case 2:     // p is a z-edge: dual will be a quad in x-y plane
                this->add_point(p, M, dx, TypeDir(-1,-1, 0));
                this->add_point(p, M, dx, TypeDir(+1,-1, 0));
                this->add_point(p, M, dx, TypeDir(+1,+1, 0));
                this->add_point(p, M, dx, TypeDir(-1,+1, 0));
                return;
        }

        AMM_error_logic(true, " OctreeCellIterator<D=%d>::add_nbrs_edge(%d) failed!\n", int(Dim), int(edge_orient));
    }

    //! add the nbrs of a 3D face (3D only)
    void add_nbrs_face(const TypeVec &p, const TypeCoord M, const TypeCoord dx, const TypeDim face_orient) {

        AMM_error_logic(2 == Dim, "OctreeCellIterator<Dim=%d>::add_nbrs_face(%d) works only for 3D!\n", int(Dim), int(face_orient));

        this->mOutput.reserve(2);
        switch(face_orient) {

            case 0:     // p is a yz-face
                this->add_point(p, M, dx, TypeDir(-1,0, 0));
                this->add_point(p, M, dx, TypeDir( 1,0, 0));
                return;

            case 1:     // p is a xy-face
                this->add_point(p, M, dx, TypeDir( 0,-1, 0));
                this->add_point(p, M, dx, TypeDir( 0,+1, 0));
                return;

            case 2:     // p is a xz-face
                this->add_point(p, M, dx, TypeDir( 0, 0,-1));
                this->add_point(p, M, dx, TypeDir( 0, 0,+1));
                return;
        }

        AMM_error_logic(true, " OctreeCellIterator<D=%d>::add_nbrs_face(%d) failed!\n", int(Dim), int(face_orient));
    }

    //! offset for the nbrs across a 3D dege
    void add_nbrs_across_edge(const TypeVec &p, const TypeCoord M, const TypeCoord dx) {

        AMM_error_logic(2 == Dim, "OctreeCellIterator<Dim=%d>::add_nbrs_across_edge(%d) works only for 3D!\n", int(Dim));

        this->mOutput.reserve(12);
        this->add_point(p, M, dx, TypeDir(-1,  0, -1));
        this->add_point(p, M, dx, TypeDir( 0, -1, -1));
        this->add_point(p, M, dx, TypeDir(+1,  0, -1));
        this->add_point(p, M, dx, TypeDir( 0, +1, -1));

        this->add_point(p, M, dx, TypeDir(-1, -1,  0));
        this->add_point(p, M, dx, TypeDir(+1, -1,  0));
        this->add_point(p, M, dx, TypeDir(+1, +1,  0));
        this->add_point(p, M, dx, TypeDir(-1, +1,  0));

        this->add_point(p, M, dx, TypeDir(-1,  0, +1));
        this->add_point(p, M, dx, TypeDir( 0, -1, +1));
        this->add_point(p, M, dx, TypeDir(+1,  0, +1));
        this->add_point(p, M, dx, TypeDir( 0, +1, +1));
    }

public:
    inline void begin() {                       mIdx = mOutput.begin();         }
    inline void next() {                        mIdx++;                         }
    inline bool end() const {                   return mIdx == mOutput.end();   }
    inline const TypeVec& val() const {         return *mIdx;                   }
    inline const TypeCornerId& count() const {          return mOutput.size();  }
    inline const std::vector<TypeVec>& all() const {    return mOutput;         }

private:
    virtual void initialize() = 0;
};

/// ----------------------------------------------------------------------------
/// Iterate through the primal cells of p, a k-cell
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class primal_cell_iterator : public octree_cell_iterator<Dim> {

    static_assert((Dim == 2 || Dim == 3), "primal_cell_iterator works for 2D only!");

    using TypeVec = typename octree_cell_iterator<Dim>::TypeVec;
    using TypeDir = typename octree_cell_iterator<Dim>::TypeDir;
    using octutils = typename octree_cell_iterator<Dim>::octutils;

public:
    primal_cell_iterator(const TypeVec &p, const TypeDim k, const TypeScale MaxLevels, const TypeScale scale = 0)
        : octree_cell_iterator<Dim>(p, k, MaxLevels, scale) {

        // dimensionality of cell depends upon Dim
        AMM_error_invalid_arg(this->mk > Dim,
                               " PrimalIterator<D=%d, L=%d>(%d, %d) -- invalid dimension of p!\n",
                               int(Dim), int(this->mMaxLevels), this->mp, (int)this->mk);

        this->initialize();
    }

private:

    void initialize() {

        const TypeCoord M = AMM_ldim(this->mMaxLevels);
        const Vec<Dim, TypeScale> scale = octutils::get_vertexScales(this->mp, this->mMaxLevels);

        TypeCoord dx = AMM_ldx(scale.max(), this->mMaxLevels);

        switch(this->mk) {

            case 0: // primal of a vertex is the cell created by connecting the axis-aligned neighbors
                    if(this->mScale > 0) {
                        dx = AMM_ldx(this->mScale, this->mMaxLevels);
                    }
                    this->add_nbrs_axis_aligned(this->mp, M, dx);
                    break;

            case 1: // primal of an edge is the edge created by connecting its 2 vertices

                    {
                    // the orientation of the edge can be computed by using the
                    // max scale that its midpoint 'p' lies on
                    // e.g., for a x-edge, this->p[0] will be 1 scale higher (more refined) than this->p[1]
                    // if p is an x-edge, primal will be a x-edge, and so on

                    const TypeScale edge_orient = scale.argmax();
                    this->add_corners_edge(this->mp, M, dx, edge_orient);
                    }
                    break;

            case 2: // primal of a face is the set of its 4 vertices
                    {
                    const TypeScale face_orient = (2==Dim) ? 0 : scale.argmin();
                    this->add_corners_face(this->mp, M, dx, face_orient);
                    }
                    break;

            case 3: // primal of a hex is the set of its 8 vertices
                    this->add_corners_hex(this->mp, M, dx);
                    break;

           default:
                    AMM_error_invalid_arg(true, "PrimalIterator(%d, %d) -- Invalid dimension!\n", this->mp, (int)this->mk);
        }

        AMM_error_invalid_arg(this->count() != (1<<this->mk),
                               "PrimalIterator<D=%d, L=%d> -- Incorrect number of primal cells! Computed %d, but should have %d\n",
                               int(Dim), int(this->mMaxLevels), int(this->count()), (1<<this->mk));
    }
};

/// ----------------------------------------------------------------------------
/// Iterate through the dual cells of p, a k-cell
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class dual_cell_iterator : public octree_cell_iterator<Dim> {

    static_assert((Dim == 2 || Dim == 3), "DualIterator works for 2D only!");

    using TypeVec   = typename octree_cell_iterator<Dim>::TypeVec;
    using TypeDir   = typename octree_cell_iterator<Dim>::TypeDir;
    using octutils = typename octree_cell_iterator<Dim>::octutils;

public:
    //! dual of a k-cell centered at 'p'
    dual_cell_iterator(const TypeVec &p, const TypeDim k, const TypeScale MaxLevels, const TypeScale scale = 0)
        : octree_cell_iterator<Dim>(p, k, MaxLevels, scale) {

        // dimensionality of cell depends upon Dim
        AMM_error_invalid_arg(this->mk > Dim,
                               " DualIterator<D=%d, L=%d>(%d, %d) -- Invalid dimension of p!\n",
                               int(Dim), int(this->mMaxLevels), this->mp, (int)this->mk);

        this->initialize();
    }

private:
    void initialize() {

        const TypeCoord M = AMM_ldim(this->mMaxLevels);
        const Vec<Dim, TypeScale> scale = octutils::get_vertexScales(this->mp, this->mMaxLevels);
        TypeCoord dx = AMM_ldx(scale.max(), this->mMaxLevels);

        // ------------------------------------------------------------------------------
        // vertex
        if (this->mk == 0) {

            // dual of a vertex is the cell created by connecting the diagonal neighbors
            if(this->mScale > 0) {
                dx = AMM_ldx(this->mScale, this->mMaxLevels);
            }
            this->add_nbrs_diagonal(this->mp, M, dx/2);
            return;
        }

        // ------------------------------------------------------------------------------
        // face (2D) and hex (3D)
        if (this->mk == Dim) {

            // dual of a d-cell is the set of midpoint of the 2*d adjacent d-1 cells
            // these will also be axis aligned neighbors (dx = half hex width)
            this->add_nbrs_axis_aligned(this->mp, M, dx);
            return;
        }

        // ------------------------------------------------------------------------------
        // edge
        if (this->mk == 1) {

            // the orientation of the edge can be computed by using the
            // max scale that its midpoint 'p' lies on
            // e.g., for a x-edge, p[0] will be 1 scale higher (more refined) than p[1]
            TypeScale edge_orient = (TypeScale)scale.argmax();

            if (2 == Dim) {
                // in 2D, the dual of an edge is the edge created by connecting the
                // midpoints of the 2 adjacent faces
                // i.e., an edge in the orthogonal direction
                // e.g., if p is an x-edge, dual will be a y-edge, and vice versa
                this->add_corners_edge(this->mp, M, dx, 1-edge_orient);
            }
            else {
                // in 3D, the dual of an edge is the edge created by connecting the
                // midpoints of 4 adjacent faces and 4 adjacent hexes (in 3D)
                this->add_nbrs_edge(this->mp, M, dx, edge_orient);
            }
            return;
        }

        // ------------------------------------------------------------------------------
        // face (3D)
        if (this->mk == 2) {

            // in 3D, dual of a face is the set of the midpoints of the 4 adjacent edges and 2 adjacent hexes

            // the orientation of the face (in 3D) can be computed by using the
            // min scale that its midpoint 'p' lies on
            // e.g., for a xy-face, p[2] will be 1 scale lower (less refined) than p[0] and p[1]
            TypeScale face_orient = (TypeScale)scale.argmin();
            this->add_nbrs_face(this->mp, M, dx, face_orient);
            return;
        }

        AMM_error_invalid_arg(true, "DualIterator(%d, %d) -- Invalid dimension!\n", this->mp, (int)this->mk);
    }
};

/// ----------------------------------------------------------------------------
/// Iterate through the dual cells of the primal cells of p, a k-cell
/// but ignore the dual of p
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class primal_dual_cell_iterator : public octree_cell_iterator<Dim> {

    static_assert((Dim == 2 || Dim == 3), "DualIterator works for 2D only!");

    using TypeVec   = typename octree_cell_iterator<Dim>::TypeVec;
    using TypeDir   = typename octree_cell_iterator<Dim>::TypeDir;
    using octutils = typename octree_cell_iterator<Dim>::octutils;

public:
    //! primal of dual of a k-cell centered at 'p'
    primal_dual_cell_iterator(const TypeVec &p, const TypeDim k, const TypeScale MaxLevels, const TypeScale scale = 0)
        : octree_cell_iterator<Dim>(p, k, MaxLevels, scale) {

        // dimensionality of cell depends upon Dim
        AMM_error_invalid_arg(this->mk > Dim,
                               " DualIterator<D=%d, L=%d>(%d, %d) -- Invalid dimension of p!\n",
                               int(Dim), int(this->mMaxLevels), this->mp, (int)this->mk);

        this->initialize();
    }

private:
    void initialize() {

        const TypeCoord M = AMM_ldim(this->mMaxLevels);
        const Vec<Dim, TypeScale> scale = octutils::get_vertexScales(this->mp, this->mMaxLevels);
        const TypeCoord dx = AMM_ldx(scale.max(), this->mMaxLevels);

        // ------------------------------------------------------------------------------
        // edge
        if (this->mk == 1) {

            // the orientation of the edge can be computed by using the
            // max scale that its midpoint 'p' lies on
            // e.g., for a x-edge, p[0] will be 1 scale higher (more refined) than p[1]
            TypeScale edge_orient = (TypeScale)scale.argmax();
            static constexpr uint8_t nnbrs = AMM_pow2(Dim);
            this->mOutput.reserve(nnbrs);
            if (2 == Dim) {
                // in 2D, the dual of an edge is the edge created by connecting the
                // midpoints of the 2 adjacent faces
                // i.e., an edge in the orthogonal direction
                // e.g., if p is an x-edge, dual will be a y-edge, and vice versa
                switch(edge_orient) {
                case 0:
                    this->add_point(this->mp, M, dx, TypeDir(-2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+2, 1));
                    this->add_point(this->mp, M, dx, TypeDir(-2, 1));
                    break;

                case 1:
                    this->add_point(this->mp, M, dx, TypeDir(-1,-2));
                    this->add_point(this->mp, M, dx, TypeDir(-1,+2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,+2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,-2));
                    break;
                }
            }
            else {
                // in 3D, the dual of an edge is the edge created by connecting the
                // midpoints of 4 adjacent faces and 4 adjacent hexes (in 3D)
                switch(edge_orient) {
                case 0:
                    this->add_point(this->mp, M, dx, TypeDir(-2,-1,-1));
                    this->add_point(this->mp, M, dx, TypeDir(-2,+1,-1));
                    this->add_point(this->mp, M, dx, TypeDir(-2,+1,+1));
                    this->add_point(this->mp, M, dx, TypeDir(-2,-1,+1));
                    this->add_point(this->mp, M, dx, TypeDir(+2,-1,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+2,+1,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+2,+1,+1));
                    this->add_point(this->mp, M, dx, TypeDir(+2,-1,+1));
                    break;

                case 1:
                    this->add_point(this->mp, M, dx, TypeDir(-1,-2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+1,-2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+1,-2,+1));
                    this->add_point(this->mp, M, dx, TypeDir(-1,-2,+1));
                    this->add_point(this->mp, M, dx, TypeDir(-1,+2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+1,+2,-1));
                    this->add_point(this->mp, M, dx, TypeDir(+1,+2,+1));
                    this->add_point(this->mp, M, dx, TypeDir(-1,+2,+1));
                    break;

                 case 2:
                    this->add_point(this->mp, M, dx, TypeDir(-1,-1,-2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,-1,-2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,+1,-2));
                    this->add_point(this->mp, M, dx, TypeDir(-1,+1,-2));
                    this->add_point(this->mp, M, dx, TypeDir(-1,-1,+2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,-1,+2));
                    this->add_point(this->mp, M, dx, TypeDir(+1,+1,+2));
                    this->add_point(this->mp, M, dx, TypeDir(-1,+1,+2));
                    break;
                }
            }
            return;
        }

        // ------------------------------------------------------------------------------
        // face (3D)
        if (this->mk == 2) {

            // in 3D, dual of a face is the set of the midpoints of the 4 adjacent edges and 2 adjacent hexes

            // the orientation of the face (in 3D) can be computed by using the
            // min scale that its midpoint 'p' lies on
            // e.g., for a xy-face, p[2] will be 1 scale lower (less refined) than p[0] and p[1]
            TypeScale face_orient = (TypeScale)scale.argmin();
            this->mOutput.reserve(16);
            switch(face_orient) {
            case 0:
                this->add_point(this->mp, M, dx, TypeDir( 1,-2,-2));
                this->add_point(this->mp, M, dx, TypeDir( 1,-2, 0));
                this->add_point(this->mp, M, dx, TypeDir( 1,-2, 2));

                this->add_point(this->mp, M, dx, TypeDir( 1, 0,-2));
                this->add_point(this->mp, M, dx, TypeDir( 1, 0, 2));

                this->add_point(this->mp, M, dx, TypeDir( 1, 2,-2));
                this->add_point(this->mp, M, dx, TypeDir( 1, 2, 0));
                this->add_point(this->mp, M, dx, TypeDir( 1, 2, 2));

                this->add_point(this->mp, M, dx, TypeDir(-1,-2,-2));
                this->add_point(this->mp, M, dx, TypeDir(-1,-2, 0));
                this->add_point(this->mp, M, dx, TypeDir(-1,-2, 2));

                this->add_point(this->mp, M, dx, TypeDir(-1, 0,-2));
                this->add_point(this->mp, M, dx, TypeDir(-1, 0, 2));

                this->add_point(this->mp, M, dx, TypeDir(-1, 2,-2));
                this->add_point(this->mp, M, dx, TypeDir(-1, 2, 0));
                this->add_point(this->mp, M, dx, TypeDir(-1, 2, 2));
                break;

            case 1:
                this->add_point(this->mp, M, dx, TypeDir(-2, 1,-2));
                this->add_point(this->mp, M, dx, TypeDir(-2, 1, 0));
                this->add_point(this->mp, M, dx, TypeDir(-2, 1, 2));

                this->add_point(this->mp, M, dx, TypeDir( 0, 1,-2));
                this->add_point(this->mp, M, dx, TypeDir( 0, 1, 2));

                this->add_point(this->mp, M, dx, TypeDir( 2, 1,-2));
                this->add_point(this->mp, M, dx, TypeDir( 2, 1, 0));
                this->add_point(this->mp, M, dx, TypeDir( 2, 1, 2));

                this->add_point(this->mp, M, dx, TypeDir(-2,-1,-2));
                this->add_point(this->mp, M, dx, TypeDir(-2,-1, 0));
                this->add_point(this->mp, M, dx, TypeDir(-2,-1, 2));

                this->add_point(this->mp, M, dx, TypeDir( 0,-1,-2));
                this->add_point(this->mp, M, dx, TypeDir( 0,-1, 2));

                this->add_point(this->mp, M, dx, TypeDir( 2,-1,-2));
                this->add_point(this->mp, M, dx, TypeDir( 2,-1, 0));
                this->add_point(this->mp, M, dx, TypeDir( 2,-1, 2));
                break;

             case 2:
                this->add_point(this->mp, M, dx, TypeDir(-2,-2, 1));
                this->add_point(this->mp, M, dx, TypeDir(-2, 0, 1));
                this->add_point(this->mp, M, dx, TypeDir(-2, 2, 1));

                this->add_point(this->mp, M, dx, TypeDir( 0,-2, 1));
                this->add_point(this->mp, M, dx, TypeDir( 0, 2, 1));

                this->add_point(this->mp, M, dx, TypeDir( 2,-2, 1));
                this->add_point(this->mp, M, dx, TypeDir( 2, 0, 1));
                this->add_point(this->mp, M, dx, TypeDir( 2, 2, 1));

                this->add_point(this->mp, M, dx, TypeDir(-2,-2,-1));
                this->add_point(this->mp, M, dx, TypeDir(-2, 0,-1));
                this->add_point(this->mp, M, dx, TypeDir(-2, 2,-1));

                this->add_point(this->mp, M, dx, TypeDir( 0,-2,-1));
                this->add_point(this->mp, M, dx, TypeDir( 0, 2,-1));

                this->add_point(this->mp, M, dx, TypeDir( 2,-2,-1));
                this->add_point(this->mp, M, dx, TypeDir( 2, 0,-1));
                this->add_point(this->mp, M, dx, TypeDir( 2, 2,-1));
                break;
            }
            return;
        }

        AMM_error_invalid_arg(true, "PrimalDualIterator(%d, %d) -- Invalid dimension!\n", this->mp, (int)this->mk);
    }
};

/// ----------------------------------------------------------------------------
/// Iterate through the neighboring 0-cells of of p, a 0-cell
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class neighboring_cell_iterator : public octree_cell_iterator<Dim> {

    static_assert((Dim == 2 || Dim == 3), "neighboring_cell_iterator works for 2D and 3D only!");

    using TypeVec   = typename octree_cell_iterator<Dim>::TypeVec;
    using TypeDir   = typename octree_cell_iterator<Dim>::TypeDir;

    const TypeDim mAcrossDim;

public:
    neighboring_cell_iterator(const TypeVec &p, const TypeDim acrossDim, const TypeScale MaxLevels, const TypeScale scale = 0)
        : octree_cell_iterator<Dim>(p, 0, MaxLevels, scale), mAcrossDim(acrossDim) {

        AMM_error_invalid_arg(mAcrossDim < 1 || mAcrossDim > Dim,
                               " NeighborIterator<D=%d, L=%d> -- Invalid acrossDim %d!\n",
                               int(Dim), int(this->mMaxLevels), mAcrossDim);

        this->initialize();
    }

private:
    void initialize() {

        const TypeCoord M = AMM_ldim(this->mMaxLevels);
        TypeCoord dx = AMM_ldx(this->mScale, this->mMaxLevels);

        switch(mAcrossDim) {

            //! across_1 neighbors are axis aligned neighbors (only one coordinate changes)
            case 1:
                    this->add_nbrs_axis_aligned(this->mp, M, dx);
                    return;

            //! across_2 neighbors are diagonal neighbors (two coordinates change)
            case 2:
                    if (2 == Dim) { // 4 (across vertices)
                        this->add_nbrs_diagonal(this->mp, M, dx);
                    }
                    else {          // 12 (across edges)
                        this->add_nbrs_across_edge(this->mp, M, dx);
                    }
                    return;

             case 3:
                    if (3 == Dim) { // 8 (across vertices)
                        this->add_nbrs_diagonal(this->mp, M, dx);
                    }
                    return;
        }

        AMM_error_invalid_arg(true, "NeighborIterator(%d, %d) -- Invalid dimension!\n", this->mp, (int)this->mAcrossDim);
    }
};


/// ----------------------------------------------------------------------------
/// Iterate through all neigboring cells
/// ----------------------------------------------------------------------------
template <TypeDim Dim>
class neighboring_cell_all_iterator : public octree_cell_iterator<Dim> {

    static_assert((Dim == 2 || Dim == 3), "neighboring_cell_all_iterator works for 2D and 3D only!");

    using TypeVec   = typename octree_cell_iterator<Dim>::TypeVec;
    using TypeDir   = typename octree_cell_iterator<Dim>::TypeDir;

public:
    neighboring_cell_all_iterator(const TypeVec &p, const TypeScale maxLevels, const TypeScale scale)
        : octree_cell_iterator<Dim>(p, 0, maxLevels, scale) {
        this->initialize();
    }

private:
    void initialize() {

        const TypeCoord M = AMM_ldim(this->mMaxLevels);
        const TypeCoord dx = AMM_ldx(this->mScale, this->mMaxLevels);

        if (2 == Dim) {
            this->mOutput.reserve(8);
            this->add_nbrs_axis_aligned(this->mp, M, dx);   // 4 (axis aligned)
            this->add_nbrs_diagonal(this->mp, M, dx);       // 4 (across vertices)
        }
        else {

            this->mOutput.reserve(26);
            this->add_nbrs_axis_aligned(this->mp, M, dx);   // 6 (axis aligned)
            this->add_nbrs_diagonal(this->mp, M, dx);       // 8 (across vertices)
            this->add_nbrs_across_edge(this->mp, M, dx);    // 12 (across edges)
        }
    }
};

/// ----------------------------------------------------------------------------
/// ----------------------------------------------------------------------------
}}   // end of namespace
#endif
