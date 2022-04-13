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
#ifndef AMM_AMM_H
#define AMM_AMM_H

#include <vector>
#include "types/dtypes.hpp"
#include "tree/amtree.hpp"



namespace amm {


//! -------------------------------------------------------------------------------------
//! AMM: Adaptive Multilinear Mesh
//! 
//! An adaptive representation of a sparse scalar field.
//! AMM is internally repesented an adaptive tree, and provides a simple API for
//! mesh traversal. AMM can be constructed using provided MeshCreators.
//!
//! -------------------------------------------------------------------------------------
template <typename TypeValue, TypeDim Dim, TypeScale MaxLevels>
class AMM : public amm::amtree::AMTree<TypeValue, Dim, MaxLevels> {

    static_assert((Dim == 2 || Dim == 3), "AMM is designed for 2D and 3D only!");

public:
    using ATree = amm::amtree::AMTree<TypeValue, Dim, MaxLevels>;

    using TypeVertexId        = TypeIndex;
    using TypeCellId          = typename ATree::TypeLocCode;

    using TypeVertex          = typename ATree::TypeVertex;
    using TypeLocCode         = typename ATree::TypeLocCode;

    using DualIterator        = typename ATree::DualIterator;
    using PrimalIterator      = typename ATree::PrimalIterator;
    using PrimalDualIterator  = typename ATree::PrimalDualIterator;
    using NeighborAllIterator = typename ATree::NeighborAllIterator;


    using OctUtils       = typename ATree::OctUtils;
    using ListFinalCells = typename ATree::ListFinalCells;
    using ListFinalVerts = typename ATree::ListFinalVerts;

private:
    const TypeVertex mDomainSize;       // size of the given data
    const TypeVertex mDomainBox1;       // bounding box of the given data

    //! --------------------------------------------------------------------------------
    //! initialize an empty mesh
    //! --------------------------------------------------------------------------------
    void init() {
        AMM_log_info << "Creating AMM<Dim="<<int(Dim)<<", L="<<int(MaxLevels)<<"> : "
                     << " size = " << this->dsize() << "\n";
    }

public:

    inline const TypeVertex& dsize() const {   return mDomainSize;  }
    inline const TypeVertex& dbox1() const {   return mDomainBox1;  }

    //! --------------------------------------------------------------------------------
    //! constructor and destructor
    //! --------------------------------------------------------------------------------
    AMM(const size_t size[], bool allow_rectangular, bool allow_vacuum) :
                ATree(size, allow_rectangular, allow_vacuum),
                mDomainSize(size),
                mDomainBox1(OctUtils::os2bb(this->origin(), mDomainSize)) {
        init();
    }


    AMM(const TypeVertex &size, bool allow_rectangular, bool allow_vacuum) :
                    ATree(size, allow_rectangular, allow_vacuum),
                    mDomainSize(size),
                    mDomainBox1(OctUtils::os2bb(this->origin(), mDomainSize)) {
        init();
    }


    ~AMM() {}


    //! --------------------------------------------------------------------------------
    //! whether a vertex is inside the domain
    //! --------------------------------------------------------------------------------
    bool is_valid_vertex(const TypeVertex &_) const {
        return OctUtils::contains_os(_, this->origin(), mDomainSize);
    }


    //! --------------------------------------------------------------------------------
    //! reconstruct the function onto a regular grid
    //! --------------------------------------------------------------------------------
    inline
    bool
    reconstruct(const TypeVertex &dorigin, const TypeVertex &dsize,
                std::vector<TypeValue> &func) const {
        return ATree::reconstruct(dorigin, dsize, func);
    }


    inline
    bool
    reconstruct(const TypeVertex &dsize,
                std::vector<TypeValue> &func) const {
        return ATree::reconstruct(this->origin(), dsize, func);
    }


    inline
    bool
    reconstruct(std::vector<TypeValue> &func) const {
        return ATree::reconstruct(this->origin(), mDomainSize, func);
    }

    //! --------------------------------------------------------------------------------
    //! reconstruct the mesh as an unstructued grid
    //! --------------------------------------------------------------------------------
    inline
    bool
    reconstruct(ListFinalVerts &nvertices, ListFinalCells &ncells) const {
        return ATree::reconstruct(this->origin(), mDomainSize, nvertices, ncells);
    }


    //! --------------------------------------------------------------------------------
    //! finalize the mesh!
    //! --------------------------------------------------------------------------------
    void
    finalize() {
        this->finalize_nodes_and_vertices();
        this->finalize_improper_nodes();
        this->finalize_boundary(mDomainBox1);
    }


#if 0
    //! --------------------------------------------------------------------------------
    //! iterators as containers
    //! --------------------------------------------------------------------------------
    class Iterator_cell : public std::vector<TypeCellId> {
    public:
        Iterator_cell(const AMM &mesh, bool sort_code) {

            this->reserve( mesh.mLeafNodes.size() + mesh.mLeaves_Improper.size());

            // first add all leaf nodes
            for(auto iter = mesh.mLeafNodes.begin(); iter != mesh.mLeafNodes.end(); iter++) {
                if (mesh.is_node_intersects(*iter, mesh.mOrigin, mesh.dBox1)) {
                    this->push_back(*iter);
                }
            }

            // now any vacuum leaves
            for(auto iter = mesh.mLeaves_Improper.begin(); iter != mesh.mLeaves_Improper.end(); iter++) {
                if (mesh.is_node_intersects(*iter, mesh.mOrigin, mesh.dBox1)) {
                    this->push_back(*iter);
                }
            }

            if (sort_code) {
                std::sort(this->begin(), this->end());
            }
        }
    };
    Iterator_cell iterator_cell(bool sort_code = false) const {
        return Iterator_cell(*this, sort_code);
    }


    //! --------------------------------------------------------------------------------
    class Iterator_vertex : public std::vector<std::pair<TypeVertexId, TypeValue>> {
    public:
        Iterator_vertex(const AMM &mesh, bool sort_rowmajor) {

            this->reserve(mesh.mVertices.size() + mesh.mVertices_Improper.size() + mesh.mVertices_at_bdry.size());

            // first add all normal vertices
            for(auto iter = mesh.mVertices.begin(); iter != mesh.mVertices.end(); iter++) {
                if (mesh.is_valid_vertex(iter->first))
                    this->push_back(*iter);
            }

            // next, add vacuum vertices
            for(auto iter = mesh.mVertices_Improper.begin(); iter != mesh.mVertices_Improper.end(); iter++) {
                if (mesh.is_valid_vertex(iter->first))
                    this->push_back(*iter);
            }

            // now add boundary vertices
            for(auto iter = mesh.mVertices_at_bdry.begin(); iter != mesh.mVertices_at_bdry.end(); iter++) {
                this->push_back(*iter);
            }

            if (sort_rowmajor) {
                std::sort(this->begin(), this->end());
            }
        }
    };
    Iterator_vertex iterator_vertex(bool sort_rowmajor = false) const {
        return Iterator_vertex(*this, sort_rowmajor);
    }


    //! --------------------------------------------------------------------------------
    class Iterator_cellvertex : public std::vector<TypeVertexId> {
    public:
        Iterator_cellvertex(const AMM &mesh, const TypeCellId &cell_id, bool sort_ccw) {

            static TypeVertex norigin, nsize;
            //static std::vector<TypeVertexId> ncorners (ATree::snCorners);

            //! get the bounds of this node
            mesh.get_node_bounds(cell_id, norigin, nsize);

            this->resize(ATree::snCorners);
            std::vector<TypeVertexId> *_ (this);
            mesh.get_node_corners(norigin, nsize, *_);

            // return vertices in CCW order
            if (sort_ccw) {
                std::swap(this->at(2),this->at(3));
                if (3 == Dim) {
                    std::swap(this->at(6),this->at(7));
                }
            }
        }
    };
    Iterator_cellvertex iterator_cellvertex(const TypeCellId &cell_id, bool sort_ccw = true) const {
        return Iterator_cellvertex(*this, cell_id, sort_ccw);
    }

    //! --------------------------------------------------------------------------------
    class Iterator_vertexcell : public std::vector<TypeCellId> {
    public:
        Iterator_vertexcell(const AMM &mesh, const TypeVertexId &vertex_id, bool sort_ccw) {

            const TypeVertex p = mesh.idx2p(vertex_id);
            std::cout << "Iterator_vertexcell("<<vertex_id<<":"<<p<<")\n";

            typename ATree::Octree::octree_cell_iterator ci;

            /*
            static TypeVertex norigin, nsize;
            //static std::vector<TypeVertexId> ncorners (ATree::snCorners);

            //! get the bounds of this node
            mesh.get_node_bounds(cell_id, norigin, nsize);

            this->resize(ATree::snCorners);
            std::vector<TypeVertexId> *_ (this);
            mesh.get_node_corners(norigin, nsize, *_);

            // return vertices in CCW order
            if (sort_ccw) {
                std::swap(this->at(2),this->at(3));
                if (3 == Dim) {
                    std::swap(this->at(6),this->at(7));
                }
            }*/
        }
    };
    Iterator_vertexcell iterator_vertexcell(const TypeVertexId &vertex_id, bool sort_ccw = true) const {
        return Iterator_vertexcell(*this, vertex_id, sort_ccw);
    }

    //! --------------------------------------------------------------------------------
    //! TODO:
    //! --------------------------------------------------------------------------------
    //! cellcell
    //! vertexcell
    //! vertexvertex?
#endif
};
}   // end of namespace
#endif
