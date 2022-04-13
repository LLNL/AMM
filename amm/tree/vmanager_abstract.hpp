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
#ifndef AMM_VERTEX_MANAGER_ABSTRACT_H
#define AMM_VERTEX_MANAGER_ABSTRACT_H

//! ------------------------------------------------------------------------------------
#include "types/dtypes.hpp"
#include "containers/vec.hpp"
#include "tree/octree_utils.hpp"

namespace amm {

//! ------------------------------------------------------------------------------------
//! A Vertex Manager for AMM (an abstract class)
//!
//! VertexManager defines the API to manage vertices and their values.
//!
//! @tparam Dim: dimensionality of vertices (2 or 3)
//! @tparam T: data type for values
//!
//! ------------------------------------------------------------------------------------
template<TypeDim Dim, typename TypeValue>
class vertex_manager {
    static_assert((Dim == 2 || Dim == 3), "vertex_manager works for 2D and 3D only!");

protected:
    using TypeVertex   = Vec<Dim, TypeCoord>;

    // dimensions of the domain (needed to convert vertex to row-major indices)
    TypeVertex mDims;

    // constructor
    vertex_manager() : mDims(TypeCoord(0)) {}


    static inline
    bool
    validate_invalid(const TypeVertex &p, const TypeValue v, const std::string &caller) {

#ifdef AMM_DEBUG_VMANAGER_INVALIDS
        if (2 == Dim) {
            AMM_error_invalid_arg(AMM_is_missing_vertex(v),
                                  "(%s): received invalid value! (%u,%u) = %f\n",
                                  caller.c_str(), p[0],p[1], v);
        }
        else {
            AMM_error_invalid_arg(AMM_is_missing_vertex(v),
                                  "(%s): received invalid value! (%u,%u,%u) = %f\n",
                                  caller.c_str(), p[0],p[1],p[2], v);
        }
#endif
        return true;
    }

    static inline
    bool
    validate_zero(const TypeVertex &p, const TypeValue v, const std::string &caller) {

#ifdef AMM_DEBUG_VMANAGER_ZEROS
        if (2 == Dim) {
            AMM_error_invalid_arg(AMM_is_zero(v),
                                  "(%s): received invalid value! (%u,%u) = %f\n",
                                  caller.c_str(), p[0],p[1], v);
        }
        else {
            AMM_error_invalid_arg(AMM_is_zero(v),
                                  "(%s): received invalid value! (%u,%u,%u) = %f\n",
                                  caller.c_str(), p[0],p[1],p[2], v);
        }
#endif
        return true;
    }


    using octutils = amm::octree_utils<Dim>;
    inline size_t maxv() const {                        return octutils::product(mDims);  }
    inline TypeIndex p2idx(const TypeVertex &_) const { return octutils::p2idx(_, mDims); }
    inline TypeVertex idx2p(const TypeIndex _) const {  return octutils::idx2p(_, mDims); }

public:
    inline void init(const TypeVertex &_) { mDims = _; }

    // basics
    virtual void clear() = 0;
    virtual size_t size() const = 0;

    // index api
    virtual bool contains(const TypeIndex &) const = 0;
    virtual TypeValue get(const TypeIndex &) const = 0;
    virtual TypeValue get(const TypeIndex &, bool &) const = 0;
    virtual void add(const TypeIndex &, const TypeValue) = 0;
    virtual void set(const TypeIndex &, const TypeValue) = 0;
    virtual TypePrecision precision(const TypeIndex &) const = 0;

    // vertex api
    virtual bool contains(const TypeVertex &) const = 0;
    virtual TypeValue get(const TypeVertex &) const = 0;
    virtual TypeValue get(const TypeVertex &, bool &) const = 0;
    virtual void add(const TypeVertex &, const TypeValue) = 0;
    virtual void set(const TypeVertex &, const TypeValue) = 0;
    virtual TypePrecision precision(const TypeVertex &) const = 0;


    virtual float  memory_in_kb() const = 0;

    // print
    inline void print(const std::string &label, const bool sorted=false) const;
};

//! ---------------------------------------------------------------------------
//! ---------------------------------------------------------------------------

}   // end of namespace
#endif
