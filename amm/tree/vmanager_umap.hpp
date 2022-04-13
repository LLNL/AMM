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
#ifndef AMM_VERTEX_MANAGER_UMAP_H
#define AMM_VERTEX_MANAGER_UMAP_H

//! ------------------------------------------------------------------------------------
#include <string>
#include <iostream>
#include <numeric>
#include <limits>
#include <iomanip>

#include "types/dtypes.hpp"
#include "containers/vec.hpp"
#include "containers/unordered_map.hpp"
#include "tree/vmanager_abstract.hpp"

namespace amm {

//! ------------------------------------------------------------------------------------
//! VertexManagerUnsortedMap uses an unsorted map to manage vertices and their values.
//!
//! @tparam Dim: dimensionality of vertices (2 or 3)
//! @tparam T: data type for values
//!
//! ------------------------------------------------------------------------------------
template<TypeDim Dim, typename TypeValue>
class vertex_manager_umap : public amm::vertex_manager<Dim, TypeValue>,
                            public amm::unordered_map<TypeIndex, TypeValue> {

    // store keys as hashed values (uint64)
    using umap = amm::unordered_map<TypeIndex, TypeValue>;
    using vmanager = amm::vertex_manager<Dim, TypeValue>;
    using TypeVertex = typename vmanager::TypeVertex;

    // ------------------------------------------------------------------------
    // compute the hash (for now, it is same as row-major idx)
    using TypeHash = TypeIndex;
    inline TypeHash hash(const TypeVertex &_) const {  return this->p2idx(_); }
    inline TypeVertex unhash(const TypeIndex _) const { return this->idx2p(_); }

public:
    // ---------------------------------------------------------------------------------
    // basics
    inline void clear() {                                   umap::clear();                    }
    inline size_t size() const {                            return umap::size();              }

    // ---------------------------------------------------------------------------------
    // index api
    inline bool contains(const TypeIndex &_) const {               return umap::contains(_);      }
    inline TypeValue get(const TypeIndex &_) const {               return umap::get(_);           }
    inline TypeValue get(const TypeIndex &_, bool &exists) const { return umap::get(_, exists);   }
    inline void add(const TypeIndex &_, const TypeValue val) {     return umap::add(_, val);      }
    inline void set(const TypeIndex &_, const TypeValue val) {     return umap::set(_, val);      }
    inline TypePrecision precision(const TypeIndex &_) const {     return umap::contains(_) ?
                                                                                    sizeof(TypeValue) : 0;}

    // ---------------------------------------------------------------------------------
    // vertex api
    inline bool contains(const TypeVertex &_) const {               return umap::contains(hash(_));      }
    inline TypeValue get(const TypeVertex &_) const {               return umap::get(hash(_));           }
    inline TypeValue get(const TypeVertex &_, bool &exists) const { return umap::get(hash(_), exists);   }
    inline void add(const TypeVertex &_, const TypeValue val) {     return umap::add(hash(_), val);      }
    inline void set(const TypeVertex &_, const TypeValue val) {     return umap::set(hash(_), val);      }
    inline TypePrecision precision(const TypeVertex &_) const {     return umap::contains(hash(_)) ?
                                                                                    sizeof(TypeValue) : 0;}

    inline float  memory_in_kb() const {
        return umap::memory_in_kb();
    }

    // print
    inline void print(const std::string &label = "vertices", const bool sorted=true) const {

        std::cout << " > printing " << label << " [= "<< this->size() << "]\n";
        if (!sorted) {
            for(auto iter = umap::begin(); iter != umap::end(); ++iter) {
                std::cout << " ["<<TypeHash(iter->first)<<"] : " << unhash(iter->first)
                          << std::fixed << std::setprecision (std::numeric_limits<TypeValue>::digits10 + 1)
                          << " = " << iter->second << std::endl;
            }
        }
        else {
            auto smap = umap::as_sorted_vector();
            for(auto iter = smap.begin(); iter != smap.end(); ++iter){
                std::cout << " ["<<TypeHash(iter->first)<<"] : " << unhash(iter->first)
                          << std::fixed << std::setprecision (std::numeric_limits<TypeValue>::digits10 + 1)
                          << " = " << iter->second << std::endl;
            }
        }
    }

    // -----------------------------------------------------------------------------
    inline void
    copy_to(vertex_manager_umap<Dim, TypeValue> &to) const {
        for(auto iter = umap::begin(); iter != umap::end(); ++iter)
            to.add(iter->first, iter->second);
    }
    inline void
    copy_from(vertex_manager_umap<Dim, TypeValue> &from) {
        for(auto iter = from.begin(); iter != from.end(); ++iter)
            add(iter->first, iter->second);
    }

    // -----------------------------------------------------------------------------
};
}   // end of namespace
#endif
