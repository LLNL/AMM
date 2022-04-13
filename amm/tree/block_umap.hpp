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
#ifndef AMM_BLOCK_UMAP_H
#define AMM_BLOCK_UMAP_H

//! ----------------------------------------------------------------------------

#include <type_traits>
#include "containers/unordered_map.hpp"
#include "tree/block_abstract.hpp"

namespace amm {
//! ----------------------------------------------------------------------------
//! A block that uses an unordered map to represent data
//! ----------------------------------------------------------------------------

template <typename TypeValue>
class block_umap : public amm::block<TypeValue>,
                   public amm::unordered_map<TypeLocalIdx, TypeValue> {

    using umap = amm::unordered_map<TypeLocalIdx, TypeValue>;

public:
    // ----------------------------------------------------------------------------
    inline size_t size() const {                                  return umap::size();      }
    inline void clear() {                                         umap::clear();    }

    // ----------------------------------------------------------------------------
    inline bool contains(const TypeLocalIdx i) const {            return umap::contains(i); }
    inline TypeValue get(const TypeLocalIdx i) const {            return umap::get(i);      }
    inline TypeValue get(const TypeLocalIdx i, bool &exists) const { return umap::get(i, exists); }
    inline void add(const TypeLocalIdx i, const TypeValue v) {    return umap::add(i,v);    }
    inline void set(const TypeLocalIdx i, const TypeValue v) {    return umap::set(i,v);    }
    inline TypePrecision precision() const {                      return sizeof(TypeValue);    }

    // ----------------------------------------------------------------------------
    inline void update(const std::unordered_map<TypeLocalIdx, TypeValue> &vals, bool do_add) {
        if (do_add) {
            for(auto i = vals.begin(); i != vals.end(); i++)
                this->add(i->first, i->second);
        }
        else {
            for(auto i = vals.begin(); i != vals.end(); i++)
                this->set(i->first, i->second);
        }
    }

    // ----------------------------------------------------------------------------
    inline float memory_in_kb() const {
        return umap::memory_in_kb();
    }
};

//! ----------------------------------------------------------------------------
}   // end of namespace

#endif
