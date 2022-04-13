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
#ifndef AMM_UMAP_H
#define AMM_UMAP_H

//! ----------------------------------------------------------------------------
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include "macros.hpp"
#include "utils/utils.hpp"

namespace amm {

//! ----------------------------------------------------------------------------
//! Unordered map that uses an unsigned integer as key.
//! Value could be any data type or class
//!
//! @tparam key_type: data type for keys
//! @tparam val_type: data type for values
//!
//! ----------------------------------------------------------------------------
template <typename key_type, typename val_type>
class unordered_map : public std::unordered_map<key_type, val_type> {

    static_assert(std::is_unsigned<key_type>::value, "unordered_map is designed for unsigned data types!");

protected:
    typedef std::unordered_map<key_type, val_type> container_type;

public:
    typedef std::vector<std::pair<key_type,val_type>> sorted_container_type;

    inline
    bool
    contains(const key_type &_) const {
        return this->find(_) != this->end();
    }


    inline
    val_type
    get(const key_type &_) const {
        auto i = this->find(_);
        return (i != this->end()) ? i->second : AMM_missing_vertex(val_type);
    }


    inline
    val_type
    get(const key_type &_, bool &exists) const {
        auto i = this->find(_);
        exists = (i != this->end());
        return exists ? i->second : AMM_missing_vertex(val_type);
    }


    inline
    void
    insert(const key_type &_, const val_type &v) {
        container_type::emplace(_, v);
    }


    inline
    void
    set(const key_type &_, const val_type &v) {
        this->operator[](_) = v;
    }


    inline
    void
    add(const key_type &_, const val_type &v) {
        this->operator[](_) += v;
    }


    sorted_container_type
    as_sorted_vector() const {
        sorted_container_type _;
        _.reserve(this->size());
        std::for_each(this->begin(), this->end(),
                      [&_](auto iter){_.push_back(iter);});

        std::sort(_.begin(), _.end());
        return _;
    }


    inline
    float
    memory_in_kb() const {
        return float(this->size()*(sizeof(key_type)+sizeof(val_type)))/1024.0;
    }

    /*//! const iterator
    using const_iterator = typename container_type::const_iterator;
    inline const_iterator begin() const {   return container_type::begin(); }
    inline const_iterator end() const {     return container_type::end();   }*/
};
}    // end of namespace


template <typename key_type, typename val_type>
std::ostream &operator<<(::std::ostream &os, const amm::unordered_map<key_type, val_type> &umap){

    for(auto i = umap->begin(); i != umap->end(); ++i) {
        os << size_t(i->first) << " = " << i->second << std::endl;
    }
    return os;
}
//! ----------------------------------------------------------------------------
#endif
