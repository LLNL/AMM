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
#ifndef AMM_USET_H
#define AMM_USET_H

//! ----------------------------------------------------------------------------
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>

namespace amm {

//! ----------------------------------------------------------------------------
//! Unordered set of unsigned integer data types
//!
//! @tparam key_type: data type
//!
//! ----------------------------------------------------------------------------
template <typename key_type>
class unordered_set : public std::unordered_set<key_type> {

    static_assert(std::is_unsigned<key_type>::value, "unordered_set is designed for unsigned data types!");
protected:
    typedef std::unordered_set<key_type> container_type;

public:
    inline
    bool
    contains(const key_type &_) const {
        return this->find(_) != this->end();
    }


    std::vector<key_type>
    as_sorted_vector() const {
        std::vector<key_type> _;
        _.reserve(this->size());
        std::for_each(this->begin(), this->end(),
                      [&_](auto iter){_.push_back(iter);});

        std::sort(_.begin(), _.end());
        return _;
    }


    inline
    float
    memory_in_kb() const {
        return float(this->size()*sizeof(key_type))/1024.0;
    }


    /*//! const iterator
    using const_iterator = typename container_type::const_iterator;
    inline const_iterator begin() const {       return container_type::begin();     }
    inline const_iterator end() const {         return container_type::end();       }*/
};
}   // end of namespace


template <typename key_type, typename val_type>
std::ostream &operator<<(::std::ostream &os, const amm::unordered_set<key_type> &uset){

    for(auto _ = uset->begin(); _ != uset->end(); ++_) {
        os << size_t(_) << std::endl;
    }
    return os;
}
//! ----------------------------------------------------------------------------
#endif
