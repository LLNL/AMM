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
#ifndef AMM_ORDERED_UNIQUE_H
#define AMM_ORDERED_UNIQUE_H

//! ----------------------------------------------------------------------------
#include <deque>
#include <unordered_set>

//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! A container to store unique ordered elements
//! ----------------------------------------------------------------------------
template<typename val_type>
class ordered_unique {

protected:
    std::deque<val_type> v;
    std::unordered_set<val_type> x;

public:
    typedef typename std::deque<val_type>::const_iterator const_iterator;
    typedef typename std::deque<val_type>::const_reference const_reference;

    ordered_unique(const size_t sz = 0) {   x.reserve(sz);          }
    ~ordered_unique() { this->clear(); }

    inline size_t size() const {            return v.size();        }
    inline bool empty() const {             return v.empty();       }
    inline void clear() {                   v.clear(); x.clear();   }
    inline void reserve(const size_t sz) {  x.reserve(sz);          }
    const_reference front() const {         return v.front();       }


    void
    pop_front() {
        x.erase(this->front());
        v.pop_front();
    }


    void
    push_back(const val_type val) {
        if (x.find(val) == x.end()) {
            v.push_back(val);
            x.insert(val);
        }
    }

    const_iterator begin() const {  return v.cbegin();  }
    const_iterator end() const {    return v.cend();    }
};
//! ----------------------------------------------------------------------------
}
#endif
