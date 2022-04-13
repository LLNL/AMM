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
#ifndef AMM_CACHE_MANAGER_H
#define AMM_CACHE_MANAGER_H

//! ----------------------------------------------------------------------------
#include <climits>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <unordered_set>

#include "types/dtypes.hpp"


//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! A container to manage cache
//! ----------------------------------------------------------------------------
template<typename key_type>
class cache_manager {

    static_assert(std::is_unsigned<key_type>::value, "cache_manager is designed for unsigned data types!");

    //! can store the data in two differnt formats!
    std::unordered_set<key_type> m_sparse;
    std::vector<bool> m_dense;

    size_t m_dsize = 0;
    bool m_is_dense = false;

    //bool m_keep_sparse = false;
    //size_t m_breakeven = 0;

    //! ----------------------------------------------------------------------------
    //! interconvert between the formats
    //! ----------------------------------------------------------------------------
    inline
    void
    dense_to_sparse() {
        size_t i = 0;
        for(auto iter = m_dense.begin(); iter != m_dense.end(); ++iter, ++i) {
            if (*iter)
                m_sparse.insert(i);
        }
        m_dense.clear();
        m_is_dense = false;
    }


    inline
    void
    sparse_to_dense() {
        m_dense.resize(m_dsize, false);
        for(auto iter = m_sparse.begin(); iter != m_sparse.end(); ++iter) {
            m_dense[*iter] = true;
        }
        m_sparse.clear();
        m_is_dense = true;
    }

public:
    //! ----------------------------------------------------------------------------
    inline
    void
    init(const size_t dsize, bool make_dense) {
                                              //keep_sparse) {
        m_dsize = dsize;
        m_is_dense = make_dense;

        //m_is_dense = false;
        //m_keep_sparse = keep_sparse;

        // number of values where switch from sparse to dense
        // unless, keep_sparse is true
        //m_breakeven = dsize / (CHAR_BIT*sizeof(key_type)) / 8;
    }


    inline
    size_t
    size() const {
        if (m_is_dense) {
            return std::count_if(m_dense.begin(), m_dense.end(), [](const bool v) { return v; });
        }
        else {
            return m_sparse.size();
        }
    }


    inline
    bool
    contains(const key_type &_) const {
        if (m_is_dense) {
            return (m_dense.empty()) ? false : m_dense[_];
        }
        else {
            return m_sparse.find(_) != m_sparse.end();
        }
    }


    inline
    void
    set(const key_type &_) {
        if (m_is_dense) {
            if (m_dense.empty()) {  // getting used for the first time!
                m_dense.resize(m_dsize, false);
            }
            m_dense[_] = true;
        }
        /*else if (m_sparse.size() >= m_breakeven && !m_keep_sparse){
            sparse_to_dense();
            set(_);
        }*/
        else {
            m_sparse.insert(_);
        }
    }


    inline
    void
    clear() {
        if (m_is_dense) {
            std::fill(m_dense.begin(), m_dense.end(), false);
        }
        else {
            m_sparse.clear();
        }
    }


    inline
    float
    memory_in_kb() const {
        float nbytes = 0.0;
        if (m_is_dense) {
            nbytes = float(m_dense.size()) / float(CHAR_BIT);
        }
        else {
            nbytes = float(m_sparse.size()*sizeof(key_type));
        }
        return nbytes / 1024.0;
    }
};
}
#endif
