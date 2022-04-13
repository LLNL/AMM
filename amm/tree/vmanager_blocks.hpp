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
#ifndef AMM_VERTEX_MANAGER_BLOCKS_H
#define AMM_VERTEX_MANAGER_BLOCKS_H

//! ------------------------------------------------------------------------------------
#include <array>
#include <vector>
#include <limits>
#include <string>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <iomanip>

#include "types/dtypes.hpp"
#include "utils/logger.hpp"
#include "containers/vec.hpp"
#include "containers/unordered_map.hpp"
#include "containers/unordered_set.hpp"
#include "tree/block_abstract.hpp"
#include "tree/octree_utils.hpp"
#include "tree/vmanager_abstract.hpp"

namespace amm {

//! ------------------------------------------------------------------------------------
//! VertexManagerBlocks uses blocks to manage vertices and their values.
//!
//! @tparam Dim: dimensionality of vertices (2 or 3)
//! @tparam T: data type for values
//!
//! ------------------------------------------------------------------------------------
template<TypeDim Dim, typename TypeValue, typename TypeBlock>
class vertex_manager_blocks : public amm::vertex_manager<Dim, TypeValue> {

    using TypeBlockIdx  = uint32_t;

    using vmanager = typename amm::vertex_manager<Dim, TypeValue>;
    using BlockMap = amm::unordered_map<TypeBlockIdx, TypeBlock>;

    using TypeVertex = typename vmanager::TypeVertex;
    using tutils = amm::octree_utils<Dim>;

    // a block will have 64 (=2^6) vertices
        // 2D block will have 2^3 vertices (in each dim)
        // 3D block will have 2^2 vertices (in each dim)
    static constexpr uint8_t sBlockBits = (Dim==2 ? 3u : 2u);   //   3 or  2
    static constexpr uint8_t sLocalMask = (Dim==2 ? 7u : 3u);   // 111 or 11
    static constexpr uint8_t sBlockDim = (1u << sBlockBits);    //   8 or  4

    // TODO: should be static!
    const TypeVertex mSizeBlock = TypeVertex(TypeCoord(sBlockDim));

    // how many blocks in the domain
    TypeVertex mDimBlocks;

    // list of all blocks
    BlockMap blocks;

    // ---------------------------------------------------------------------------------
    // compute the block index and local index of a given vertex
    inline std::pair<TypeBlockIdx, TypeLocalIdx>
    hash(const TypeVertex &p) const {

        static TypeVertex block_count, vid_in_block;
        static std::pair<TypeBlockIdx,TypeLocalIdx> block_idx_loc_idx;

        for(TypeScale d = 0; d < Dim; d++) {
            block_count[d] = p[d] >> sBlockBits;
            vid_in_block[d] = p[d] & sLocalMask;
        }

        block_idx_loc_idx.first = tutils::p2idx(block_count, mDimBlocks);
        block_idx_loc_idx.second = tutils::p2idx(vid_in_block, mSizeBlock);
        return block_idx_loc_idx;
    }


    inline TypeVertex
    unhash(TypeBlockIdx b, TypeLocalIdx l) const {

        static TypeVertex block_origin, vid_in_block;
        static TypeVertex p;

        block_origin = tutils::idx2p(b, mDimBlocks);
        vid_in_block = tutils::idx2p(l, mSizeBlock);

        for(TypeScale d = 0; d < Dim; d++) {
            p[d] = (block_origin[d] << sBlockBits) | vid_in_block[d];
        }

        return p;
    }


    inline TypeVertex
    unhash(const std::pair<TypeBlockIdx, TypeLocalIdx> &blidx) const {
        return unhash(blidx.first, blidx.second);
    }

public:
    // ---------------------------------------------------------------------------------
    // basics
    inline void clear() {
        for(typename BlockMap::iterator iter = blocks.begin(); iter != blocks.end(); ++iter)
            iter->second.clear();
        blocks.clear();
    }
    inline size_t size() const {
        size_t sz = 0;
        for(typename BlockMap::const_iterator  iter = blocks.begin(); iter != blocks.end(); ++iter)
            sz += (iter->second).size();
        return sz;
    }

    // ---------------------------------------------------------------------------------
    // index api
    inline bool contains(const TypeIndex &_) const {               return contains(this->idx2p(_));     }
    inline TypeValue get(const TypeIndex &_) const {               return get(this->idx2p(_));          }
    inline TypeValue get(const TypeIndex &_, bool &exists) const { return get(this->idx2p(_), exists);  }
    inline void add(const TypeIndex &_, const TypeValue val) {     return add(this->idx2p(_), val);     }
    inline void set(const TypeIndex &_, const TypeValue val) {     return set(this->idx2p(_), val);     }
    inline TypePrecision precision(const TypeIndex &_) const {     return precision(this->idx2p(_));    }


    // ---------------------------------------------------------------------------------
    // vertex api
    inline
    bool
    contains(const TypeVertex &_) const {

        auto blidx = hash(_);
        auto biter = blocks.find(blidx.first);
        if (biter == blocks.end())
            return false;
        return biter->second.contains(blidx.second);
    }


    inline
    TypeValue
    get(const TypeVertex &_) const {
        bool e;     return get(_,e);
    }


    inline
    TypeValue
    get(const TypeVertex &_, bool &is_found) const {

        auto blidx = hash(_);
        auto biter = blocks.find(blidx.first);
        if (biter == blocks.end()) {
            is_found = false;
            return AMM_missing_vertex(TypeValue);
        }

        const TypeBlock &b = biter->second;
        TypeValue val = b.get(blidx.second, is_found);

        return is_found ? val : AMM_missing_vertex(TypeValue);
    }


    inline
    void
    add(const TypeVertex &_, const TypeValue v) {
        auto blidx = hash(_);
        blocks[blidx.first].add(blidx.second, v);
    }


    inline
    void
    set(const TypeVertex &_, const TypeValue v) {
        auto blidx = hash(_);
        blocks[blidx.first].set(blidx.second, v);
    }


    inline
    TypePrecision
    precision(const TypeVertex &_) const {
        auto blidx = hash(_);
        auto biter = blocks.find(blidx.first);
        return biter != blocks.end() ?
               biter->second.contains(blidx.second) ?
                        biter->second.precision() : 0 : 0;
    }


    void
    update_block(const TypeBlockIdx _, const TypeBlock &b) {
        if (b.size() > 0) {
            blocks[_].update(b, true);
        }
    }


    void
    update_block(const TypeBlockIdx _, const amm::unordered_map<TypeLocalIdx, TypeValue> &b) {
        if (b.size() > 0) {
            blocks[_].update(b, true);
        }
    }


    // ---------------------------------------------------------------------------------
    inline
    void
    init(const TypeVertex &_) {

        vmanager::init(_);
        for(TypeScale d = 0; d < Dim; d++) {
            mDimBlocks[d] = this->mDims[d] >> sBlockBits;
            if (this->mDims[d] & sLocalMask) {
                mDimBlocks[d] += 1;
            }
        }
    }

    const TypeBlock* get_block(TypeBlockIdx i) const {
       return (blocks.find(i) == blocks.end()) ? nullptr : &(blocks.at(i));
    }
    inline size_t nblocks() const {         return blocks.size();   }
    const BlockMap& get_blocks() const {    return blocks;          }

    inline size_t nblocks(std::array<size_t, 8> &blocks_at_p,
                          std::array<size_t, 8> &verts_at_p,
                          std::array<size_t, 64> &blocks_with_verts) const {

        std::fill(blocks_at_p.begin(), blocks_at_p.end(), 0);
        std::fill(verts_at_p.begin(), verts_at_p.end(), 0);
        std::fill(blocks_with_verts.begin(), blocks_with_verts.end(), 0);

        for(auto iter = blocks.begin(); iter != blocks.end(); ++iter) {

            const TypeBlock &b = iter->second;
            const TypePrecision &p = b.precision();
            const size_t &v = b.size();

            blocks_at_p[p-1] ++;
            verts_at_p[p-1] += v;
            blocks_with_verts[v-1] ++;
        }
        return blocks.size();
   }


    // ---------------------------------------------------------------------------------
    inline
    void
    print(const std::string &label = "vertices", const bool sorted=false) const {

        std::vector<typename BlockMap::const_iterator> siters (blocks.size());

        std::iota(siters.begin(), siters.end(), blocks.begin());
        std::sort(siters.begin(), siters.end(), [](auto a, auto b) { return a->first < b->first; });

        for(auto biter = siters.begin(); biter != siters.end(); ++biter) {
            (*biter)->second.print();
        }
    }


    template<class B>
    inline void
    copy_to(vertex_manager_blocks<Dim, TypeValue, B> &to) const {
        for(auto biter = blocks.begin(); biter != blocks.end(); ++biter) {
            to.update_block(biter->first, biter->second);
        }
    }

    float
    memory_in_kb() const {

        float kb = blocks.memory_in_kb();
        for(auto biter = blocks.begin(); biter != blocks.end(); ++biter) {
            kb += biter->second.memory_in_kb();
        }
        return kb;
    }


public:
    //! ---------------------------------------------------------------------------------
    //! const iterator
    //! ---------------------------------------------------------------------------------
    class const_iterator {

    public:
        typedef std::pair<TypeIndex, TypeValue> value_type;

    private:

        typedef const value_type& reference;
        typedef const value_type* pointer;
        typedef const_iterator self_type;
        typedef int difference_type;
        typedef std::forward_iterator_tag iterator_category;

        const vertex_manager_blocks &_vmanager;
        typename BlockMap::const_iterator _block;
        typename TypeBlock::const_iterator _vertex;
        value_type _iter;

        inline void set_end() {         _block = _vmanager.blocks.end();            }
        inline bool is_end() const {    return (_block == _vmanager.blocks.end());  }
        inline void update() {
            _iter = is_end() ? value_type(_vmanager.maxv(), 0) :
                               value_type(_vmanager.p2idx(_vmanager.unhash(_block->first, _vertex->first)), _vertex->second);
        }

    public:
        const_iterator(const vertex_manager_blocks &vmanager, const bool &is_begin)
            : _vmanager(vmanager){

            if (is_begin) {
                _block = _vmanager.blocks.begin();
                if (!this->is_end()) {
                    _vertex = _block->second.begin();
                }
            } else {
                set_end();
            }
            update();
        }

        self_type operator++() {
            self_type i = *this;    this->operator ++(0);   return i;
        }

        self_type operator++(int) {

            // go to the next vertex in the current block
            _vertex++;

            // if this was the last vertex in the current block
            if (_vertex == _block->second.end()) {

                // go to the next block
                _block++;

                // if the next block is not the last,
                // set the vertex to its begin
                if (!this->is_end()) {
                    _vertex = _block->second.begin();
                }
            }

            update();
            return *this;
        }
        reference operator*() const {   return _iter;                       }
        pointer operator->() const {    return &_iter;                      }

        bool operator==(const self_type& rhs) {
            return !(*this != rhs);
        }
        bool operator!=(const self_type& rhs) {

            if (&_vmanager != &rhs._vmanager)   return true;
            if (this->is_end() && rhs.is_end()) return false;
            return _block != rhs._block || _vertex != rhs._vertex;
        }

        inline void print(const std::string &label = "vertices", const bool sorted=true) const {

            if (is_end())    return;
            std::cout << " vertex " << int(_iter.first) << " : " << _vmanager.idx2p(_iter.first)
            << " [[" << int(_block->first) << " : " << int(_vertex->first) << "]]"
            << " = " << _iter.second << "\n";
        }
    };

    inline const_iterator begin() const {   return const_iterator(*this, true);     }
    inline const_iterator end() const {     return const_iterator(*this, false);    }
};

}   // end of namespace
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
#endif
