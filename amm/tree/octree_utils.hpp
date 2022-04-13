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
#ifndef AMM_TREE_UTILS_H
#define AMM_TREE_UTILS_H

//! --------------------------------------------------------------------------------
#include <cstdlib>
#include <vector>
#include <bitset>
#include <algorithm>
#include <unordered_set>

#include "types/dtypes.hpp"
#include "types/enums.hpp"
#include "containers/vec.hpp"
#include "utils/utils.hpp"

//! --------------------------------------------------------------------------------
using Vertex2 = Vec<2, TypeCoord>;
using Vertex3 = Vec<3, TypeCoord>;


namespace amm {

//! --------------------------------------------------------------------------------
//! some utilities on octree vertices and sizes (based on dimensionality)
//! --------------------------------------------------------------------------------
template <TypeDim Dim>
struct octree_utils {

    static_assert(Dim == 2 || Dim == 3, "tree_utils works for 2D and 3D only!");

    using Vertex  = Vec<Dim, TypeCoord>;


    //! whether any given coordinate is 1
    static inline
    bool
    is_unit(const Vertex &_) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] == TypeCoord(1))   return true;
        }
        return false;
    }


    //! whether any given coordinate is 2 or less
    static inline
    bool
    is_smaller_than_2(const Vertex &_) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] <= TypeCoord(2))   return true;
        }
        return false;
    }


    //! whether a given vertex has equal components
    static inline
    bool
    is_square(const Vertex &_) {
        for(TypeDim d = 1; d < Dim; d++) {
            if (_[0] != _[d])   return false;
        }
        return true;
    }


    //! whether a given vertex has all 2^L+1 components
    static inline
    bool
    is_pow2plus1(const Vertex &_) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (!AMM_is_pow2(_[d]-1))   return false;
        }
        return true;
    }

    //! whether a given size is valid
    static inline
    bool
    is_valid_node_sz(const Vertex &_, bool allow_rectangular) {

        // should be pow2+1
        if (!is_pow2plus1(_))
            return false;

        // should be square!
        if (!allow_rectangular) {
            if (!is_square(_))
                return false;
        }

        // could be rectangular, but only by a factor of 2
        else {
            const TypeCoord x = _[0]-1;
            for(TypeDim d = 1; d < Dim; d++) {
                const TypeCoord yorz = _[d]-1;
                if((x != yorz) && (x != yorz<<1) && (x<<1 != yorz))
                    return false;
            }
        }

        return true;
    }


    //! check whether [a,b] lies outside [A,B]
    static inline
    bool is_outside(const Vertex &a, const Vertex &b, const Vertex &A, const Vertex &B) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (b[d] < A[d] || a[d] > B[d])     return true;
        }
        return false;
    }

    //! check of the range bdry lies between a and b
    static inline
    bool
    is_across_boundary(const Vertex &a, const Vertex &b, const Vertex &bdry) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (a[d] < bdry[d] && b[d] > bdry[d])     return true;
        }
        return false;
    }

    static inline
    bool
    is_at_boundary(const Vertex &_, const Vertex &b) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] != b[d])   return false;
        }
        return true;
    }


    static inline
    bool
    is_on_boundary(const Vertex &_, const Vertex &b) {

        // if any dimension is outside the bounding box
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] > b[d])    return false;
        }

        // if any dimension is on the boundary
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] == b[d])   return true;
        }
        return false;
    }


    //! whether a vertex is contained in a bounding box
    static inline
    bool
    contains_bb(const Vertex &_, const Vertex &bb0, const Vertex &bb1) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] < bb0[d] || _[d] > bb1[d])
                return false;
        }
        return true;
    }


    static inline
    bool
    contains_os(const Vertex &_, const Vertex &o, const Vertex &s) {
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] < o[d] || _[d] >= s[d])
                return false;
        }
        return true;
    }


    //! expand a node to the next valid size
    static inline
    void
    expand_node(Vertex &_, bool allow_rectangular) {

        // first expand to the next pow2+1
        for(TypeDim d = 0; d < Dim; d++) {
            if (!AMM_is_pow2(_[d]-1))
                _[d] = amm::utils::next_pow2_plus_1(_[d]);
        }

        // if not rectangular, this should be a square
        if (!allow_rectangular) {
            TypeCoord dmx = _.max();
            for(TypeDim d = 0; d < Dim; d++) {
                if (_[d] < dmx) _[d] = dmx;
            }
            return;
        }

        // both 2 and 3 are pow_2+1
        // so one corner case is a node with one dim as 5 and other as 2
        if (Dim == 2) {
            if      (_[0] == 2 && _[1] == 5) {   _[0] = 3;   }
            else if (_[1] == 2 && _[0] == 5) {   _[1] = 3;   }
        }
        else {
            for(TypeDim d = 0; d < Dim; d++) {
                if      (_[d] == 2 && _[(d+1)%3] == 5) { _[d] = 3;   }
                else if (_[d] == 2 && _[(d+2)%3] == 5) { _[d] = 3;   }
            }
        }
    }


    //! convert the (origin, size) to bbox1 and center
    static inline
    void
    os2bb(const Vertex &o, const Vertex &s, Vertex &bb1) {
        static const Vertex uvec = Vertex(TypeCoord(1));
        bb1 = o + s - uvec;
    }

    static inline
    void
    os2c(const Vertex &o, const Vertex &s, Vertex &cnt) {
        static const Vertex uvec = Vertex(TypeCoord(1));
        cnt = o + (s - uvec) / TypeCoord(2);
    }

    static inline
    void
    bb2s(const Vertex &o, const Vertex &bb1, Vertex &s) {
        static const Vertex uvec = Vertex(TypeCoord(1));
        s = bb1 - o + uvec;
    }

    static inline
    Vertex
    os2bb(const Vertex &o, const Vertex &s) {
        Vertex _;  os2bb(o,s,_);    return _;
    }

    static inline
    Vertex
    os2c(const Vertex &o, const Vertex &s) {
        Vertex _;  os2c(o,s,_);    return _;
    }

    static inline
    Vertex
    bb2s(const Vertex &o, const Vertex &bb1) {
        Vertex _;  bb2s(o,bb1,_);    return _;
    }


    //! find the axes where the given two vertexs match
    static inline
    EnumAxes
    matching_axes(const Vertex &a, const Vertex &b) {
        TypeDim _ = 0;
        for(TypeDim d = 0; d < Dim; d++) {
            if (a[d] == b[d])   AMM_set_bit32(_, d);
        }
        return EnumAxes(_);
    }


    //! trim [a,b] to [A,B]
    static inline
    uint8_t
    trim_bounds(Vertex &a, Vertex &b, const Vertex &A, const Vertex &B) {

        // return
            // the number of dimensions that are trimmed (0,...,Dim)
            // Dim+1 if the box is completely outside
            // Dim+2 if trimmed node is of zero width

        uint8_t ntrims = 0;
        for(TypeDim d = 0; d < Dim; d++) {
            if (b[d] < A[d] || a[d] > B[d]) {   return Dim+1;           }
            if (a[d] < A[d]) {                  a[d] = A[d];  ntrims++; }
            if (b[d] > B[d]) {                  b[d] = B[d];  ntrims++; }
            if (a[d] == b[d]) {                 return Dim+2;           }
        }
        return ntrims;
    }


    //! find corners for a set of node bounds
    static inline
    void
    get_node_corners(const Vertex &bbox0, const Vertex &bbox1,
                     std::vector<Vertex> &_) {

        const Vertex bbox[2] = {bbox0, bbox1};
        TypeCornerId i = 0;

        switch(Dim) {

            case 2:
                    for(TypeDim y = 0; y < 2; y++)
                    for(TypeDim x = 0; x < 2; x++){

                        _[i][0] = bbox[x][0];
                        _[i][1] = bbox[y][1];
                        i++;
                    }
                    return;

            case 3:
                    for(TypeDim z = 0; z < 2; z++)
                    for(TypeDim y = 0; y < 2; y++)
                    for(TypeDim x = 0; x < 2; x++){

                        _[i][0] = bbox[x][0];
                        _[i][1] = bbox[y][1];
                        _[i][2] = bbox[z][2];
                        i++;
                    }
                    return;
        }
    }


    static inline
    void
    get_node_corners(const TypeIndex &norigin, const Vertex &nsize, const Vertex &dsize,
                     std::vector<TypeIndex> &_) {

        const TypeIndex sX = dsize[0];
        const TypeIndex X = nsize[0]-1;
        const TypeIndex Y = nsize[1]-1;
        const TypeIndex YsX = Y*sX;

        switch(Dim) {
        case 2: {
                _[0] = norigin;         _[1] = _[0] + X;
                _[2] = _[0] + YsX;      _[3] = _[2] + X;
                }
                return;

        case 3: {
                static const TypeIndex sXY = sX*dsize[1];
                const TypeIndex Z = nsize[2]-1;

                _[0] = norigin;         _[1] = _[0] + X;
                _[2] = _[0] + YsX;      _[3] = _[2] + X;

                _[4] = norigin + Z*sXY; _[5] = _[4] + X;
                _[6] = _[4] + YsX;      _[7] = _[6] + X;
                }
                return;
        }
    }


    static inline
    void
    get_node_corners(const TypeIndex &norigin, const Vertex &nsize, const Vertex &dsize,
                     std::unordered_set<TypeIndex> &_) {

        const TypeIndex sX = dsize[0];
        const TypeIndex X = nsize[0]-1;
        const TypeIndex Y = nsize[1]-1;
        const TypeIndex y1 = norigin + Y*sX;  // topleft in xy plane (high y)

        switch(Dim) {
        case 2: {
                _.insert(norigin);
                _.insert(norigin+X);
                _.insert(y1);
                _.insert(y1+X);
                }
                return;

        case 3: {
                static const TypeIndex sXY = sX*dsize[1];
                const TypeIndex Z = nsize[2]-1;

                _.insert(norigin);
                _.insert(norigin+X);
                _.insert(y1);
                _.insert(y1+X);

                const TypeIndex zo = norigin + Z*sXY;
                const TypeIndex z1 = zo + Y*sX;

                _.insert(zo);
                _.insert(zo+X);
                _.insert(z1);
                _.insert(z1+X);
                }
                return;
        }
    }


    //! convert a given node's bounds to its child's/parent's bounds
    static inline
    void
    get_bounds_child(const TypeChildId &_, Vertex &bb0, Vertex &sz) {

        // first shrink and then shift
        for(TypeDim d = 0; d < Dim; d++) {
            sz[d] = AMM_lsize_half(sz[d]);
            if (AMM_is_set_bit32(_,d))
                bb0[d] = AMM_lsize_dwn(bb0[d], sz[d]);
        }
    }


    static inline
    void
    get_bounds_parent(const TypeChildId &_, Vertex &bb0, Vertex &sz) {

        // first shift and then expand
        for(TypeDim d = 0; d < Dim; d++) {
            if (AMM_is_set_bit32(_,d))
                bb0[d] = AMM_lsize_up(bb0[d], sz[d]);
            sz[d] = AMM_lsize_dbl(sz[d]);
        }
    }


    //! get the child id that contains a given point (wrt node's center)
    static inline
    TypeChildId
    get_child_containing(const Vertex &_, const Vertex &ncenter) {

        TypeChildId child_id = 0;
        for(TypeDim d = 0; d < Dim; d++) {
            if (_[d] >= ncenter[d])
            AMM_set_bit32(child_id, d);
        }
        return child_id;
    }


    //! create a vector of 2^l+1 for fast lookup of node sizes!
    static inline
    std::vector<TypeCoord> level_sizes(const TypeScale L) {

        std::vector<TypeCoord> lsizes;
        lsizes.resize(L+1,0);
        for(TypeScale l = 0; l <= L; l++){
            lsizes[l] = AMM_pow2(L-l)+1;
        }
        return lsizes;
    }


    //! the dimension of a vertex is the number of components that are not multiples powers of 2
    //! can be used to identify the type of coefficient
    static inline
    uint8_t
    get_vertexDimension(const Vertex &_, const TypeScale l) {

        uint8_t rval = 0;
        for(TypeDim d = 0; d < Dim; d++) {
            if (!AMM_is_pow2k(_[d], l))    // is not multiple of 2^l
                rval++;
        }
        return rval;
    }


    //! get the scale of each dimensional component
    static inline
    Vec<Dim,TypeScale>
    get_vertexScales(const Vertex &_, const TypeScale &L) {

        Vec<Dim,TypeScale> s;
        for(uint8_t d = 0; d < Dim; d++) {
            s[d] = _[d] == 0 ? 0 : L-amm::utils::bcount_tz(_[d]);
        }
        return s;
    }


    static inline
    uint8_t
    aspect_ratio(const Vertex &_) {
        const TypeCoord a = _.max();
        const TypeCoord b = _.min();
        return (a==b) ? 1 : (a-1 == (b-1)<<1) ? 2 : 0;
    }


    //! given the size of the input domain, figure out the tree_size and out_size
    static inline
    Vertex
    size_tree(const Vertex &in_size, const TypeScale L) {

        const TypeCoord max_sz = TypeCoord(AMM_pow2(L)+1);

        Vertex _;
        for(TypeDim d = 0; d < Dim; d++) {
            _[d] = std::min(max_sz,
                            static_cast<TypeCoord>(amm::utils::next_pow2_plus_1(in_size[d])));
        }

        // tree size must be square
        _ = Vertex(_.max());
        return _;
    }


    static inline
    Vertex
    size_domain(const Vertex &in_size, const TypeScale L) {

        Vertex _;
        const Vertex tree_size = size_tree(in_size, L);
        for(TypeDim d = 0; d < Dim; d++) {
            _[d] = std::min(tree_size[d], in_size[d]);
        }
        return _;
    }


    static inline
    Vertex
    size_tree(const size_t in_size[], const TypeScale L) {
        return size_tree(Vertex(in_size), L);
    }


    static inline
    Vertex
    size_domain(const size_t in_size[], const TypeScale L) {
        return size_domain(Vertex(in_size), L);
    }


    //! convert from coordinates to vertex TypeIndex and back
    static inline
    TypeIndex p2idx(const Vertex2 &_, const Vertex2 &size) {
        return (TypeIndex(_[0]) + TypeIndex(size[0])*TypeIndex(_[1]));
    }
    static inline
    TypeIndex p2idx(const Vertex3 &_, const Vertex3 &size) {
        return (TypeIndex(_[0]) + TypeIndex(size[0])*(TypeIndex(_[1]) + TypeIndex(size[1])*TypeIndex(_[2])));
    }

    static inline
    Vertex2 idx2p(const TypeIndex &_, const Vertex2 &size) {
        return Vertex2(TypeCoord(_%size[0]), TypeCoord(_/size[0]));
    }
    static inline
    Vertex3 idx2p(const TypeIndex &_, const Vertex3 &size) {
        const TypeIndex XY = size[0]*size[1];
        const TypeIndex xy = _%XY;
        return Vertex3(TypeCoord(xy%size[0]), TypeCoord(xy/size[0]), TypeCoord(_/XY));
    }


    static inline
    TypeIndex product(const Vertex2 &_) {
        return TypeIndex(_[0])*TypeIndex(_[1]);
    }
    static inline
    TypeIndex product(const Vertex3 &_) {
        return TypeIndex(_[0])*TypeIndex(_[1])*TypeIndex(_[2]);
    }


    //! print a single node
    template<typename Tlcode, typename Tlvl>
    static inline void
    print_a_node(const Tlcode lcode, const Tlvl lvl,
                 const Vertex &bb0, const Vertex &sz, const bool end_line = true) {

        static Vertex bb1;
        os2bb(bb0, sz, bb1);

        for(Tlvl l = 0; l < lvl+1; l++)
            std::cout << " ";

        std::cout << " [" << std::bitset<CHAR_BIT*sizeof(lcode)>(lcode) << "] "
                  << " : " << static_cast<size_t>(lcode) << " @ lvl " << static_cast<size_t>(lvl)
                  << " :: bb0 = " << bb0 << ", bb1 = " << bb1 << ", sz = " << sz;

        if (end_line)
            std::cout << std::endl;
    }
};

}  // end of namespace
#endif
