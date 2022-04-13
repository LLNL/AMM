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
#ifndef AMM_MULTILINEAR_SPLITTER_H
#define AMM_MULTILINEAR_SPLITTER_H

//! --------------------------------------------------------------------------------
#include <vector>
#include <sstream>
#include <stdexcept>
#include <numeric>

#include "types/byte_traits.hpp"
#include "types/dtypes.hpp"
#include "utils/utils.hpp"
#include "containers/vec.hpp"
#include "containers/bitmask.hpp"
#include "tree/vmanager_abstract.hpp"


//! ----------------------------------------------------------------------------
#define AMM_split_edge(a,b,_a,_b)                              (0.500*(AMM_if_val(a,_a)+AMM_if_val(b,_b)))
#define AMM_split_face(a,b,c,d,_a,_b,_c,_d)                    (0.250*(AMM_if_val(a,_a)+AMM_if_val(b,_b)+AMM_if_val(c,_c)+AMM_if_val(d,_d)))
#define AMM_split_hex(a,b,c,d,e,f,g,h,_a,_b,_c,_d,_e,_f,_g,_h) (0.125*(AMM_if_val(a,_a)+AMM_if_val(b,_b)+AMM_if_val(c,_c)+AMM_if_val(d,_d)+AMM_if_val(e,_e)+AMM_if_val(f,_f)+AMM_if_val(g,_g)+AMM_if_val(h,_h)))

#define AMM_MLSPLIT_invalid_val(T) AMM_nan(T)
#define AMM_MLSPLIT_is_invalid(_)  AMM_is_nan(_)

namespace amm {

//! ----------------------------------------------------------------------------
//! Multilinear Node Splitter
//! 
//! NodeSplitter computes the split points using a cannonical representation
//! split points are indexed in row-major order
//! 2D: 5 split points, 3D: 19 split pointss
//! 
//! @tparam Dim: spatial dimensionality of nodes
//! @tparam TypeValue: data type of cell data values
//! 
//! ----------------------------------------------------------------------------
template<TypeDim Dim, typename TypeValue>
class MultilinearNodeSplitter {

    static_assert((Dim == 2 || Dim == 3), "MultilinearNodeSplitter works for 2D and 3D only!");

    using TypeVertex = Vec<Dim, TypeCoord>;
    using VertexManager = amm::vertex_manager<Dim, TypeValue>;

    static constexpr TypeChildId snChildren = 2 == Dim ? 8 : 26;
    static constexpr TypeCornerId snSplits = 2 == Dim ? 5 : 19;        // number of split points of a node
    static constexpr TypeCornerId snCorners = AMM_pow2(Dim);           // number of corners of a node
    static constexpr TypeValue invalid_val = AMM_MLSPLIT_invalid_val(TypeValue);

public:
    using TypeSplitFlag = amm::bitmask<snSplits>;


    //! ------------------------------------------------------------------------
    inline static
    void
    midpoint(const TypeVertex &a, const TypeVertex &b,
             TypeVertex &_) {
        for(TypeDim k = 0; k < Dim; k++) {
            _[k] = (a[k]+b[k]) >> 1;
        }
    }


    inline static
    void
    midpoint(const TypeVertex &a, const TypeVertex &b, const TypeVertex &c, const TypeVertex &d,
             TypeVertex &_) {
        for(TypeDim k = 0; k < Dim; k++) {
            _[k] = (a[k]+b[k]+c[k]+d[k]) >> 2;
        }
    }


    inline static
    void
    midpoint(const TypeVertex &a, const TypeVertex &b, const TypeVertex &c, const TypeVertex &d,
             const TypeVertex &e, const TypeVertex &f, const TypeVertex &g, const TypeVertex &h,
             TypeVertex &_) {
        for(TypeDim k = 0; k < Dim; k++) {
            _[k] = (a[k]+b[k]+c[k]+d[k]+e[k]+f[k]+g[k]+h[k]) >> 3;
        }
    }

    //! ------------------------------------------------------------------------
    //! convert the cannonical index of a split point into the actual vertex
    inline static
    void
    get_split_point(const TypeCornerId &split_id, const std::vector<TypeVertex> &corners,
                    TypeVertex &_) {


        if (Dim == 2) {
            switch(split_id) {
                case 0:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], _);                          return;
                case 1:     MultilinearNodeSplitter::midpoint(corners[0], corners[2], _);                          return;
                case 3:     MultilinearNodeSplitter::midpoint(corners[1], corners[3], _);                          return;
                case 4:     MultilinearNodeSplitter::midpoint(corners[2], corners[3], _) ;                         return;
                case 2:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], corners[2], corners[3], _);  return;
            }
        }

        else {
            switch(split_id) {
                case 0:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], _);                          return;
                case 1:     MultilinearNodeSplitter::midpoint(corners[0], corners[2], _);                          return;
                case 3:     MultilinearNodeSplitter::midpoint(corners[1], corners[3], _);                          return;
                case 4:     MultilinearNodeSplitter::midpoint(corners[2], corners[3], _);                          return;

                case 5:     MultilinearNodeSplitter::midpoint(corners[0], corners[4], _);                          return;
                case 7:     MultilinearNodeSplitter::midpoint(corners[1], corners[5], _);                          return;
                case 11:    MultilinearNodeSplitter::midpoint(corners[2], corners[6], _);                          return;
                case 13:    MultilinearNodeSplitter::midpoint(corners[3], corners[7], _);                          return;

                case 14:    MultilinearNodeSplitter::midpoint(corners[4], corners[5], _);                          return;
                case 15:    MultilinearNodeSplitter::midpoint(corners[4], corners[6], _);                          return;
                case 17:    MultilinearNodeSplitter::midpoint(corners[5], corners[7], _);                          return;
                case 18:    MultilinearNodeSplitter::midpoint(corners[6], corners[7], _);                          return;

                case 2:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], corners[2], corners[3], _);  return;
                case 6:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], corners[4], corners[5], _);  return;
                case 8:     MultilinearNodeSplitter::midpoint(corners[0], corners[2], corners[4], corners[6], _);  return;
                case 10:    MultilinearNodeSplitter::midpoint(corners[1], corners[3], corners[5], corners[7], _);  return;
                case 12:    MultilinearNodeSplitter::midpoint(corners[2], corners[3], corners[6], corners[7], _);  return;
                case 16:    MultilinearNodeSplitter::midpoint(corners[4], corners[5], corners[6], corners[7], _);  return;

                case 9:     MultilinearNodeSplitter::midpoint(corners[0], corners[1], corners[2], corners[3],
                                                              corners[4], corners[5], corners[6], corners[7], _);  return;
            }
        }

        AMM_error_invalid_arg(true,
                               " multilinearNodeSplitter<D=%d>.get_split(%d) got invalid split_id! There are %d splits in a %d dimensional node!\n",
                               int(Dim), int(split_id), int(snSplits), int(Dim));
    }


    //! ------------------------------------------------------------------------
    //! update the split flag by setting the bits for split points that must be added to create a given child
    inline static
    void
    set_splits_flags(const TypeChildId &child_id, TypeSplitFlag &_) {


        AMM_error_invalid_arg(child_id >= snChildren,
                               " multilinearNodeSplitter<D=%d>.set_split_flags(%d) got invalid child_id! There are %d types of children of a %d dimensional node!\n",
                               int(Dim), int(child_id), int(snChildren), int(Dim));

        // create a static list of split flags!
        static std::vector<TypeSplitFlag> split_flags;

        if (split_flags.empty()) {

            split_flags.resize(snChildren);
            if (2 == Dim) {
                split_flags[0].init_list({0,1,2});
                split_flags[1].init_list({0,2,3});
                split_flags[2].init_list({1,2,4});
                split_flags[3].init_list({2,3,4});
                split_flags[4].init_list({1,3});
                split_flags[5].init_list({1,3});
                split_flags[6].init_list({0,4});
                split_flags[7].init_list({0,4});
            }

            else {
                split_flags[0].init_list({0,1,2, 5,6,8,9});
                split_flags[1].init_list({0,2,3, 6,7,9,10});
                split_flags[2].init_list({1,2,4, 8,9,11,12});
                split_flags[3].init_list({2,3,4, 9,10,12,13});
                split_flags[4].init_list({5,6,8,9, 14,15,16});
                split_flags[5].init_list({6,7,9,10, 14,16,17});
                split_flags[6].init_list({8,9,11,12, 15,16,18});
                split_flags[7].init_list({9,10,12,13, 16,17,18});

                split_flags[8].init_list({1,3, 5,7,8,10});
                split_flags[9].init_list({1,3, 8,10,11,13});
                split_flags[10].init_list({5,7,8,10, 15,17});
                split_flags[11].init_list({8,10,11,13, 15,17});

                split_flags[12].init_list({0,4, 5,6,11,12});
                split_flags[13].init_list({0,4, 6,7,12,13});
                split_flags[14].init_list({5,6,11,12, 14,18});
                split_flags[15].init_list({6,7,12,13, 14,18});

                split_flags[16].init_list({0,1,2, 14,15,16});
                split_flags[17].init_list({0,2,3, 14,16,17});
                split_flags[18].init_list({1,2,4, 15,16,18});
                split_flags[19].init_list({2,3,4, 16,17,18});

                split_flags[20].init_list({5,7,11,13});
                split_flags[21].init_list({5,7,11,13});

                split_flags[22].init_list({1,3,15,17});
                split_flags[23].init_list({1,3,15,17});

                split_flags[24].init_list({0,4,14,18});
                split_flags[25].init_list({0,4,14,18});
            }
        }

        // set the requested flags!
        _ |= split_flags[child_id];
    }


    //! ------------------------------------------------------------------------
    //! the function set_splits_flags creates the flag with respect to a given node
    //! however, when the a rectangular child must be split into smaller children
    //! the split point must be transformed from the rectangular child to its parent
    inline static
    void
    offset_longnode(const TypeChildId &child_id, std::vector<TypeValue> &split_values) {

        static std::vector<TypeValue> fixed_splits (snSplits);
        std::fill(fixed_splits.begin(), fixed_splits.end(), invalid_val);

        AMM_error_invalid_arg(child_id < (1 << Dim) || child_id >= snChildren,
                              "multilinearNodeSplitter<D=%d>.offset_longnode(%d) got invalid child_id! The function works for non-standard children only!\n",
                              int(Dim), int(child_id));

        if (2 ==  Dim) {

            switch(child_id)  {

            case 4:     fixed_splits[0] = split_values[0];
                        fixed_splits[2] = split_values[4];
                        break;

            case 5:     fixed_splits[2] = split_values[0];
                        fixed_splits[4] = split_values[4];
                        break;

            case 6:     fixed_splits[1] = split_values[1];
                        fixed_splits[2] = split_values[3];
                        break;

            case 7:     fixed_splits[3] = split_values[3];
                        fixed_splits[2] = split_values[1];
                        break;
            }
        }

        else {

            switch(child_id)  {

            case 8:     fixed_splits[0] = split_values[0];
                        fixed_splits[2] = split_values[4];
                        fixed_splits[6] = split_values[14];
                        fixed_splits[9] = split_values[18];
                        break;

            case 9:     fixed_splits[2] = split_values[0];
                        fixed_splits[4] = split_values[4];
                        fixed_splits[9] = split_values[14];
                        fixed_splits[12] = split_values[18];
                        break;

            case 10:    fixed_splits[6] = split_values[0];
                        fixed_splits[9] = split_values[4];
                        fixed_splits[14] = split_values[14];
                        fixed_splits[16] = split_values[18];
                        break;

            case 11:    fixed_splits[9] = split_values[0];
                        fixed_splits[12] = split_values[4];
                        fixed_splits[16] = split_values[14];
                        fixed_splits[18] = split_values[18];
                        break;

            case 12:    fixed_splits[1] = split_values[1];
                        fixed_splits[2] = split_values[3];
                        fixed_splits[8] = split_values[15];
                        fixed_splits[9] = split_values[17];
                        break;

            case 13:    fixed_splits[2] = split_values[1];
                        fixed_splits[3] = split_values[3];
                        fixed_splits[9] = split_values[15];
                        fixed_splits[10] = split_values[17];
                        break;

            case 14:    fixed_splits[8] = split_values[1];
                        fixed_splits[9] = split_values[3];
                        fixed_splits[15] = split_values[15];
                        fixed_splits[16] = split_values[17];
                        split_values = fixed_splits;
                        return;

            case 15:    fixed_splits[9] = split_values[1];
                        fixed_splits[10] = split_values[3];
                        fixed_splits[16] = split_values[15];
                        fixed_splits[17] = split_values[17];
                        break;

            case 16:    fixed_splits[5] = split_values[5];
                        fixed_splits[6] = split_values[7];
                        fixed_splits[8] = split_values[11];
                        fixed_splits[9] = split_values[13];
                        break;

            case 17:    fixed_splits[6] = split_values[5];
                        fixed_splits[7] = split_values[7];
                        fixed_splits[9] = split_values[11];
                        fixed_splits[10] = split_values[13];
                        break;

            case 18:    fixed_splits[8] = split_values[5];
                        fixed_splits[9] = split_values[7];
                        fixed_splits[11] = split_values[11];
                        fixed_splits[12] = split_values[13];
                        break;

            case 19:    fixed_splits[9] = split_values[5];
                        fixed_splits[10] = split_values[7];
                        fixed_splits[12] = split_values[11];
                        fixed_splits[13] = split_values[13];
                        break;

            case 20:    fixed_splits[0] = split_values[0];
                        fixed_splits[1] = split_values[1];
                        fixed_splits[2] = split_values[2];
                        fixed_splits[3] = split_values[3];
                        fixed_splits[4] = split_values[4];

                        fixed_splits[6] = split_values[14];
                        fixed_splits[8] = split_values[15];
                        fixed_splits[9] = split_values[16];
                        fixed_splits[10] = split_values[17];
                        fixed_splits[12] = split_values[18];
                        break;

            case 21:    fixed_splits[6] = split_values[0];
                        fixed_splits[8] = split_values[1];
                        fixed_splits[9] = split_values[2];
                        fixed_splits[10] = split_values[3];
                        fixed_splits[12] = split_values[4];

                        fixed_splits[14] = split_values[14];
                        fixed_splits[15] = split_values[15];
                        fixed_splits[16] = split_values[16];
                        fixed_splits[17] = split_values[17];
                        fixed_splits[18] = split_values[18];
                        break;

            case 22:    fixed_splits[0] = split_values[0];
                        fixed_splits[5] = split_values[5];
                        fixed_splits[6] = split_values[6];
                        fixed_splits[7] = split_values[7];
                        fixed_splits[14] = split_values[14];

                        fixed_splits[2] = split_values[4];
                        fixed_splits[8] = split_values[11];
                        fixed_splits[9] = split_values[12];
                        fixed_splits[10] = split_values[13];
                        fixed_splits[16] = split_values[18];
                        break;

            case 23:    fixed_splits[2] = split_values[0];
                        fixed_splits[8] = split_values[5];
                        fixed_splits[9] = split_values[6];
                        fixed_splits[10] = split_values[7];
                        fixed_splits[16] = split_values[14];

                        fixed_splits[4] = split_values[4];
                        fixed_splits[11] = split_values[11];
                        fixed_splits[12] = split_values[12];
                        fixed_splits[13] = split_values[13];
                        fixed_splits[18] = split_values[18];
                        break;

            case 24:    fixed_splits[1] = split_values[1];
                        fixed_splits[5] = split_values[5];
                        fixed_splits[8] = split_values[8];
                        fixed_splits[11] = split_values[11];
                        fixed_splits[15] = split_values[15];

                        fixed_splits[2] = split_values[3];
                        fixed_splits[6] = split_values[7];
                        fixed_splits[9] = split_values[10];
                        fixed_splits[12] = split_values[13];
                        fixed_splits[16] = split_values[17];
                        break;

            case 25:    fixed_splits[2] = split_values[1];
                        fixed_splits[6] = split_values[5];
                        fixed_splits[9] = split_values[8];
                        fixed_splits[12] = split_values[11];
                        fixed_splits[16] = split_values[15];

                        fixed_splits[3] = split_values[3];
                        fixed_splits[7] = split_values[7];
                        fixed_splits[10] = split_values[10];
                        fixed_splits[13] = split_values[13];
                        fixed_splits[17] = split_values[17];
                        break;
            }
        }
        split_values = fixed_splits;
    }


    //! ------------------------------------------------------------------------
    //! split_node function fills a pre-allocated array with values
    //! at the split points indexed in the cannonical form
    //! if all corners needed to compute a split do not exist,
    //! then an invalid value is pushed in!
    inline static
    void
    split_node(const TypeVertex &nsize, const std::vector<TypeVertex> &corners,
               const VertexManager &vmanager, std::vector<TypeValue> &split_values) {

        AMM_error_invalid_arg(split_values.size() != snSplits,
                              "multilinearNodeSplitter<D=%d>.split_node() expects a pre-allocated vector of size %d to store split values!\n",
                              int(Dim), int(snSplits));

        AMM_error_invalid_arg(corners.size() != snCorners,
                              "multilinearNodeSplitter<D=%d>.split_node() expects %d corners!\n",
                              int(Dim), int(snCorners));

        // initialize all splits to invalid
        std::fill(split_values.begin(), split_values.end(), invalid_val);

        // collect all the corners and their values!
        static std::vector<TypeValue> cvalues (snCorners);
        static std::vector<bool> cfound (snCorners);
        bool tmp;

        for(TypeCornerId i = 0; i < snCorners; i++) {
            cvalues[i] = vmanager.get(corners[i], tmp);
            cfound[i] = tmp;
        }

        // if the sum of all booleans is zero, no vertex exists!
        if (0 == std::accumulate(cfound.begin(), cfound.end(), int(0)))
            return;

        // ---------------------------------------------------------------------
        // split a 2D node!
        if (2 == Dim) {

            // split x edges
            if (nsize[0] > 2){
                if (cfound[0] || cfound[1]) {   split_values[0] = AMM_split_edge(cvalues[0], cvalues[1], cfound[0], cfound[1]); }
                if (cfound[2] || cfound[3]) {   split_values[4] = AMM_split_edge(cvalues[2], cvalues[3], cfound[2], cfound[3]); }
            }

            // split y edges
            if (nsize[1] > 2){
                if (cfound[0] || cfound[2]) {   split_values[1] = AMM_split_edge(cvalues[0], cvalues[2], cfound[0], cfound[2]); }
                if (cfound[1] || cfound[3]) {   split_values[3] = AMM_split_edge(cvalues[1], cvalues[3], cfound[1], cfound[3]); }
            }

            // split the face
            if (nsize[0] > 2 || nsize[1] > 2) {
                split_values[2] = AMM_split_face(cvalues[0], cvalues[1], cvalues[2], cvalues[3],
                                                 cfound[0], cfound[1], cfound[2], cfound[3]);
            }
        }

        // ---------------------------------------------------------------------
        // split a 3D node!
        else {

            if (nsize[0] > 2){
                if (cfound[0] || cfound[1]) {   split_values[0]  = AMM_split_edge(cvalues[0], cvalues[1], cfound[0], cfound[1]); }
                if (cfound[2] || cfound[3]) {   split_values[4]  = AMM_split_edge(cvalues[2], cvalues[3], cfound[2], cfound[3]); }
                if (cfound[4] || cfound[5]) {   split_values[14] = AMM_split_edge(cvalues[4], cvalues[5], cfound[4], cfound[5]); }
                if (cfound[6] || cfound[7]) {   split_values[18] = AMM_split_edge(cvalues[6], cvalues[7], cfound[6], cfound[7]); }
            }

            if (nsize[1] > 2) {
                if (cfound[0] || cfound[2]) {   split_values[1]  = AMM_split_edge(cvalues[0], cvalues[2], cfound[0], cfound[2]); }
                if (cfound[1] || cfound[3]) {   split_values[3]  = AMM_split_edge(cvalues[1], cvalues[3], cfound[1], cfound[3]); }
                if (cfound[4] || cfound[6]) {   split_values[15] = AMM_split_edge(cvalues[4], cvalues[6], cfound[4], cfound[6]); }
                if (cfound[5] || cfound[7]) {   split_values[17] = AMM_split_edge(cvalues[5], cvalues[7], cfound[5], cfound[7]); }
            }

            if (nsize[2] > 2) {
                if (cfound[0] || cfound[4]) {   split_values[5]  = AMM_split_edge(cvalues[0], cvalues[4], cfound[0], cfound[4]); }
                if (cfound[1] || cfound[5]) {   split_values[7]  = AMM_split_edge(cvalues[1], cvalues[5], cfound[1], cfound[5]); }
                if (cfound[2] || cfound[6]) {   split_values[11] = AMM_split_edge(cvalues[2], cvalues[6], cfound[2], cfound[6]); }
                if (cfound[3] || cfound[7]) {   split_values[13] = AMM_split_edge(cvalues[3], cvalues[7], cfound[3], cfound[7]); }
            }

            if (nsize[0] > 2 && nsize[1] > 2) {
                if (cfound[0] || cfound[1] || cfound[2] || cfound[3]) {
                    split_values[2]  = AMM_split_face(cvalues[0], cvalues[1], cvalues[2], cvalues[3],
                                                      cfound[0], cfound[1], cfound[2], cfound[3]);
                }
                if (cfound[4] || cfound[5] || cfound[6] || cfound[7]) {
                    split_values[16] = AMM_split_face(cvalues[4], cvalues[5], cvalues[6], cvalues[7],
                                                      cfound[4], cfound[5], cfound[6], cfound[7]);
                }
            }

            if (nsize[0] > 2 && nsize[2] > 2) {

                if (cfound[0] || cfound[1] || cfound[4] || cfound[5]) {
                    split_values[6]  = AMM_split_face(cvalues[0], cvalues[1], cvalues[4], cvalues[5],
                                                      cfound[0], cfound[1], cfound[4], cfound[5]);
                }
                if (cfound[2] || cfound[3] || cfound[6] || cfound[7]) {
                    split_values[12] = AMM_split_face(cvalues[2], cvalues[3], cvalues[6], cvalues[7],
                                                      cfound[2], cfound[3], cfound[6], cfound[7]);
                }
            }
            if (nsize[1] > 2 && nsize[2] > 2) {

                if (cfound[0] || cfound[2] || cfound[4] || cfound[6]) {
                    split_values[8]  = AMM_split_face(cvalues[0], cvalues[2], cvalues[4], cvalues[6],
                                                      cfound[0], cfound[2], cfound[4], cfound[6]);
                }
                if (cfound[1] || cfound[3] || cfound[5] || cfound[7]) {
                    split_values[10] = AMM_split_face(cvalues[1], cvalues[3], cvalues[5], cvalues[7],
                                                      cfound[1], cfound[3], cfound[5], cfound[7]);
                }
            }

            if (nsize[0] > 0 && nsize[1] > 2 && nsize[2] > 2) {
                split_values[9] = AMM_split_hex(cvalues[0], cvalues[1], cvalues[2], cvalues[3],
                                                cvalues[4], cvalues[5], cvalues[6], cvalues[7],
                                                cfound[0], cfound[1], cfound[2], cfound[3],
                                                cfound[4], cfound[5], cfound[6], cfound[7]);
            }
        }

        return;
    }


    //! ------------------------------------------------------------------------
};
//! ----------------------------------------------------------------------------
}   // end of namespace

#endif
