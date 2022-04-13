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
#ifndef AMM_NODECONFIG_CORE_H
#define AMM_NODECONFIG_CORE_H

/// ------------------------------------------------------------------------------------
/*#include <vector>
 *
#include "types/dtypes.hpp"
#include "types/enums.hpp"
#include "containers/bitmask.hpp"*/

namespace amm {

/// ------------------------------------------------------------------------------------
/// a utility structure to manipulate child flags
/// the "CoreConfiguration" returns precomputed configuration changes
/// and is not used directly by the user
/// ------------------------------------------------------------------------------------

template <TypeDim Dim>
struct OctreeNodeCoreConfiguration {

    static_assert((Dim == 2 || Dim == 3), "OctreeNodeCoreConfiguration works for 2D and 3D only!");

protected:
    //! child mask needs to store 8 or 26 bits (one per child_
    using TypeChildMask = amm::bitmask<Dim==2?8:26>;
    using TypeChildFlag = typename TypeChildMask::word_type;

    //! occupancy mask only needs 4 or 8 bits (one per quad/oct)
    using TypeOccupancyMask = amm::bitmask<Dim==2?4:8>;

    //! containers
    using VecOccMask = typename std::vector<TypeOccupancyMask>;
    using VecChildMask = typename std::vector<TypeChildMask>;
    using VecVecChildMask = typename std::vector<VecChildMask>;

    //! config ids
#ifdef AMM_ENCODE_CFLAGS
    using TypeConfig = uint8_t;
#else
    using TypeConfig = typename TypeChildMask::word_type;
#endif



    /// --------------------------------------------------------------------------------
    /// relevant static variables
    /// --------------------------------------------------------------------------------
    //! number of possible types of axes
    static constexpr EnumAxes snAxes =  EnumAxes(AMM_pow2(Dim));    // (0,x,y,xy) or (0,x,y,z,xy,yz,xz,xyz)

    //! number of possible types of standard and non-standard children
    static const  TypeChildId snChildren = AMM_pow2(Dim);
    static const  TypeChildId snChildren_nonstandard = (Dim==2) ? 8 : 26;

    //! config id for a fully refined node
    static const TypeChildFlag sCflags_full = (2==Dim) ? 15 : 255; // first 4/8 bits set




    /// --------------------------------------------------------------------------------
    /// core functions to create required flags (to be specialized for Dim)
    /// --------------------------------------------------------------------------------

    //! based on the child_id, return the axes in which the child is long
    static inline
    std::vector<EnumAxes>
    long_axes();


    //! based on the child_id, return the axis that the parent should be split at
    static inline
    std::vector<EnumAxes>
    split_axes();


    //! based on the child_id, return a bitmap for what axes needs to be scaled and shifted
    static inline
    std::vector<TypeDim>
    bounds_conversion_flag_for_child_id();


    //! based on the child_id, return a bitmap for what axes needs to be scaled and shifted
    static inline
    TypeChildId
    child_id_for_bounds_conversion_flag(const TypeDim &_);


    //! get occupancy flags for each node
    static inline
    VecOccMask
    occupancy_flags();


    //! get child ids invalid with respect to dimensionality
    static inline
    VecChildMask
    conflicts_for_axes();


    //! get components of a child respect to dimensionality
    static inline
    VecVecChildMask
    components_for_axes();


    //! get conflicts for each child node as a vector of childflags
    static inline
    VecChildMask
    conflicts_for_child();


    //! how to split a child, when another has been requested
    static inline
    TypeChildMask
    components_for_child(const  TypeChildId &to_split, const  TypeChildId &to_create);


    //! components of a node
    static inline
    VecChildMask
    node_components();

#ifdef UNUSED
    //! get merge components of all child nodes
    static inline
    VecVecChildMask
    merge_components();
#endif

};



/// -----------------------------------------------------------------------------------
/// template specializations
/// -----------------------------------------------------------------------------------

/// -----------------------------------------------------------------------------------
//! get long axes for each child
/// -----------------------------------------------------------------------------------

template<>
inline std::vector<EnumAxes>
OctreeNodeCoreConfiguration<2>::long_axes() {

    std::vector<EnumAxes> _(snChildren_nonstandard);
    _[0] =  EnumAxes::None;
    _[1] =  EnumAxes::None;
    _[2] =  EnumAxes::None;
    _[3] =  EnumAxes::None;
    _[4] =  EnumAxes::X;
    _[5] =  EnumAxes::X;
    _[6] =  EnumAxes::Y;
    _[7] =  EnumAxes::Y;
    return _;
}

template<>
inline std::vector<EnumAxes>
OctreeNodeCoreConfiguration<3>::long_axes() {

    std::vector<EnumAxes> _(snChildren_nonstandard);
    _[0] =  EnumAxes::None;
    _[1] =  EnumAxes::None;
    _[2] =  EnumAxes::None;
    _[3] =  EnumAxes::None;
    _[4] =  EnumAxes::None;
    _[5] =  EnumAxes::None;
    _[6] =  EnumAxes::None;
    _[7] =  EnumAxes::None;
    _[8] =  EnumAxes::X;
    _[9] =  EnumAxes::X;
    _[10] =  EnumAxes::X;
    _[11] =  EnumAxes::X;
    _[12] =  EnumAxes::Y;
    _[13] =  EnumAxes::Y;
    _[14] =  EnumAxes::Y;
    _[15] =  EnumAxes::Y;
    _[16] =  EnumAxes::Z;
    _[17] =  EnumAxes::Z;
    _[18] =  EnumAxes::Z;
    _[19] =  EnumAxes::Z;
    _[20] =  EnumAxes::XY;
    _[21] =  EnumAxes::XY;
    _[22] =  EnumAxes::XZ;
    _[23] =  EnumAxes::XZ;
    _[24] =  EnumAxes::YZ;
    _[25] =  EnumAxes::YZ;
    return _;
}


/// -----------------------------------------------------------------------------------
//! get split axes for each child
/// -----------------------------------------------------------------------------------

template<>
inline std::vector<EnumAxes>
OctreeNodeCoreConfiguration<2>::split_axes() {
    std::vector<EnumAxes> _(snChildren_nonstandard);
    _[0] =  EnumAxes::XY;
    _[1] =  EnumAxes::XY;
    _[2] =  EnumAxes::XY;
    _[3] =  EnumAxes::XY;
    _[4] =  EnumAxes::Y;
    _[5] =  EnumAxes::Y;
    _[6] =  EnumAxes::X;
    _[7] =  EnumAxes::X;
    return _;
}

template<>
inline std::vector<EnumAxes>
OctreeNodeCoreConfiguration<3>::split_axes() {
    std::vector<EnumAxes> _(snChildren_nonstandard);
    _[0] =  EnumAxes::XYZ;
    _[1] =  EnumAxes::XYZ;
    _[2] =  EnumAxes::XYZ;
    _[3] =  EnumAxes::XYZ;
    _[4] =  EnumAxes::XYZ;
    _[5] =  EnumAxes::XYZ;
    _[6] =  EnumAxes::XYZ;
    _[7] =  EnumAxes::XYZ;
    _[8]  =  EnumAxes::YZ;
    _[9]  =  EnumAxes::YZ;
    _[10] =  EnumAxes::YZ;
    _[11] =  EnumAxes::YZ;
    _[12] =  EnumAxes::XZ;
    _[13] =  EnumAxes::XZ;
    _[14] =  EnumAxes::XZ;
    _[15] =  EnumAxes::XZ;
    _[16] =  EnumAxes::XY;
    _[17] =  EnumAxes::XY;
    _[18] =  EnumAxes::XY;
    _[19] =  EnumAxes::XY;
    _[20] =  EnumAxes::Z;
    _[21] =  EnumAxes::Z;
    _[22] =  EnumAxes::Y;
    _[23] =  EnumAxes::Y;
    _[24] =  EnumAxes::X;
    _[25] =  EnumAxes::X;
    return _;
}


/// -----------------------------------------------------------------------------------
//! get conversion flags to convert size to a child
/// -----------------------------------------------------------------------------------

template<>
inline std::vector<TypeDim>
OctreeNodeCoreConfiguration<2>::bounds_conversion_flag_for_child_id() {

    std::vector<TypeDim> _(snChildren_nonstandard);

    // 2D: regular children
    _[0] = 3;   // 00 11
    _[1] = 7;   // 01 11
    _[2] = 11;  // 10 11
    _[3] = 15;  // 11 11

    // 2D: horizontal rectangles  (2x,x)
        // expand in y, and shift in y for 5
    _[4] = 2;   // 00 10
    _[5] = 10;  // 10 10

    // 2D: vertical rectangles (x,2x)
        // expand in x, and shift in x for 5
    _[6] = 1;   // 00 01
    _[7] = 5;   // 01 01

    return _;
}

template<>
inline std::vector<TypeDim>
OctreeNodeCoreConfiguration<3>::bounds_conversion_flag_for_child_id() {

    std::vector<TypeDim> _(snChildren_nonstandard);

    // 3D: regular children
    _[0] = 7;       // 000 111
    _[1] = 15;      // 001 111
    _[2] = 23;      // 010 111
    _[3] = 31;      // 011 111
    _[4] = 39;      // 100 111
    _[5] = 47;      // 101 111
    _[6] = 55;      // 110 111
    _[7] = 63;      // 111 111

    // 3D: rectangles in x-dimension (2x,x,x)
        // expand in y and z
        // shift in y for 9 and 11
        // shift in z for 10 and 11
    _[8] = 6;       // 000 110
    _[9] = 22;      // 010 110
    _[10] = 38;      // 100 110
    _[11] = 54;      // 110 110

    // 3D: rectangles in y-dimension (x,2x,x)
          // expand in x and z
          // shift in x for 13 and 15
          // shift in z for 14 and 15
    _[12] = 5;       // 000 101
    _[13] = 13;      // 001 101
    _[14] = 37;      // 100 101
    _[15] = 45;      // 101 101

    // 3D: rectangles in z-dimension (x,x,2x)
        // expand in x and y
        // shift in x for 17 and 19
        // shift in y for 18 and 19
    _[16] = 3;       // 000 011
    _[17] = 11;      // 001 011
    _[18] = 19;      // 010 011
    _[19] = 27;      // 011 011

    // 3D: slabs in xy-dimension (2x,2x,x)
        // expand in z
        // shift in z for 21
    _[20] = 4;       // 000 100
    _[21] = 36;      // 100 100

    // 3D: slabs in xz-dimension (2x,x,2x)
        // expand in y
        // shift in y for 23
    _[22] = 2;       // 000 010
    _[23] = 18;      // 010 010

    // 3D: slabs in yz-dimension (x,2x,2x)
        // expand in x
        // shift in x for 25
    _[24] = 1;       // 000 001
    _[25] = 9;       // 001 001

    return _;
}


/// -----------------------------------------------------------------------------------
//! get child id for a bound conversion flag
/// -----------------------------------------------------------------------------------

template<>
inline  TypeChildId
OctreeNodeCoreConfiguration<2>::child_id_for_bounds_conversion_flag(const TypeDim &_) {
    switch (_) {
        case 1:  return 6;      // 00 01
        case 2:  return 4;      // 00 10
        case 3:  return 0;      // 00 11
        case 5:  return 7;      // 01 01
        case 7:  return 1;      // 01 11
        case 11: return 2;      // 10 11
        case 10: return 5;      // 10 10
        case 15: return 3;      // 11 11
    }

    AMM_error_invalid_arg(true,
                           "NodeConfig<2>::get_child_id_from_bounds_conversion_flags(%d) got bounds flag!\n",
                           int(_));
}

template<>
inline  TypeChildId
OctreeNodeCoreConfiguration<3>::child_id_for_bounds_conversion_flag(const TypeDim &_) {
    switch (_) {
        case 1:  return 24;
        case 2:  return 22;
        case 3:  return 16;
        case 4:  return 20;
        case 5:  return 12;
        case 6:  return 8;
        case 7:  return 0;
        case 9:  return 25;
        case 11: return 17;
        case 13: return 13;
        case 15: return 1;
        case 18: return 23;
        case 19: return 18;
        case 22: return 9;
        case 23: return 2;
        case 27: return 19;
        case 31: return 3;
        case 36: return 21;
        case 37: return 14;
        case 38: return 10;
        case 39: return 4;
        case 45: return 15;
        case 47: return 5;
        case 54: return 11;
        case 55: return 6;
        case 63: return 7;
    }

    AMM_error_invalid_arg(true,
                          "NodeConfig<3>::get_child_id_from_bounds_conversion_flags(%d : %s) got bounds flag!\n",
                          int(_), std::bitset<6>(_).to_string().c_str());
}


/// -----------------------------------------------------------------------------------
//! get occupancy flags for each child
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::VecOccMask
OctreeNodeCoreConfiguration<2>::occupancy_flags() {

    VecOccMask _(snChildren_nonstandard);

    _[0].init_val(1);          // {0}
    _[1].init_val(2);          // {1}
    _[2].init_val(4);          // {2}
    _[3].init_val(8);          // {3}
    _[4].init_list({0,1});
    _[5].init_list({2,3});
    _[6].init_list({0,2});
    _[7].init_list({1,3});

    return _;
}

template<>
inline OctreeNodeCoreConfiguration<3>::VecOccMask
OctreeNodeCoreConfiguration<3>::occupancy_flags() {

    VecOccMask _(snChildren_nonstandard);

    _[0].init_val(1);          // {0}
    _[1].init_val(2);          // {1}
    _[2].init_val(4);          // {2}
    _[3].init_val(8);          // {3}
    _[4].init_val(16);         // {4}
    _[5].init_val(32);         // {5}
    _[6].init_val(64);         // {6}
    _[7].init_val(128);        // {7}

    _[8].init_list({0,1});
    _[9].init_list({2,3});
    _[10].init_list({4,5});
    _[11].init_list({6,7});

    _[12].init_list({0,2});
    _[13].init_list({1,3});
    _[14].init_list({4,6});
    _[15].init_list({5,7});

    _[16].init_list({0,4});
    _[17].init_list({1,5});
    _[18].init_list({2,6});
    _[19].init_list({3,7});

    _[20].init_list({0,1,2,3});
    _[21].init_list({4,5,6,7});

    _[22].init_list({0,1,4,5});
    _[23].init_list({2,3,6,7});

    _[24].init_list({0,2,4,6});
    _[25].init_list({1,3,5,7});

    return _;
}


/// -----------------------------------------------------------------------------------
//! get child ids invalid with respect to dimensionality
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::VecChildMask
OctreeNodeCoreConfiguration<2>::conflicts_for_axes() {

    VecChildMask _ (snChildren);

    //_[toInt(TypeAxes::TypeAxes_Unknown] = 0;              // default is 0
    _[as_utype( EnumAxes::X)] .init_list({4,5     });
    _[as_utype( EnumAxes::Y)] .init_list({     6,7});
    _[as_utype( EnumAxes::XY)].init_list({4,5, 6,7});
    return _;
}

template<>
inline OctreeNodeCoreConfiguration<3>::VecChildMask
OctreeNodeCoreConfiguration<3>::conflicts_for_axes() {

    VecChildMask _ (snChildren);

    //_[toInt(TypeAxes::TypeAxes_Unknown] = 0;              // default is 0
    _[as_utype( EnumAxes::X)]  .init_list({8,9,10,11,                           20,21, 22,23       });
    _[as_utype( EnumAxes::Y)]  .init_list({           12,13,14,15,              20,21,        24,25});
    _[as_utype( EnumAxes::Z)]  .init_list({                        16,17,18,19,        22,23, 24,25});
    _[as_utype( EnumAxes::XY)] .init_list({8,9,10,11, 12,13,14,15,              20,21, 22,23, 24,25});
    _[as_utype( EnumAxes::XZ)] .init_list({8,9,10,11,              16,17,18,19, 20,21, 22,23, 24,25});
    _[as_utype( EnumAxes::YZ)] .init_list({           12,13,14,15, 16,17,18,19, 20,21, 22,23, 24,25});
    _[as_utype( EnumAxes::XYZ)].init_list({8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21, 22,23, 24,25});
    return _;
}


/// -----------------------------------------------------------------------------------
//! get components of a child respect to dimensionality
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::VecVecChildMask
OctreeNodeCoreConfiguration<2>::components_for_axes() {

    VecVecChildMask _(snChildren_nonstandard+1);
    for( TypeChildId i = 0; i <= snChildren_nonstandard; i++) {
        _[i].resize(4);
    }

    // all defaults are 0
    // change only the ones needed!

    _[4][as_utype( EnumAxes::X)].init_list({0,1});
    _[5][as_utype( EnumAxes::X)].init_list({2,3});
    _[6][as_utype( EnumAxes::Y)].init_list({0,2});
    _[7][as_utype( EnumAxes::Y)].init_list({1,3});

    _[8][as_utype( EnumAxes::X)].init_list({6,7});
    _[8][as_utype( EnumAxes::Y)].init_list({4,5});
    _[8][as_utype( EnumAxes::XY)].init_list({0,1,2,3});

    return _;
}

template<>
inline OctreeNodeCoreConfiguration<3>::VecVecChildMask
OctreeNodeCoreConfiguration<3>::components_for_axes() {

    VecVecChildMask _(snChildren_nonstandard+1);
    for( TypeChildId i = 0; i <= snChildren_nonstandard; i++) {
        _[i].resize(8);
    }

    // all defaults are 0
    // change only the ones needed!

    _[8][as_utype( EnumAxes::X)]  .init_list({0,1});
    _[8][as_utype( EnumAxes::XZ)] .init_list({0,1});
    _[8][as_utype( EnumAxes::XY)] .init_list({0,1});
    _[8][as_utype( EnumAxes::XYZ)].init_list({0,1});

    _[9][as_utype( EnumAxes::X)]  .init_list({2,3});
    _[9][as_utype( EnumAxes::XZ)] .init_list({2,3});
    _[9][as_utype( EnumAxes::XY)] .init_list({2,3});
    _[9][as_utype( EnumAxes::XYZ)].init_list({2,3});

    _[10][as_utype( EnumAxes::X)]  .init_list({4,5});
    _[10][as_utype( EnumAxes::XZ)] .init_list({4,5});
    _[10][as_utype( EnumAxes::XY)] .init_list({4,5});
    _[10][as_utype( EnumAxes::XYZ)].init_list({4,5});

    _[11][as_utype( EnumAxes::X)]  .init_list({6,7});
    _[11][as_utype( EnumAxes::XZ)] .init_list({6,7});
    _[11][as_utype( EnumAxes::XY)] .init_list({6,7});
    _[11][as_utype( EnumAxes::XYZ)].init_list({6,7});

    _[12][as_utype( EnumAxes::Y)]  .init_list({0,2});
    _[12][as_utype( EnumAxes::YZ)] .init_list({0,2});
    _[12][as_utype( EnumAxes::XY)] .init_list({0,2});
    _[12][as_utype( EnumAxes::XYZ)].init_list({0,2});

    _[13][as_utype( EnumAxes::Y)]  .init_list({1,3});
    _[13][as_utype( EnumAxes::YZ)] .init_list({1,3});
    _[13][as_utype( EnumAxes::XY)] .init_list({1,3});
    _[13][as_utype( EnumAxes::XYZ)].init_list({1,3});

    _[14][as_utype( EnumAxes::Y)]  .init_list({4,6});
    _[14][as_utype( EnumAxes::YZ)] .init_list({4,6});
    _[14][as_utype( EnumAxes::XY)] .init_list({4,6});
    _[14][as_utype( EnumAxes::XYZ)].init_list({4,6});

    _[15][as_utype( EnumAxes::Y)]  .init_list({5,7});
    _[15][as_utype( EnumAxes::YZ)] .init_list({5,7});
    _[15][as_utype( EnumAxes::XY)] .init_list({5,7});
    _[15][as_utype( EnumAxes::XYZ)].init_list({5,7});

    _[16][as_utype( EnumAxes::Z)]  .init_list({0,4});
    _[16][as_utype( EnumAxes::YZ)] .init_list({0,4});
    _[16][as_utype( EnumAxes::XZ)] .init_list({0,4});
    _[16][as_utype( EnumAxes::XYZ)].init_list({0,4});

    _[17][as_utype( EnumAxes::Z)]  .init_list({1,5});
    _[17][as_utype( EnumAxes::YZ)] .init_list({1,5});
    _[17][as_utype( EnumAxes::XZ)] .init_list({1,5});
    _[17][as_utype( EnumAxes::XYZ)].init_list({1,5});

    _[18][as_utype( EnumAxes::Z)]  .init_list({2,6});
    _[18][as_utype( EnumAxes::YZ)] .init_list({2,6});
    _[18][as_utype( EnumAxes::XZ)] .init_list({2,6});
    _[18][as_utype( EnumAxes::XYZ)].init_list({2,6});

    _[19][as_utype( EnumAxes::Z)]  .init_list({3,7});
    _[19][as_utype( EnumAxes::YZ)] .init_list({3,7});
    _[19][as_utype( EnumAxes::XZ)] .init_list({3,7});
    _[19][as_utype( EnumAxes::XYZ)].init_list({3,7});

    _[20][as_utype( EnumAxes::Y)]  .init_list({8,9});
    _[20][as_utype( EnumAxes::YZ)] .init_list({8,9});
    _[20][as_utype( EnumAxes::X)]  .init_list({12,13});
    _[20][as_utype( EnumAxes::XZ)] .init_list({12,13});
    _[20][as_utype( EnumAxes::XY)] .init_list({0,1,2,3});
    _[20][as_utype( EnumAxes::XYZ)].init_list({0,1,2,3});

    _[21][as_utype( EnumAxes::Y)]  .init_list({10,11});
    _[21][as_utype( EnumAxes::YZ)] .init_list({10,11});
    _[21][as_utype( EnumAxes::X)]  .init_list({14,15});
    _[21][as_utype( EnumAxes::XZ)] .init_list({14,15});
    _[21][as_utype( EnumAxes::XY)] .init_list({4,5,6,7});
    _[21][as_utype( EnumAxes::XYZ)].init_list({4,5,6,7});

    _[22][as_utype( EnumAxes::Z)]  .init_list({8,10});
    _[22][as_utype( EnumAxes::YZ)] .init_list({8,10});
    _[22][as_utype( EnumAxes::X)]  .init_list({16,17});
    _[22][as_utype( EnumAxes::XY)] .init_list({16,17});
    _[22][as_utype( EnumAxes::XZ)] .init_list({0,1,4,5});
    _[22][as_utype( EnumAxes::XYZ)].init_list({0,1,4,5});

    _[23][as_utype( EnumAxes::Z)]  .init_list({9,11});
    _[23][as_utype( EnumAxes::YZ)] .init_list({9,11});
    _[23][as_utype( EnumAxes::X)]  .init_list({18,19});
    _[23][as_utype( EnumAxes::XY)] .init_list({18,19});
    _[23][as_utype( EnumAxes::XZ)] .init_list({2,3,6,7});
    _[23][as_utype( EnumAxes::XYZ)].init_list({2,3,6,7});

    _[24][as_utype( EnumAxes::Z)]  .init_list({12,14});
    _[24][as_utype( EnumAxes::XZ)] .init_list({12,14});
    _[24][as_utype( EnumAxes::Y)]  .init_list({16,18});
    _[24][as_utype( EnumAxes::XY)] .init_list({16,18});
    _[24][as_utype( EnumAxes::YZ)] .init_list({0,2,4,6});
    _[24][as_utype( EnumAxes::XYZ)].init_list({0,2,4,6});

    _[25][as_utype( EnumAxes::Z)]  .init_list({13,15});
    _[25][as_utype( EnumAxes::XZ)] .init_list({13,15});
    _[25][as_utype( EnumAxes::Y)]  .init_list({17,19});
    _[25][as_utype( EnumAxes::XY)] .init_list({17,19});
    _[25][as_utype( EnumAxes::YZ)] .init_list({1,3,5,7});
    _[25][as_utype( EnumAxes::XYZ)].init_list({1,3,5,7});

    _[26][as_utype( EnumAxes::X)]  .init_list({24,25});
    _[26][as_utype( EnumAxes::Y)]  .init_list({22,23});
    _[26][as_utype( EnumAxes::Z)]  .init_list({20,21});
    _[26][as_utype( EnumAxes::XY)] .init_list({16,17,18,19});
    _[26][as_utype( EnumAxes::XZ)] .init_list({12,13,14,15});
    _[26][as_utype( EnumAxes::YZ)] .init_list({8,9,10,11});
    _[26][as_utype( EnumAxes::XYZ)].init_list({0,1,2,3,4,5,6,7});

    return _;
}


/// -----------------------------------------------------------------------------------
//! get conflicts of every child node
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::VecChildMask
OctreeNodeCoreConfiguration<2>::conflicts_for_child() {

    VecChildMask _(snChildren_nonstandard);

    // regular children
    _[0].init_list({ 0, 4,6});
    _[1].init_list({ 1, 4,7});
    _[2].init_list({ 2, 5,6});
    _[3].init_list({ 3, 5,7});

    // long in x dimension
    _[4].init_list({ 4, 0,1, 6,7});
    _[5].init_list({ 5, 2,3, 6,7});

    // long in y dimension
    _[6].init_list({ 6, 0,2, 4,5});
    _[7].init_list({ 7, 1,3, 4,5});
    return _;
}

template<>
inline OctreeNodeCoreConfiguration<3>::VecChildMask
OctreeNodeCoreConfiguration<3>::conflicts_for_child() {

    VecChildMask _(snChildren_nonstandard);

    // regular children
    _[0].init_list({ 0, 8,12,16, 20,22,24});
    _[1].init_list({ 1, 8,13,17, 20,22,25});
    _[2].init_list({ 2, 9,12,18, 20,23,24});
    _[3].init_list({ 3, 9,13,19, 20,23,25});
    _[4].init_list({ 4, 10,14,16, 21,22,24});
    _[5].init_list({ 5, 10,15,17, 21,22,25});
    _[6].init_list({ 6, 11,14,18, 21,23,24});
    _[7].init_list({ 7, 11,15,19, 21,23,25});

    // long in x dimension
    _[8].init_list({ 8, 0,1, 12,13,14,15, 16,17,18,19, 20,22, 24,25});
    _[9].init_list({ 9, 2,3, 12,13,14,15, 16,17,18,19, 20,23, 24,25});
    _[10].init_list({10, 4,5, 12,13,14,15, 16,17,18,19, 21,22, 24,25});
    _[11].init_list({11, 6,7, 12,13,14,15, 16,17,18,19, 21,23, 24,25});

    // long in y dimension
    _[12].init_list({12, 0,2,  8, 9,10,11, 16,17,18,19, 20,24, 22,23});
    _[13].init_list({13, 1,3,  8, 9,10,11, 16,17,18,19, 20,25, 22,23});
    _[14].init_list({14, 4,6,  8, 9,10,11, 16,17,18,19, 21,24, 22,23});
    _[15].init_list({15, 5,7,  8, 9,10,11, 16,17,18,19, 21,25, 22,23});

    // long in z dimension
    _[16].init_list({16, 0,4,  8, 9,10,11, 12,13,14,15, 22,24, 20,21});
    _[17].init_list({17, 1,5,  8, 9,10,11, 12,13,14,15, 22,25, 20,21});
    _[18].init_list({18, 2,6,  8, 9,10,11, 12,13,14,15, 23,24, 20,21});
    _[19].init_list({19, 3,7,  8, 9,10,11, 12,13,14,15, 23,25, 20,21});

    // long in xy dimensions
    _[20].init_list({20, 0,1,2,3,  8, 9, 12,13,   16,17,18,19, 22,23,24,25});
    _[21].init_list({21, 4,5,6,7, 10,11, 14,15,   16,17,18,19, 22,23,24,25});

    // long in xz dimensions
    _[22].init_list({22, 0,1,4,5,  8,10, 16,17,   12,13,14,15, 20,21,24,25});
    _[23].init_list({23, 2,3,6,7,  9,11, 18,19,   12,13,14,15, 20,21,24,25});

    // long in yz dimensions
    _[24].init_list({24, 0,2,4,6, 12,14, 16,18,    8, 9,10,11, 20,21,22,23});
    _[25].init_list({25, 1,3,5,7, 13,15, 17,19,    8, 9,10,11, 20,21,22,23});
    return _;
}


/// -----------------------------------------------------------------------------------
//! split child
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::TypeChildMask
OctreeNodeCoreConfiguration<2>::components_for_child(const  TypeChildId &to_split, const  TypeChildId &to_create) {

    switch (to_split) {

    // long in x axis
    case 4:
        switch(to_create) {
        case 0:
        case 1: return 3;               //TypeChildMask::create_mask({0,1});
        }   break;
    case 5:
        switch(to_create) {
        case 2:
        case 3: return 12;              //TypeChildMask::create_mask({2,3});
        }   break;
    case 6:
        switch(to_create) {
        case 0:
        case 2: return 5;               //TypeChildMask::create_mask({0,2});
        }   break;
    case 7:
        switch(to_create) {
        case 1:
        case 3: return 10;              //TypeChildMask::create_mask({1,3});
        }   break;
    }

    AMM_error_invalid_arg(true, "OctreeChildNodes<2>::split_child(%d,%d) got invalid input!\n", int(to_split), int(to_create));
}

template<>
inline OctreeNodeCoreConfiguration<3>::TypeChildMask
OctreeNodeCoreConfiguration<3>::components_for_child(const  TypeChildId &to_split, const  TypeChildId &to_create) {

    switch (to_split) {

    // long in x axis
    case 8:
        switch(to_create) {
        case 0:
        case 1: return 3;               //TypeChildMask::create_mask({0,1});
        }   break;
    case 9:
        switch(to_create) {
        case 2:
        case 3: return 12;              //TypeChildMask::create_mask({2,3});
        }   break;
    case 10:
        switch(to_create) {
        case 4:
        case 5: return 48;              //TypeChildMask::create_mask({4,5});
        }   break;
    case 11:
        switch(to_create) {
        case 6:
        case 7: return 192;             //TypeChildMask::create_mask({6,7});
        }   break;

    // long in y axis
    case 12:
        switch(to_create) {
        case 0:
        case 2: return 5;               //TypeChildMask::create_mask({0,2});
        }   break;
    case 13:
        switch(to_create) {
        case 1:
        case 3: return 10;              //TypeChildMask::create_mask({1,3});
        }   break;
    case 14:
        switch(to_create) {
        case 4:
        case 6: return 80;              //TypeChildMask::create_mask({4,6});
        }   break;
    case 15:
        switch(to_create) {
        case 5:
        case 7: return 160;             //TypeChildMask::create_mask({5,7});
        }   break;

    // long in z axis
    case 16:
        switch(to_create) {
        case 0:
        case 4: return 17;              //TypeChildMask::create_mask({0,4});
        }   break;
    case 17:
        switch(to_create) {
        case 1:
        case 5: return 34;              //TypeChildMask::create_mask({1,5});
        }   break;
    case 18:
        switch(to_create) {
        case 2:
        case 6: return 68;              //TypeChildMask::create_mask({2,6});
        }   break;
    case 19:
        switch(to_create) {
        case 3:
        case 7: return 136;             //TypeChildMask::create_mask({3,7});
        }   break;

    // long in xy plane
    case 20:
        switch(to_create) {
        case 0:
        case 1: return 515;             //TypeChildMask::create_mask({0,1,9});
        case 2:
        case 3: return 268;             //TypeChildMask::create_mask({2,3,8});
        case 8:
        case 9: return 768;             //TypeChildMask::create_mask({8,9});
        case 12:
        case 13:return 12288;           //TypeChildMask::create_mask({12,13});
        }   break;
    case 21:
        switch(to_create) {
        case 4:
        case 5: return 2096;            //TypeChildMask::create_mask({4,5,11});
        case 6:
        case 7: return 1216;            //TypeChildMask::create_mask({6,7,10});
        case 10:
        case 11:return 3072;            //TypeChildMask::create_mask({10,11});
        case 14:
        case 15:return 49152;           //TypeChildMask::create_mask({14,15});
        }   break;

    // long in xz plane
    case 22:
        switch(to_create) {
        case 0:
        case 1:     return 1027;        //TypeChildMask::create_mask({0,1,10});
        case 4:
        case 5:     return 304;         //TypeChildMask::create_mask({4,5,8});
        case 8:
        case 10:    return 1280;        //TypeChildMask::create_mask({8,10});
        case 16:
        case 17:    return 196608;      //TypeChildMask::create_mask({16,17});
        }   break;
    case 23:
        switch(to_create) {
        case 2:
        case 3:     return 2060;        //TypeChildMask::create_mask({2,3,11});
        case 6:
        case 7:     return 704;         //TypeChildMask::create_mask({6,7,9});
        case 9:
        case 11:    return 2560;        //TypeChildMask::create_mask({9,11});
        case 18:
        case 19:    return 786432;      //TypeChildMask::create_mask({18,19});
        }   break;

    // long in yz plane
    case 24:
        switch(to_create) {
        case 0:
        case 2:     return 16389;       //TypeChildMask::create_mask({0,2,14});
        case 4:
        case 6:     return 4176;        //TypeChildMask::create_mask({4,6,12});
        case 12:
        case 14:    return 20480;       //TypeChildMask::create_mask({12,14});
        case 16:
        case 18:    return 327680;      //TypeChildMask::create_mask({16,18});
        }   break;
    case 25:
        switch(to_create) {
        case 1:
        case 3:     return 32778;       //TypeChildMask::create_mask({1,3,15});
        case 5:
        case 7:     return 8352;        //TypeChildMask::create_mask({5,7,13});
        case 13:
        case 15:    return 40960;       //TypeChildMask::create_mask({13,15});
        case 17:
        case 19:    return 655360;      //TypeChildMask::create_mask({17,19});
        }   break;

    // no child exists!
    case 26:
        std::cout << "OctreeNodeConfiguration<3>::split_child: unused case!\n";
        exit(1);
        switch(to_create) {
        case 0:
        case 1:     return 2097667;     //TypeChildMask::create_mask({0,1,9,21});
        case 2:
        case 3:     return 2097420;     //TypeChildMask::create_mask({2,3,8,21});
        case 4:
        case 5:     return 1050672;     //TypeChildMask::create_mask({4,5,11,20});
        case 6:
        case 7:     return 1049792;     //TypeChildMask::create_mask({6,7,10,20});
        case 8:
        case 9:     return 2097920;     //TypeChildMask::create_mask({8,9,21});
        case 10:
        case 11:    return 1051648;     //TypeChildMask::create_mask({10,11,20});
        case 12:
        case 13:    return 2109440;     //TypeChildMask::create_mask({12,13,21});
        case 14:
        case 15:    return 1097728;     //TypeChildMask::create_mask({14,15,20});
        case 16:
        case 17:    return 8585216;     //TypeChildMask::create_mask({16,17,23});
        case 18:
        case 19:    return 4980736;     //TypeChildMask::create_mask({18,19,22});
        case 20:
        case 21:    return 3145728;     //TypeChildMask::create_mask({20,21});
        case 22:
        case 23:    return 12582912;    //TypeChildMask::create_mask({22,23});
        case 24:
        case 25:    return 50331648;    //TypeChildMask::create_mask({24,25});
        }   break;
    }

    AMM_error_invalid_arg(true,
                           "OctreeChildNodes<3>::split_child_for_another_child(%d,%d) got invalid input!\n",
                           int(to_split), int(to_create));
}


/// -----------------------------------------------------------------------------------
//! get split components
/// -----------------------------------------------------------------------------------

template<>
inline OctreeNodeCoreConfiguration<2>::VecChildMask
OctreeNodeCoreConfiguration<2>::node_components()  {

    VecChildMask _ (snChildren_nonstandard);

    _[0].init_val(1);          // {0}
    _[1].init_val(2);          // {1}
    _[2].init_val(4);          // {2}
    _[3].init_val(8);          // {3}
    _[4].init_val(3);          // {0,1}
    _[5].init_val(12);         // {2,3}
    _[6].init_val(5);          // {0,2}
    _[7].init_val(10);         // {1,3}

    return _;
}

template<>
inline OctreeNodeCoreConfiguration<3>::VecChildMask
OctreeNodeCoreConfiguration<3>::node_components()  {

    VecChildMask _ (snChildren_nonstandard);

    _[0].init_val(1);          // {0}
    _[1].init_val(2);          // {1}
    _[2].init_val(4);          // {2}
    _[3].init_val(8);          // {3}
    _[4].init_val(16);         // {4}
    _[5].init_val(32);         // {5}
    _[6].init_val(64);         // {6}
    _[7].init_val(128);        // {7}

    _[8].init_list({0,1});
    _[9].init_list({2,3});
    _[10].init_list({4,5});
    _[11].init_list({6,7});

    _[12].init_list({0,2});
    _[13].init_list({1,3});
    _[14].init_list({4,6});
    _[15].init_list({5,7});

    _[16].init_list({0,4});
    _[17].init_list({1,5});
    _[18].init_list({2,6});
    _[19].init_list({3,7});

    _[20].init_list({0,1,2,3,8,9,12,13});
    _[21].init_list({4,5,6,7,10,11,14,15});

    _[22].init_list({0,1,4,5,8,10,16,17});
    _[23].init_list({2,3,6,7,9,11,18,19});

    _[24].init_list({0,2,4,6,12,14,16,18});
    _[25].init_list({1,3,5,7,13,15,17,19});

    return _;
}

/// -----------------------------------------------------------------------------------
//! get merge components
/// -----------------------------------------------------------------------------------

#ifdef UNUSED
template<>
inline OctreeNodeConfiguration<2>::VecVecBitMask
OctreeNodeConfiguration<2>::merge_components() {

    using _Bm = typename OctreeNodeConfiguration<2>::TypeBitMask;

    VecVecBitMask _ (snChildren_nonstandard+1);

    _[0].push_back(_Bm::create_mask({0}));
    _[1].push_back(_Bm::create_mask({1}));
    _[2].push_back(_Bm::create_mask({2}));
    _[3].push_back(_Bm::create_mask({3}));
    _[4].push_back(_Bm::create_mask({0,1}));
    _[5].push_back(_Bm::create_mask({2,3}));
    _[6].push_back(_Bm::create_mask({0,2}));
    _[7].push_back(_Bm::create_mask({1,3}));
    _[8].push_back(_Bm::create_mask({4,5}));
    _[8].push_back(_Bm::create_mask({6,7}));

    return _;
}

template<>
inline OctreeNodeConfiguration<3>::VecVecBitMask
OctreeNodeConfiguration<3>::merge_components() {

    using _Bm = typename OctreeNodeConfiguration<2>::TypeBitMask;

    VecVecBitMask _ (snChildren_nonstandard+1);

    _[0].push_back(_Bm::create_mask({0}));
    _[1].push_back(_Bm::create_mask({1}));
    _[2].push_back(_Bm::create_mask({2}));
    _[3].push_back(_Bm::create_mask({3}));
    _[4].push_back(_Bm::create_mask({4}));
    _[5].push_back(_Bm::create_mask({5}));
    _[6].push_back(_Bm::create_mask({6}));
    _[7].push_back(_Bm::create_mask({7}));

    _[8].push_back(_Bm::create_mask({0,1}));
    _[9].push_back(_Bm::create_mask({2,3}));
    _[10].push_back(_Bm::create_mask({4,5}));
    _[11].push_back(_Bm::create_mask({6,7}));

    _[12].push_back(_Bm::create_mask({0,2}));
    _[13].push_back(_Bm::create_mask({1,3}));
    _[14].push_back(_Bm::create_mask({4,6}));
    _[15].push_back(_Bm::create_mask({5,7}));

    _[16].push_back(_Bm::create_mask({0,4}));
    _[17].push_back(_Bm::create_mask({1,5}));
    _[18].push_back(_Bm::create_mask({2,6}));
    _[19].push_back(_Bm::create_mask({3,7}));

    _[20].push_back(_Bm::create_mask({8,9}));
    _[20].push_back(_Bm::create_mask({12,13}));

    _[21].push_back(_Bm::create_mask({10,11}));
    _[21].push_back(_Bm::create_mask({14,15}));

    _[22].push_back(_Bm::create_mask({8,10}));
    _[22].push_back(_Bm::create_mask({16,17}));

    _[23].push_back(_Bm::create_mask({9,11}));
    _[23].push_back(_Bm::create_mask({18,19}));

    _[24].push_back(_Bm::create_mask({12,14}));
    _[24].push_back(_Bm::create_mask({16,18}));

    _[25].push_back(_Bm::create_mask({13,15}));
    _[25].push_back(_Bm::create_mask({17,19}));

    _[26].push_back(_Bm::create_mask({20,21}));
    _[26].push_back(_Bm::create_mask({22,23}));
    _[26].push_back(_Bm::create_mask({24,25}));

    return _;
}
#endif

}       // end of namespace
#endif
