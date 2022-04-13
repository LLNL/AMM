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
#ifndef STREAM_TYPES_H
#define STREAM_TYPES_H

#include "amm/utils/utils.hpp"
#include "amm/types/enums.hpp"
#include "amm/types/strformat.hpp"
#include "wavelets/enums.hpp"

//! ----------------------------------------------------------------------------
//! dtypes, enumerations, and associated utilities for streams
//! ----------------------------------------------------------------------------

//! types of stream
enum struct EnumStream: std::uint8_t {
    Unknown = 0,
    By_RowMajor = 1,
    By_Subband_Rowmajor = 2,
    By_Coeff_Wavelet_Norm = 3,
    By_Wavelet_Norm = 4,
    By_Level = 5,
    By_BitPlane = 6,
    By_Magnitude = 7
};

//! ----------------------------------------------------------------------------
namespace streams {

inline static bool
is_valid_stream(const EnumStream &stype) {
    //return EnumStream::Unknown != stype;
    switch(stype) {
       case EnumStream::By_RowMajor:               return true;    // spatial streams
       case EnumStream::By_Subband_Rowmajor:       return true;
       case EnumStream::By_Coeff_Wavelet_Norm:     return true;
       case EnumStream::By_Wavelet_Norm:           return true;    // precision streams
       case EnumStream::By_Level:                  return true;
       case EnumStream::By_BitPlane:               return true;
       case EnumStream::By_Magnitude:              return true;
       case EnumStream::Unknown:                   return false;
   }
    return false;
}

inline static bool
is_precision_stream(const EnumStream &stype) {
    return stype == EnumStream::By_Wavelet_Norm || stype == EnumStream::By_Level || stype == EnumStream::By_BitPlane;
}

inline static bool
is_spatial_stream(const EnumStream &stype) {
    return stype == EnumStream::By_RowMajor || stype == EnumStream::By_Subband_Rowmajor || stype == EnumStream::By_Coeff_Wavelet_Norm;
}

static std::string
get_stream_name(const EnumStream &stype) {
    switch(stype) {
        case EnumStream::By_RowMajor:               return "by_rowmajor";
        case EnumStream::By_Subband_Rowmajor:       return "by_subband_rowmajor";
        case EnumStream::By_Coeff_Wavelet_Norm:     return "by_coeff_wavelet_norm";
        case EnumStream::By_Wavelet_Norm:           return "by_wavelet_norm";
        case EnumStream::By_Level:                  return "by_level";
        case EnumStream::By_BitPlane:               return "by_bitplane";
        case EnumStream::By_Magnitude:              return "by_magnitude";
        case EnumStream::Unknown:                   return "unknown_stream";
    }
    return "unknown stream";
}

static EnumStream
get_stream_type(const std::string &sname) {

    if (AMM_is_same(sname, "by_rowmajor"))              return EnumStream::By_RowMajor;
    if (AMM_is_same(sname, "by_subband_rowmajor"))      return EnumStream::By_Subband_Rowmajor;
    if (AMM_is_same(sname, "by_coeff_wavelet_norm"))    return EnumStream::By_Coeff_Wavelet_Norm;
    if (AMM_is_same(sname, "by_wavelet_norm"))          return EnumStream::By_Wavelet_Norm;
    if (AMM_is_same(sname, "by_level"))                 return EnumStream::By_Level;
    if (AMM_is_same(sname, "by_bitplane"))              return EnumStream::By_BitPlane;
    if (AMM_is_same(sname, "by_magnitude"))             return EnumStream::By_Magnitude;
    return EnumStream::Unknown;
}

static EnumArrangement
get_stream_suitable_arrangement(const EnumStream &stype) {
    return stype != EnumStream::Unknown ? EnumArrangement::Original : EnumArrangement::Unknown;
}

}
//! ----------------------------------------------------------------------------

#endif
