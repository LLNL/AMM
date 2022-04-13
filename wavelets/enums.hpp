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
#ifndef WAVELET_TYPES_H
#define WAVELET_TYPES_H

#include <string>

//! ----------------------------------------------------------------------------
//! dtypes, enumerations, and associated utilities for wavelets
//! ----------------------------------------------------------------------------

//! type of wavelet transform
enum struct EnumWavelet: std::uint8_t {
    Unknown = 0,
    Interpolating = 1,
    Approximating = 2
};

//! type of sample arrangement
enum struct EnumArrangement: std::uint8_t {
    Unknown = 0,
    Original = 1,
    Subband = 2
};


//! ----------------------------------------------------------------------------
namespace wavelets {

static EnumArrangement
get_arrangement_type(const std::string &sname) {

    if (0 == sname.compare("Original"))     return EnumArrangement::Original;
    if (0 == sname.compare("Subband"))      return EnumArrangement::Subband;
    return EnumArrangement::Unknown;
}

static std::string
get_arrangement_name(const EnumArrangement &stype) {
    switch(stype) {
        case EnumArrangement::Original:     return "Original";
        case EnumArrangement::Subband:      return "Subband";
        case EnumArrangement::Unknown:      return "unknown_arrangement";
    }
    return "unknown_arrangement";
}

static std::string
get_wavelets_name(const EnumWavelet &stype) {
    switch(stype) {
        case EnumWavelet::Interpolating:    return "Interpolating";
        case EnumWavelet::Approximating:    return "Approximating";
        case EnumWavelet::Unknown:          return "unknown_wavelet_type";
    }
    return "unknown_wavelet_type";
}

}
//! ----------------------------------------------------------------------------

#endif
