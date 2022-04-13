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
#ifndef STREAM_STATE_H
#define STREAM_STATE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>

#include "amm/types/dtypes.hpp"
#include "amm/types/enums.hpp"
#include "enums.hpp"

namespace streams {

//! -------------------------------------------------------------------------------------
//! traits used in streams
//! -------------------------------------------------------------------------------------
// TODO: use amm::traits_bytes for these
template <typename t>
struct Traits;

template <>
struct Traits<int32_t> {
  using signed_t = int32_t;
  using unsigned_t = uint32_t;
  using floating_t = float;
  static constexpr uint32_t negabinary_mask = 0xaaaaaaaa;
};

template <>
struct Traits<int64_t> {
  using signed_t = int64_t;
  using unsigned_t = uint64_t;
  using floating_t = double;
  static constexpr uint64_t negabinary_mask = 0xaaaaaaaaaaaaaaaaULL;
};

template <>
struct Traits<float> {
  using signed_t = int32_t;
  static constexpr int exponent_bits = 8;
  static constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
};

template <>
struct Traits<double> {
  using signed_t = int64_t;
  static constexpr int exponent_bits = 11;
  static constexpr int exponent_bias = (1 << (exponent_bits - 1)) - 1;
};


//! -------------------------------------------------------------------------------------
//! additional utility strucutres
//! -------------------------------------------------------------------------------------

template <typename t>
struct WaveletCoeff {
  t Val;
  int64_t Pos;
};

struct extent {
  uint64_t Pos;
  uint64_t Dims;

  // added constructor to support emplace_back
  extent(const uint64_t &p, const uint64_t &d) :
      Pos(p), Dims(d) {}
};


//! -------------------------------------------------------------------------------------
//! main functionality for streaming
//! -------------------------------------------------------------------------------------

struct subband_bitplane {
  int Subband;
  int Bitplane;
  double Score;

  // added constructor to support emplace_back
  subband_bitplane(const int &sb, const int &bp, const double &s=0) :
      Subband(sb), Bitplane(bp), Score(s) {}
};

struct pos_pair {
  int8_t Subband;
  int32_t Pos; // within subband
  float Score;

  // added constructor to support emplace_back
  pos_pair(const int8_t &sb, const int32_t &p, const float &s) :
      Subband(sb), Pos(p), Score(s) {}
};

template <typename t>
struct state {
  std::vector<t> DataCopy; // in case we need to copy the data
  const t* DataPtr;
  using i = typename Traits<t>::signed_t;
  using u = typename Traits<i>::unsigned_t;
  int Nx, Ny, Nz;
  int Bits;
  int NumLevels;
  std::vector<extent> Subbands;
  std::vector<int8_t> Bitplanes;
  std::vector<int8_t> FirstOnes; // for all subbands
  std::vector<subband_bitplane> Order;
  std::vector<pos_pair> CoeffOrder;
  int64_t WavPos; // the position of the current coefficient in the wavelet domain
  int OrderPos; // where am I in Order
  int EMax;
  EnumArrangement Arrangement;
  bool SkipLeadingZeros;
  EnumStream Type;
  int BitsRead;
  std::ofstream OutFile;
};

//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
}
#endif
