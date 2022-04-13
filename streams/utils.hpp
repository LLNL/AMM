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
#ifndef STREAM_UTILS_H
#define STREAM_UTILS_H

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

#include "amm/types/dtypes.hpp"
#include "wavelets/enums.hpp"
#include "enums.hpp"
#include "state.hpp"


template <typename T>
void profile_vector(const std::vector<T> &d, const std::string &name = "") {
    std::cout<<"> vector<"<<name<<">: size="<<d.size()<<", cap="<<d.capacity()<<", dtype="<<sizeof(T)<<"\n";
}

namespace streams {
//! -------------------------------------------------------------------------------------
//! hardcoded constants for wavelet norms
//! -------------------------------------------------------------------------------------

std::vector<double> WaveletNorms = {
  0.84779124789065852,
  0.96014321848357598,
  1.25934010497561788,
  1.74441071711910789,
  2.45387130367507256,
  3.46565176950887555,
  4.89952763985978557,
  6.92839704021608416,
  9.79802749401314443,
  13.85643068711126524,
  19.59592650765355870,
  27.71281594942454873,
  39.19183695520458599,
  55.42562622074440526,
  78.38367190289591235,
  110.85125173172568225
};

std::vector<double> ScalingNorms = {
  1.22474487139158894,
  1.65831239517769990,
  2.31840462387392598,
  3.26917420765550526,
  4.61992965314408188,
  6.53237131522695957,
  9.23774526061419365,
  13.06399512974495813,
  18.47522623339156667,
  26.12789681906103922,
  36.95041943055247913,
  52.25578195804627768,
  73.90083473157416449,
  104.51156245608291329,
  147.80166894695696556,
  209.02312472966460177
};

std::array<uint8_t, 8> SubbandOrders[4] = {
{ 127, 127, 127, 127, 127, 127, 127, 127 }, // not used
{ 0, 1, 127, 127, 127, 127, 127, 127 },     // for 1D
{ 0, 1, 2, 3, 127, 127, 127, 127 },         // for 2D
{ 0, 1, 2, 4, 3, 5, 6, 7 }                  // for 3D
};

//! -------------------------------------------------------------------------------------
//! basic utilities for streams
//! -------------------------------------------------------------------------------------

/* Return the bit plane of the most significant one-bit. Counting starts from the least significant
bit plane. Examples: Bsr(0) = -1, Bsr(2) = 1, Bsr(5) = 2, Bsr(8) = 3 */
#if defined(__clang__) || defined(__GNUC__)
int8_t Msb(uint32_t V) {
  if (V == 0) return -1;
  return int8_t(sizeof(V) * 8 - 1 - __builtin_clz(V));
}
int8_t Msb(uint64_t V) {
  if (V == 0) return -1;
  return int8_t(sizeof(V) * 8 - 1 - __builtin_clzll(V));
}
#elif defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)
int8_t Msb(uint32_t V) {
  if (V == 0) return -1;
  unsigned long Index = 0;
  _BitScanReverse(&Index, V);
  return (int8_t)Index;
}
int8_t Msb(uint64_t V) {
  if (V == 0) return -1;
  unsigned long Index = 0;
  _BitScanReverse64(&Index, V);
  return (int8_t)Index;
}
#endif

template <typename t>
bool IsOdd(t V) { return (V & 1) == 1; }

template <typename t>
bool CheckBit(t Val, int I) {
  assert(I < (int)(sizeof(t) * 8));
  return 1 & (Val >> I);
}

template <typename t>
int Exponent(t Val) {
  if (Val > 0) {
    int E;
    frexp(Val, &E);
    /* clamp exponent in case Val is denormal */
    return std::max(E, 1 - streams::Traits<t>::exponent_bias);
  }
  return -streams::Traits<t>::exponent_bias;
}

template <typename t>
void ConvertToNegabinary(const t* FIn, int64_t Size, typename streams::Traits<t>::unsigned_t* FOut) {
  for (int64_t I = 0; I < Size; ++I) {
    auto Mask = streams::Traits<t>::negabinary_mask;
    FOut[I] = (typename streams::Traits<t>::unsigned_t)((FIn[I] + Mask) ^ Mask);
  }
}

template <typename t>
void ConvertFromNegabinary(const typename streams::Traits<t>::unsigned_t* FIn, int64_t Size, t* FOut) {
  for (int64_t I = 0; I < Size; ++I) {
    auto Mask = streams::Traits<t>::negabinary_mask;
    FOut[I] = (t)((FIn[I] ^ Mask) - Mask);
  }
}

template <typename t>
int GetEMax(const t* FIn, int64_t Size) {
  t Max = *std::max_element(FIn, FIn + Size, [](auto A, auto B) { return std::abs(A) < std::abs(B); });
  int EMax = Exponent(std::abs(Max));
  return EMax;
}

template <typename t>
void QuantizeWithEMax(const t* FIn, int64_t Size, int Bits, int EMax, typename streams::Traits<t>::signed_t* FOut) {
  double Scale = ldexp(1, Bits - 1 - EMax);
  for (int64_t I = 0; I < Size; ++I)
    FOut[I] = (typename streams::Traits<t>::signed_t)(Scale * FIn[I]);
}

template <typename t>
void Quantize(const t* FIn, int64_t Size, int Bits, typename streams::Traits<t>::signed_t* FOut) {
  int EMax = GetEMax(FIn, Size);
  QuantizeWithEMax(FIn, Size, Bits, EMax, FOut);
}

template <typename t>
void Dequantize(const t* FIn, int64_t Size, int EMax, int Bits, typename streams::Traits<t>::floating_t* FOut) {

  double Scale = 1.0 / ldexp(1, Bits - 1 - EMax);
  for (int64_t I = 0; I < Size; ++I)
    FOut[I] = (typename streams::Traits<t>::floating_t)(Scale * FIn[I]);
}

uint32_t Stuff3Ints32(int X, int Y, int Z) {
  return uint32_t(X) + (uint32_t(Y) << 10) + (uint32_t(Z) << 20);
}

std::array<int, 3> Extract3Ints32(uint32_t V) {
  return { int(V & 0x3FF), int((V & 0xFFC00) >> 10), int((V & 0x3FFFFC00) >> 20) };
}

uint64_t Stuff3Ints(int X, int Y, int Z) {
  return uint64_t(X) + (uint64_t(Y) << 21) + (uint64_t(Z) << 42);
}

std::array<int, 3> Extract3Ints(uint64_t V) {
  return { int(V & 0x1FFFFF), int((V & 0x3FFFFE00000) >> 21), int((V & 0x7FFFFC0000000000ull) >> 42) };
}

int64_t XyzToI(int Nx, int Ny, int Nz, int Px, int Py, int Pz) {
  return int64_t(Pz) * Nx * Ny + int64_t(Py) * Nx + Px;
}

std::array<int, 3> IToXyz(int64_t I, int Nx, int Ny, int Nz) {
  int Z = int(I / (Nx * Ny));
  int X = I % Nx;
  int Y = int((I - int64_t(Z) * (Nx * Ny)) / Nx);
  return { X, Y, Z };
}

//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------

/* (In 3D) Given a coefficient position(xyz), return <from(xyz), level(xyz)>, for octree-style subband decomposition */
std::tuple<int, int, int, int, int, int>
WaveletLevel(int Cx, int Cy, int Cz, int Nx, int Ny, int Nz, int NumLevels) {
  int Fromx = Nx, Fromy = Ny, Fromz = Nz;
  int Lx = 0, Ly = 0, Lz = 0;
  bool Foundx = Fromx <= Cx;
  bool Foundy = Fromy <= Cy;
  bool Foundz = Fromz <= Cz;
  for (int I = 0; I < NumLevels && !(Foundx || Foundy || Foundz); ++I) {
    Fromx = (Fromx + 1) >> 1;
    Fromy = (Fromy + 1) >> 1;
    Fromz = (Fromz + 1) >> 1;
    Lx += Fromx > Cx;
    Ly += Fromy > Cy;
    Lz += Fromz > Cz;
    Foundx = Fromx <= Cx;
    Foundy = Fromy <= Cy;
    Foundz = Fromz <= Cz;
  }
  Fromx *= Foundx;
  Fromy *= Foundy;
  Fromz *= Foundz;
  Lx = NumLevels - Lx;
  Ly = NumLevels - Ly;
  Lz = NumLevels - Lz;
  return { Fromx, Fromy, Fromz, Lx, Ly, Lz };
}

/* (In 3D) From a coefficient position, return the corresponding sample position(xyz) */
std::array<int, 3>
CoeffToPos(int Cx, int Cy, int Cz, int Nx, int Ny, int Nz, int NumLevels) {
  int Fromx, Fromy, Fromz, Levelx, Levely, Levelz;
  std::tie(Fromx, Fromy, Fromz, Levelx, Levely, Levelz) = WaveletLevel(Cx, Cy, Cz, Nx, Ny, Nz, NumLevels);
  int L = std::max(std::max(Levelx, Levely), Levelz);
  /* find the middle index */
  Cx -= Fromx;
  Cy -= Fromy;
  Cz -= Fromz;
  if (L != 0) {
    Cx = Cx * 2 + (Levelx == L);
    Cy = Cy * 2 + (Levely == L);
    Cz = Cz * 2 + (Levelz == L);
  }
  Cx <<= (NumLevels - L);
  Cy <<= (NumLevels - L);
  Cz <<= (NumLevels - L);
  return { Cx, Cy, Cz };
}

inline std::uint64_t
SplitBy2(std::uint64_t X) {
    X &= 0x00000000001fffffULL;
    X = (X | X << 32) & 0x001f00000000ffffULL;
    X = (X | X << 16) & 0x001f0000ff0000ffULL;
    X = (X | X << 8) & 0x100f00f00f00f00fULL;
    X = (X | X << 4) & 0x10c30c30c30c30c3ULL;
    X = (X | X << 2) & 0x1249249249249249ULL;
    return X;
}

inline std::uint64_t
EncodeMorton3(unsigned int X, unsigned int Y, unsigned int Z) {
    return SplitBy2(X) | (SplitBy2(Y) << 1) | (SplitBy2(Z) << 2);
}

inline std::uint32_t
CompactBy2(std::uint64_t X) {
    X &= 0x1249249249249249ULL;
    X = (X ^ (X >> 2)) & 0x10c30c30c30c30c3ULL;
    X = (X ^ (X >> 4)) & 0x100f00f00f00f00fULL;
    X = (X ^ (X >> 8)) & 0x001f0000ff0000ffULL;
    X = (X ^ (X >> 16)) & 0x001f00000000ffffULL;
    X = (X ^ (X >> 32)) & 0x00000000001fffffULL;
    return (std::uint32_t)X;
}

inline std::array<std::uint32_t, 3>
DecodeMorton3(std::uint64_t Code) {
    return std::array<std::uint32_t, 3>{CompactBy2(Code >> 0), CompactBy2(Code >> 1), CompactBy2(Code >> 2)};
}

std::array<int, 3>
Pos2Levels(unsigned int X, unsigned int Y, unsigned int Z, int Lmax) {    
    std::uint64_t M = EncodeMorton3(X, Y, Z);
    if (M == 0) {
        return std::array<int, 3>{Lmax, Lmax, Lmax};
    }
    std::uint64_t N = M;
    int L = 0;
    while ((N & 0x7) == 0) {
        N >>= 3;
        ++L;
    }
    int Lx = L, Ly = L, Lz = L;
    Lz += (N & 1) > 0;
    Ly += (N & 2) > 0;
    Lx += (N & 4) > 0;
    return std::array<int, 3>{Lx, Ly, Lz};
}

//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
std::array<int, 3> SubbandToLevel(int NDims, int Sb) {
  assert(NDims == 1 || NDims == 2 || NDims == 3);
  if (Sb == 0) {
    return { 0, 0, 0 };
  }
  int D = (1 << NDims) - 1;
  int Lvl = (Sb + D - 1) / D; // handle the special case of level 0 which has only 1 subband
                              // (levels 1, 2, .. has D subbands)
  Sb -= D * (Lvl - 1); // subtract all subbands on previous levels (except the subband 0);
                       // basically it reduces the case to the 2x2x2 case where subband 0 is in corner
                       // bit 0 -> x axis offset; bit 1 -> y axis offset; bit 2 -> z axis offset
                       // we subtract from lvl as it corresponds to the +x,+y,+z corner
  if (NDims == 1)
    return { Sb, 0, 0 };
  else if (NDims == 2)
    return { Lvl - !CheckBit(Sb, 1), Lvl - !CheckBit(Sb, 0) };
  return { Lvl - !CheckBit(Sb, 2), Lvl - !CheckBit(Sb, 1), Lvl - !CheckBit(Sb, 0) };
}

/* From wavelet domain to original domain */
std::array<int, 3> TranslateCoord(
  int Cx, int Cy, int Cz, int Nx, int Ny, int Nz, int NumLevels, EnumArrangement Arrangement) {
  if (Arrangement == EnumArrangement::Subband)
    return { Cx, Cy, Cz };
  return CoeffToPos(Cx, Cy, Cz, Nx, Ny, Nz, NumLevels);
}

inline int NumDims(int Nx, int Ny, int Nz) {
  if (Ny == 1 && Nz == 1)
    return 1;
  if (Nz == 1)
    return 2;
  return 3;
}

/* Here we assume the wavelet transform is done in X, then Y, then Z */
void BuildSubbands(int NDims, int Nx, int Ny, int Nz, int NumLevels, std::vector<extent>* Subbands) {
  assert(NDims <= 3);
  assert(Nz == 1 || NDims == 3);
  assert(Ny == 1 || NDims >= 2);
  const std::array<uint8_t, 8>& Order = SubbandOrders[NDims];
  Subbands->reserve(((1 << NDims) - 1) * NumLevels + 1);
  int Mx = Nx, My = Ny, Mz = Nz;
  for (int I = 0; I < NumLevels; ++I) {
    int Px = (Mx + 1) >> 1, Py = (My + 1) >> 1, Pz = (Mz + 1) >> 1;
    for (int J = (1 << NDims) - 1; J > 0; --J) {
      uint8_t Z = Order[J] & 1u,
              Y = (Order[J] >> (NDims - 2)) & 1u,
              X = (Order[J] >> (NDims - 1)) & 1u;
      int Sx = (X == 0) ? Px : Mx - Px,
          Sy = (Y == 0) ? Py : My - Py,
          Sz = (Z == 0) ? Pz : Mz - Pz;
      if (NDims == 3 && Sx != 0 && Sy != 0 && Sz != 0) // child exists
        Subbands->emplace_back(Stuff3Ints(X * Px, Y * Py, Z * Pz), Stuff3Ints(Sx, Sy, Sz));
      else if (NDims == 2 && Sx != 0 && Sy != 0) // child exists
        Subbands->emplace_back(Stuff3Ints(X * Px, Y * Py, 0), Stuff3Ints(Sx, Sy, 1));
    }
    Mx = Px; My = Py, Mz = Pz;
  }
  Subbands->emplace_back(0, Stuff3Ints(Mx, My, Mz));
  std::reverse(Subbands->begin(), Subbands->end());
}

//! -------------------------------------------------------------------------------------
//! main functionality for streaming
//! -------------------------------------------------------------------------------------
template <typename t>
struct coeff_litest {
    t val;
    size_t widx;
    coeff_litest(const t& _v, const size_t &_w) : val(_v), widx(_w) {}
};
template <typename t>
struct coeff_lite {
    t val;
    size_t widx;
    double score;
    coeff_lite(const t& _v, const size_t &_w) : val(_v), widx(_w), score(0) {}
    coeff_lite(const t& _v, const size_t &_w, const double& _score) : val(_v), widx(_w), score(_score) {}
};
template <typename t>
void BuildStream(
  const t* DataPtr, std::vector<coeff_lite<t>>* SortedData, int Nx, int Ny, int Nz, int NumLevels, 
  EnumArrangement Arrangement, EnumStream Type, int Bits, bool SkipLeadingZeroes, state<t>* State)
{
  using i = typename streams::Traits<t>::signed_t;
  using u = typename Traits<i>::unsigned_t;
  int64_t NumSamples = (int64_t)Nx * (int64_t)Ny * (int64_t)Nz;
  State->DataPtr = DataPtr;
  State->EMax = GetEMax(State->DataPtr, NumSamples);
  State->Nx = Nx;
  State->Ny = Ny;
  State->Nz = Nz;
  State->Bits = Bits;
  State->Arrangement = Arrangement;
  State->NumLevels = NumLevels;
  State->SkipLeadingZeros = SkipLeadingZeroes;
  State->Type = Type;
  State->BitsRead = 0;
  int NDims = (Nx > 1) + (Ny > 1) + (Nz > 1);
  BuildSubbands(NDims, Nx, Ny, Nz, NumLevels, &State->Subbands);
  State->WavPos = State->OrderPos = 0;
  if (SkipLeadingZeroes) {
    State->Bitplanes.resize(int64_t(Nx) * int64_t(Ny) * int64_t(Nz), -2);
    State->FirstOnes.resize(State->Subbands.size(), -1);
    for (int Sb = 0; Sb < State->Subbands.size(); ++Sb) {
      auto SbPos3 = Extract3Ints(State->Subbands[Sb].Pos);
      auto SbDims3 = Extract3Ints(State->Subbands[Sb].Dims);
      for (int Z = SbPos3[2]; Z < SbPos3[2] + SbDims3[2]; ++Z) {
        for (int Y = SbPos3[1]; Y < SbPos3[1] + SbDims3[1]; ++Y) {
          for (int X = SbPos3[0]; X < SbPos3[0] + SbDims3[0]; ++X) {
            auto Pos3 = TranslateCoord(X, Y, Z, Nx, Ny, Nz, NumLevels, Arrangement);
            int64_t Idx = XyzToI(Nx, Ny, Nz, Pos3[0], Pos3[1], Pos3[2]);
            u NegabinaryCoeff;
            i QuantizedCoeff;
            QuantizeWithEMax(&State->DataPtr[Idx], 1, State->Bits - 1, State->EMax, &QuantizedCoeff);
            ConvertToNegabinary(&QuantizedCoeff, 1, &NegabinaryCoeff);
            State->FirstOnes[Sb] = std::max(State->FirstOnes[Sb], Msb(NegabinaryCoeff));
          }
        }
      }
    }
  }

  const size_t sz_subbands = State->Subbands.size();

  if (Type == EnumStream::By_Level) {
    State->Order.reserve(sz_subbands*Bits);
    for (size_t L = 0; L < sz_subbands; ++L)
      for (int B = 0; B < Bits; ++B)
        State->Order.emplace_back(L, B);
  }
  else if (Type == EnumStream::By_Subband_Rowmajor) {
    State->Order.reserve(sz_subbands);
    for (size_t L = 0; L < sz_subbands; ++L)
      State->Order.emplace_back(L, 0);
  }
  else if (Type == EnumStream::By_BitPlane) {
    State->Order.reserve(sz_subbands*Bits);
    for (int B = 0; B < Bits; ++B)
      for (size_t L = 0; L < sz_subbands; ++L)
        State->Order.emplace_back(L, B);
  }
  else if (Type == EnumStream::By_Wavelet_Norm) {
    State->Order.reserve(sz_subbands*Bits);
    for (size_t Sb = 0; Sb < sz_subbands; ++Sb) {
      double Score = 0;
      if (Sb == 0) {
          double Sx = State->Nx > 1 ? ScalingNorms[State->NumLevels - 1] : 1;
          double Sy = State->Ny > 1 ? ScalingNorms[State->NumLevels - 1] : 1;
          double Sz = State->Nz > 1 ? ScalingNorms[State->NumLevels - 1] : 1;
          Score = (Sx * Sy * Sz);
      }
      else {
          int NDims = (State->Nx > 1) + (State->Ny > 1) + (State->Nz > 1);
          auto L = SubbandToLevel(NDims, Sb);
          int Lmax = std::max(std::max(L[0], L[1]), L[2]);
          double Sx = State->Nx == 1 ? 1 : ((L[0] == Lmax) ? WaveletNorms[State->NumLevels - Lmax]
              : ScalingNorms[State->NumLevels - Lmax]);
          double Sy = State->Ny == 1 ? 1 : ((L[1] == Lmax) ? WaveletNorms[State->NumLevels - Lmax]
              : ScalingNorms[State->NumLevels - Lmax]);
          double Sz = State->Nz == 1 ? 1 : ((L[2] == Lmax) ? WaveletNorms[State->NumLevels - Lmax]
              : ScalingNorms[State->NumLevels - Lmax]);
          Score = Sx * Sy * Sz;
      }
      for (int B = 0; B < State->Bits; ++B) {
        State->Order.emplace_back(Sb, B, Score * (1ull << (State->Bits - 1 - B)));
      }
    }
    // HB changed from stable_sort to sort
    std::sort(State->Order.begin(), State->Order.end(), [](const auto& E1, const auto& E2) {
      return E1.Score > E2.Score;
    });
  }
  else if (Type == EnumStream::By_Coeff_Wavelet_Norm) {
      size_t xy = State->Nx * State->Ny;
      size_t N = size_t(Nx) * size_t(Ny) * size_t(Nz);
      for (size_t i = 0; i < N; ++i) {
          if (DataPtr[i] == 0)
              continue;
          size_t z = i / xy;
          size_t y = (i % xy) / (State->Nx);
          size_t x = (i % xy) % (State->Nx);
          std::array<int, 3> c = Pos2Levels(x, y, z, State->NumLevels);
          double Score = 0;
          int Lmax = std::max(std::max(c[0], c[1]), c[2]);
          double Sx = State->Nx == 1 ? 1 : ((c[0] == Lmax) ? WaveletNorms[Lmax]
                                                           : ScalingNorms[Lmax]);
          double Sy = State->Ny == 1 ? 1 : ((c[1] == Lmax) ? WaveletNorms[Lmax]
                                                           : ScalingNorms[Lmax]);
          double Sz = State->Nz == 1 ? 1 : ((c[2] == Lmax) ? WaveletNorms[Lmax]
                                                           : ScalingNorms[Lmax]);
          Score = Sx * Sy * Sz * fabs(DataPtr[i]);
          // DUONG: TODO: pass the threshold here
          SortedData->emplace_back(DataPtr[i], i, Score);
      }
      // HB changed from stable_sort to sort
      std::sort(SortedData->rbegin(), SortedData->rend(), [](const auto& c1, const auto& c2) {
          return c1.score < c2.score;
      });
  }
}


template <typename t>
bool NextCoefficients(state<t>* State, std::vector<t>* Coeffs,
  std::vector<std::array<int, 3>>* CoeffPos3, int* ncoeffs, int* nbits, int NumCoefficientsWanted = 1) {
  assert(NumCoefficientsWanted >= 1);
  if (State->Type != EnumStream::By_Level && State->Type != EnumStream::By_BitPlane &&
      State->Type != EnumStream::By_Wavelet_Norm) {
    std::cerr << "Stream type not supported\n";
    exit(EXIT_FAILURE);
  }
  Coeffs->clear();
  CoeffPos3->clear();
  *ncoeffs = *nbits = 0;
  // Get the next bit from the correct coefficients
  using i = typename streams::Traits<t>::signed_t;
  using u = typename Traits<i>::unsigned_t;

  int NumCoefficientsRead = 0;
  int NumBitsRead = 0;
  while (NumCoefficientsRead < NumCoefficientsWanted) {
    /* if we have no more item in Order to go to, exit the loop */
    if (State->OrderPos >= State->Order.size()) {
      break;
    }
    int Sb = State->Order[State->OrderPos].Subband;
    int Bp = State->Order[State->OrderPos].Bitplane;
    if (State->SkipLeadingZeros && State->Bits - State->FirstOnes[Sb] - 1 > Bp) {
      ++State->OrderPos;
      State->WavPos = 0;
      //State->OutFile << "One ";
      continue;
    }
    auto SbPos3 = Extract3Ints(State->Subbands[Sb].Pos);
    auto SbDims3 = Extract3Ints(State->Subbands[Sb].Dims);
    /* get the bound of the current subband */
    auto WavPosLocal3 = IToXyz(State->WavPos, SbDims3[0], SbDims3[1], SbDims3[2]); // in wavelet domain
    std::array<int, 3> WavPosGlobal3{ WavPosLocal3[0] + SbPos3[0],
      WavPosLocal3[1] + SbPos3[1],
      WavPosLocal3[2] + SbPos3[2] };
    int64_t WavPosGlobal = XyzToI(State->Nx, State->Ny, State->Nz,
      WavPosGlobal3[0], WavPosGlobal3[1], WavPosGlobal3[2]);
    /* if the next coefficient is out of this bound, move to the next item in Order */
    if ((SbDims3[0] <= WavPosLocal3[0]) || (SbDims3[1] <= WavPosLocal3[1]) || (SbDims3[2] <= WavPosLocal3[2])) {
      ++State->OrderPos;
      State->WavPos = 0;
      continue;
    }
    /* now that we have found a coefficient, zero out its low-order bits and push to the std::vector */
    auto Pos3 = TranslateCoord(WavPosGlobal3[0], WavPosGlobal3[1], WavPosGlobal3[2],
      State->Nx, State->Ny, State->Nz, State->NumLevels, State->Arrangement);
    int64_t Pos = XyzToI(State->Nx, State->Ny, State->Nz, Pos3[0], Pos3[1], Pos3[2]);
    bool Visited = State->SkipLeadingZeros ? State->Bitplanes[Pos] != -2 : false;
    if (State->SkipLeadingZeros && Visited && (State->Bits - State->Bitplanes[Pos] - 1 > Bp)) {
      // move to the next coefficient and continue
      ++State->WavPos;
      continue;
    }
    int Shift = State->Bits - 1 - Bp;
    assert(0 <= Shift && Shift <= 63);
    u NegabinaryCoeff;
    i QuantizedCoeff;
    t FloatCoeff;
    QuantizeWithEMax(&State->DataPtr[Pos], 1, State->Bits - 1, State->EMax, &QuantizedCoeff);
    ConvertToNegabinary(&QuantizedCoeff, 1, &NegabinaryCoeff);
    if (State->SkipLeadingZeros && !Visited) {
      State->Bitplanes[Pos] = Msb(NegabinaryCoeff);
      if (State->Bits - State->Bitplanes[Pos] - 1 > Bp) { //
        ++State->WavPos;
        continue;
      }
    }
    u ShiftedNegabinaryCoeff = (NegabinaryCoeff >> Shift) << Shift;
    ConvertFromNegabinary(&ShiftedNegabinaryCoeff, 1, &QuantizedCoeff);
    Dequantize(&QuantizedCoeff, 1, State->EMax, State->Bits - 1, &FloatCoeff);
    // compute the previous value
    if (Shift < sizeof(u) * 8 - 1) {
      u PrevShiftedNegabinaryCoeff = (NegabinaryCoeff >> (Shift + 1)) << (Shift + 1);
      ConvertFromNegabinary(&PrevShiftedNegabinaryCoeff, 1, &QuantizedCoeff);
      t PrevFloatCoeff;
      Dequantize(&QuantizedCoeff, 1, State->EMax, State->Bits - 1, &PrevFloatCoeff);
      FloatCoeff = FloatCoeff - PrevFloatCoeff;
    }

    /* add the coefficient and its position to the output */
    Coeffs->push_back(FloatCoeff);
    CoeffPos3->push_back(Pos3);
    ++State->WavPos;
    ++NumCoefficientsRead;
    ++State->BitsRead;
    ++NumBitsRead;
  }
  *ncoeffs = NumCoefficientsRead;
  *nbits = NumBitsRead;
  return !Coeffs->empty();
}

/* in-out param: ncoeffs = number of coefficients read 
*  in-out param: nbits   = number of bits read */
template <typename t>
bool NextCoefficientsFullPrecision(state<t>* State, std::vector<t>* Coeffs,
  std::vector<std::array<int, 3>>* CoeffPos3, double threshold, int* ncoeffs, int* nbits, int NumCoefficientsWanted = 1) {
  assert(NumCoefficientsWanted >= 1);
  Coeffs->clear();
  CoeffPos3->clear();
  Coeffs->reserve(NumCoefficientsWanted);
  CoeffPos3->reserve(NumCoefficientsWanted);
  *ncoeffs = *nbits = 0;
  /* threshold / count / bytes */
  if (State->Type == EnumStream::By_Subband_Rowmajor) {
    int NumCoefficientsRead = 0;
    while (NumCoefficientsRead < NumCoefficientsWanted) {
      /* if we have no more item in Order to go to, exit the loop */
      if (State->OrderPos >= State->Order.size())
        break;
      int Sb = State->Order[State->OrderPos].Subband;
      auto SbPos3 = Extract3Ints(State->Subbands[Sb].Pos);
      auto SbDims3 = Extract3Ints(State->Subbands[Sb].Dims);
      /* get the bound of the current subband */
      auto WavPosLocal3 = IToXyz(State->WavPos, SbDims3[0], SbDims3[1], SbDims3[2]); // in wavelet domain
      //auto WavPosLocal3 = DecodeMorton3(State->WavPos); // DUONG: Z order does not make any difference it seems
      std::array<int, 3> WavPosGlobal3{ WavPosLocal3[0] + SbPos3[0],
                                        WavPosLocal3[1] + SbPos3[1],
                                        WavPosLocal3[2] + SbPos3[2] };
      /* if the next coefficient is out of this bound, move to the next item in Order */
      if ((SbDims3[0] <= WavPosLocal3[0]) ||
          (SbDims3[1] <= WavPosLocal3[1]) ||
          (SbDims3[2] <= WavPosLocal3[2]))
      {
        ++State->OrderPos;
        State->WavPos = 0;
        continue;
      }
      /* now that we have found a coefficient, zero out its low-order bits and push to the std::vector */
      auto Pos3 = TranslateCoord(WavPosGlobal3[0], WavPosGlobal3[1], WavPosGlobal3[2],
                                 State->Nx, State->Ny, State->Nz, State->NumLevels, State->Arrangement);
      int64_t Pos = XyzToI(State->Nx, State->Ny, State->Nz, Pos3[0], Pos3[1], Pos3[2]);
      /* add the coefficient and its position to the output */
      if (std::abs(State->DataPtr[Pos]) > threshold) {
        static int Count = 0;
        Coeffs->push_back(State->DataPtr[Pos]);
        CoeffPos3->push_back(Pos3);
        ++NumCoefficientsRead;
      }
      ++State->WavPos;
    }
    *ncoeffs = NumCoefficientsRead;
    *nbits = NumCoefficientsRead * sizeof(t) * CHAR_BIT;
  }
  else {
    std::cerr << "Stream typed not supported\n";
    exit(EXIT_FAILURE);
  }

  return !Coeffs->empty();
}


//! -------------------------------------------------------------------------------------
//! error and norm computation
//! -------------------------------------------------------------------------------------
/* Compute the squared norm of an nD signal */
template <typename t>
double SqrNorm(const t* F, int64_t Nf) {
  double Norm = 0;
  for (int64_t I = 0; I < Nf; ++I) {
    Norm += F[I] * F[I];
  }
  return Norm;
}

template <typename t>
double Norm(const t* F, int64_t Nf) {
  return std::sqrt(SqrNorm(F, Nf));
}

template<typename T>
T SquaredError(const T* F, const T* G, int64_t Size) {
  T Err = 0;
  for (size_t I = 0; I < Size; ++I) {
    T Diff = F[I] - G[I];
    Err += Diff * Diff;
  }
  return Err;
}

template<typename T>
T RMSError(const T* F, const T* G, int64_t Size) {
  return std::sqrt(SquaredError<T>(F, G, Size) / Size);
}

template<typename T>
T PSNR(const T* F, const T* G, size_t Size) {
  T Err = SquaredError<T>(F, G, Size);
  auto MinMax = std::minmax_element(F, F + Size);
  T D = 0.5 * (*(MinMax.second) - *(MinMax.first));
  Err /= Size;
  return 20.0 * log10(D) - 10.0 * log10(Err);
}

//! -------------------------------------------------------------------------------------
//! utilities for wavelets
//! -------------------------------------------------------------------------------------

/* Convenient wrapper around the above convolve */
template <typename t>
std::vector<t> Convolve(const t* F, int Nf, const t* G, int Ng) {
  int Ny = Nf + Ng - 1;
  std::vector<t> Y(Ny);
  for (int N = 0; N < Ny; ++N) {
    Y[N] = 0;
    for (int K = 0; K < Nf; ++K) {
      if (N >= K && N < Ng + K) {
        Y[N] += F[K] * G[N - K];
      }
    }
  }
  return Y;
}

/* Convenient wrapper around the above upsample_zeros */
template <typename t>
std::vector<t> UpsampleZeros(const t* F, int Nf) {
  int Ng = Nf * 2 - 1;
  std::vector<t> G(Ng);
  for (int I = 0; I < Ng; I += 2) {
    G[I] = F[I / 2];
    if (I + 1 < Ng) {
      G[I + 1] = 0;
    }
  }
  return G;
}

//! -------------------------------------------------------------------------------------
//! typedefs for wavelet transforms
//! -------------------------------------------------------------------------------------

using uint = unsigned int;
using byte = uint8_t;
using int8 = int8_t;
using i8 = int8;
using int16 = int16_t;
using i16 = int16;
using int32 = int32_t;
using i32 = int32;
using int64 = int64_t;
using i64 = int64;
using uint8 = uint8_t;
using u8 = uint8;
using uint16 = uint16_t;
using u16 = uint16;
using uint32 = uint32_t;
using u32 = uint32;
using uint64 = uint64_t;
using u64 = uint64;
using float32 = float;
using f32 = float32;
using float64 = double;
using f64 = float64;
using str = char*;
using cstr = const char*;

//! -------------------------------------------------------------------------------------
//! 3d vector support for wavelets
//! -------------------------------------------------------------------------------------

/* Vector in 3D, supports .X, .XY, .UV, .RGB and [] */
template <typename t>
struct v3 {
  t X, Y, Z;
  static v3 Zero();
  static v3 One();
  v3();
  explicit v3(t V);
  v3(t X, t Y, t Z);
  template <typename u> v3(v3<u> Other);
  template <typename u> v3& operator=(v3<u> other);
};
using v3i = v3<i32>;
using v3u = v3<u32>;
using v3l = v3<i64>;
using v3ul = v3<u64>;
using v3f = v3<f32>;
using v3d = v3<f64>;

/* v3 stuffs */
template <typename t> v3<t> v3<t>::Zero() { static v3<t> Z(0); return Z; }
template <typename t> v3<t> v3<t>::One() { static v3<t> O(1); return O; }
template <typename t> v3<t>::v3() = default;
template <typename t> v3<t>::v3(t V) : X(V), Y(V), Z(V) {}
template <typename t> v3<t>::v3(t X, t Y, t Z) : X(X), Y(Y), Z(Z) {}
template <typename t> template <typename u>
v3<t>::v3(v3<u> Other) : X(Other.X), Y(Other.Y), Z(Other.Z) {}
template <typename t> template <typename u>
v3<t>& v3<t>::operator=(v3<u> other) { X = other.X; Y = other.Y; Z = other.Z; return *this; }

#define mg_IdxX(x, y, z, N) i64(z) * N.X * N.Y + i64(y) * N.X + (x)
#define mg_IdxY(y, x, z, N) i64(z) * N.X * N.Y + i64(y) * N.X + (x)
#define mg_IdxZ(z, x, y, N) i64(z) * N.X * N.Y + i64(y) * N.X + (x)

template <typename t, typename u>
t Prod(v3<u> Vec) {
  return t(Vec.X) * t(Vec.Y) * t(Vec.Z);
}

template <typename t>
v3<t> operator+(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>{ Lhs.X + Rhs.X, Lhs.Y + Rhs.Y, Lhs.Z + Rhs.Z };
}

template <typename t>
v3<t> operator+(v3<t> Lhs, t Val) {
  return v3<t>{ Lhs.X + Val, Lhs.Y + Val, Lhs.Z + Val };
}

template <typename t>
v3<t> operator-(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>{ Lhs.X - Rhs.X, Lhs.Y - Rhs.Y, Lhs.Z - Rhs.Z };
}

template <typename t>
v3<t> operator-(v3<t> Lhs, t Val) {
  return v3<t>{ Lhs.X - Val, Lhs.Y - Val, Lhs.Z - Val };
}

template <typename t>
v3<t> operator*(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>{ Lhs.X * Rhs.X, Lhs.Y * Rhs.Y, Lhs.Z * Rhs.Z };
}

template <typename t>
v3<t> operator*(v3<t> Lhs, t Val) {
  return v3<t>{ Lhs.X * Val, Lhs.Y * Val, Lhs.Z * Val };
}

template <typename t>
v3<t> operator/(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>{ Lhs.X / Rhs.X, Lhs.Y / Rhs.Y, Lhs.Z / Rhs.Z };
}

template <typename t>
v3<t> operator/(v3<t> Lhs, t Val) {
  return v3<t>{ Lhs.X / Val, Lhs.Y / Val, Lhs.Z / Val };
}

template <typename t>
bool operator==(v3<t> Lhs, v3<t> Rhs) {
  return Lhs.X == Rhs.X && Lhs.Y == Rhs.Y && Lhs.Z == Rhs.Z;
}

template <typename t>
bool operator<=(v3<t> Lhs, v3<t> Rhs) {
  return Lhs.X <= Rhs.X && Lhs.Y <= Rhs.Y && Lhs.Z <= Rhs.Z;
}

template <typename t>
v3<t> Min(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>(Min(Lhs.X, Rhs.X), Min(Lhs.Y, Rhs.Y), Min(Lhs.Z, Rhs.Z));
}

template <typename t>
v3<t> Max(v3<t> Lhs, v3<t> Rhs) {
  return v3<t>(Max(Lhs.X, Rhs.X), Max(Lhs.Y, Rhs.Y), Max(Lhs.Z, Rhs.Z));
}

//! -------------------------------------------------------------------------------------
//! forward and inverse transforms
//! -------------------------------------------------------------------------------------

/* Forward x lifting */
#define mg_ForwardLiftCdf53(z, y, x)\
template <typename t>\
void ForwardLiftCdf53##x(t* F, v3i N, v3i L) {\
  v3i P(1 << L.X, 1 << L.Y, 1 << L.Z);\
  v3i M = (N + P - 1) / P;\
  if (M.x <= 1)\
    return;\
  for (int z = 0; z < M.z; ++z   ) {\
  for (int y = 0; y < M.y; ++y   ) {\
  for (int x = 1; x < M.x; x += 2) {\
    int XLeft = x - 1;\
    int XRight = x < M.x - 1 ? x + 1 : x - 1;\
    t & Val = F[mg_Idx##x(x, y, z, N)];\
    Val -= F[mg_Idx##x(XLeft, y, z, N)] / 2;\
    Val -= F[mg_Idx##x(XRight, y, z, N)] / 2;\
  }}}\
  for (int z = 0; z < M.z; ++z   ) {\
  for (int y = 0; y < M.y; ++y   ) {\
  for (int x = 1; x < M.x; x += 2) {\
    int XLeft = x - 1;\
    int XRight = x < M.x - 1 ? x + 1 : x - 1;\
    t Val = F[mg_Idx##x(x, y, z, N)];\
    F[mg_Idx##x(XLeft, y, z, N)] += Val / 4;\
    F[mg_Idx##x(XRight, y, z, N)] += Val / 4;\
  }}}\
  std::vector<t> Temp(M.x / 2);\
  int S##x = (M.x + 1) / 2;\
  for (int z = 0; z < M.z; ++z) {\
  for (int y = 0; y < M.y; ++y) {\
    for (int x = 1; x < M.x; x += 2) {\
      Temp[x / 2] = F[mg_Idx##x(x    , y, z, N)];\
      F[mg_Idx##x(x / 2, y, z, N)] = F[mg_Idx##x(x - 1, y, z, N)];\
    }\
    if (IsOdd(M.x))\
      F[mg_Idx##x(M.x / 2, y, z, N)] = F[mg_Idx##x(M.x - 1, y, z, N)];\
    for (int x = 0; x < (M.x / 2); ++x)\
      F[mg_Idx##x(S##x + x, y, z, N)] = Temp[x];\
  }}\
}\

mg_ForwardLiftCdf53(Z, Y, X) // X forward lifting
mg_ForwardLiftCdf53(Z, X, Y) // Y forward lifting
mg_ForwardLiftCdf53(Y, X, Z) // Z forward lifting
#undef mg_ForwardLiftCdf53

#define mg_InverseLiftCdf53(z, y, x)\
template <typename t>\
void InverseLiftCdf53##x(t* F, v3i N, v3i L) {\
  v3i P(1 << L.X, 1 << L.Y, 1 << L.Z);\
  v3i M = (N + P - 1) / P;\
  if (M.x <= 1)\
    return;\
  std::vector<t> Temp(M.x / 2);\
  int S##x = (M.x + 1) >> 1;\
  for (int z = 0; z < M.z; ++z) {\
  for (int y = 0; y < M.y; ++y) {\
    for (int x = 0; x < (M.x / 2); ++x)\
      Temp[x] = F[mg_Idx##x(S##x + x, y, z, N)];\
    if (IsOdd(M.x))\
      F[mg_Idx##x(M.x - 1, y, z, N)] = F[mg_Idx##x(M.x >> 1, y, z, N)];\
    for (int x = (M.x / 2) * 2 - 1; x >= 1; x -= 2) {\
      F[mg_Idx##x(x - 1, y, z, N)] = F[mg_Idx##x(x >> 1, y, z, N)];\
      F[mg_Idx##x(x    , y, z, N)] = Temp[x / 2];\
    }\
  }}\
  for (int z = 0; z < M.z; ++z   ) {\
  for (int y = 0; y < M.y; ++y   ) {\
  for (int x = 1; x < M.x; x += 2) {\
    int XLeft = x - 1;\
    int XRight = x < M.x - 1 ? x + 1 : x - 1;\
    t Val = F[mg_Idx##x(x, y, z, N)];\
    F[mg_Idx##x(XLeft, y, z, N)] -= Val / 4;\
    F[mg_Idx##x(XRight, y, z, N)] -= Val / 4;\
  }}}\
  for (int z = 0; z < M.z; ++z   ) {\
  for (int y = 0; y < M.y; ++y   ) {\
  for (int x = 1; x < M.x; x += 2) {\
    int XLeft = x - 1;\
    int XRight = x < M.x - 1 ? x + 1 : x - 1;\
    t & Val = F[mg_Idx##x(x, y, z, N)];\
    Val += F[mg_Idx##x(XLeft, y, z, N)] / 2;\
    Val += F[mg_Idx##x(XRight, y, z, N)] / 2;\
  }}}\
}\

mg_InverseLiftCdf53(Z, Y, X) // X inverse lifting
mg_InverseLiftCdf53(Z, X, Y) // Y inverse lifting
mg_InverseLiftCdf53(Y, X, Z) // Z inverse lifting
#undef mg_InverseLiftCdf53

#undef mg_IdxX
#undef mg_IdxX
#undef mg_IdxX

//! -------------------------------------------------------------------------------------
//! main interface for wavelet transforms
//! -------------------------------------------------------------------------------------

  template <typename t>
void Cdf53Forward(t* Data, int Nx, int Ny, int Nz, int NLevels) {
  for (int I = 0; I < NLevels; ++I) {
    ForwardLiftCdf53X(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
    ForwardLiftCdf53Y(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
    ForwardLiftCdf53Z(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
  }
}

template <typename t>
void Cdf53Inverse(t* Data, int Nx, int Ny, int Nz, int NLevels) {
  for (int I = NLevels - 1; I >= 0; --I) {
    InverseLiftCdf53Z(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
    InverseLiftCdf53Y(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
    InverseLiftCdf53X(Data, v3i(Nx, Ny, Nz), v3i(I, I, I));
  }
}

//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
}
#endif
