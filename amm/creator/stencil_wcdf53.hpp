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
#ifndef AMM_STENCIL_WCDF53_H
#define AMM_STENCIL_WCDF53_H

//! ----------------------------------------------------------------------------
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "macros.hpp"
#include "containers/vec.hpp"
#include "types/dtypes.hpp"
#include "stencil.hpp"
#include "wavelets/enums.hpp"

//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! static struct to initialize cdf53 stencils
//! ----------------------------------------------------------------------------
template<TypeDim Dim>
struct stencil_cdf53 {
    using St = amm::stencil<Dim>;
    static inline std::vector<St> init(const EnumWavelet);
};

template<>
inline std::vector<stencil_cdf53<1>::St>
stencil_cdf53<1>::init(const EnumWavelet m_wtype) {

    using St1 = stencil_cdf53<1>::St;

    std::vector<St1> m_stencils (2);
    m_stencils[as_utype(EnumWCoefficient::S)] = St1 (EnumWCoefficient::S,
                                                     {0.0, 0.5, 1.0, 0.5, 0.0});

    if (m_wtype == EnumWavelet::Interpolating) {
        m_stencils[as_utype(EnumWCoefficient::W)] = St1 (EnumWCoefficient::W,
                                                         {0.0, 1.0, 0.0});
    }
    else {
        m_stencils[as_utype(EnumWCoefficient::W)] = St1 (EnumWCoefficient::W,
                                                         {0.0, -0.125, -0.25, 0.75, -0.25, -0.125, 0.0});
    }
    return m_stencils;
}

template<>
inline std::vector<stencil_cdf53<2>::St>
stencil_cdf53<2>::init(const EnumWavelet m_wtype) {

    using St1 = stencil_cdf53<1>::St;
    using St2 = stencil_cdf53<2>::St;

    // create a 1d stencil
    const std::vector<St1> st1 = stencil_cdf53<1>::init(m_wtype);
    auto s = st1[as_utype(EnumWCoefficient::S)];
    auto w = st1[as_utype(EnumWCoefficient::W)];

    // create a 2d stencil as the tensor product
    std::vector<St2> m_stencils (4);
    m_stencils[as_utype(EnumWCoefficient::SS)] = St2 (s, s);
    m_stencils[as_utype(EnumWCoefficient::SW)] = St2 (s, w);
    m_stencils[as_utype(EnumWCoefficient::WS)] = St2 (w, s);
    m_stencils[as_utype(EnumWCoefficient::WW)] = St2 (w, w);
    return m_stencils;
}

template<>
inline std::vector<stencil_cdf53<3>::St>
stencil_cdf53<3>::init(const EnumWavelet m_wtype) {

    using St1 = stencil_cdf53<1>::St;
    using St2 = stencil_cdf53<2>::St;
    using St3 = stencil_cdf53<3>::St;

    // create 1d and 2d stencils
    const std::vector<St1> st1 = stencil_cdf53<1>::init(m_wtype);
    const std::vector<St2> st2 = stencil_cdf53<2>::init(m_wtype);
    auto s = st1[as_utype(EnumWCoefficient::S)];
    auto w = st1[as_utype(EnumWCoefficient::W)];
    auto ss = st2[as_utype(EnumWCoefficient::SS)];
    auto sw = st2[as_utype(EnumWCoefficient::SW)];
    auto ws = st2[as_utype(EnumWCoefficient::WS)];
    auto ww = st2[as_utype(EnumWCoefficient::WW)];

    // create a 3d stencil as the tensor product
    std::vector<St3> m_stencils (8);
    m_stencils[as_utype(EnumWCoefficient::SSS)] = St3 (s, ss);
    m_stencils[as_utype(EnumWCoefficient::SSW)] = St3 (s, sw);
    m_stencils[as_utype(EnumWCoefficient::SWS)] = St3 (s, ws);
    m_stencils[as_utype(EnumWCoefficient::SWW)] = St3 (s, ww);
    m_stencils[as_utype(EnumWCoefficient::WSS)] = St3 (w, ss);
    m_stencils[as_utype(EnumWCoefficient::WSW)] = St3 (w, sw);
    m_stencils[as_utype(EnumWCoefficient::WWS)] = St3 (w, ws);
    m_stencils[as_utype(EnumWCoefficient::WWW)] = St3 (w, ww);
    return m_stencils;
}

}   // end of namespace
/// --------------------------------------------------------------------------------
/// --------------------------------------------------------------------------------
#endif
