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
#ifndef AMM_TIMER_H
#define AMM_TIMER_H

//! ----------------------------------------------------------------------------
#include <cmath>
#include <chrono>
#include <ostream>
#include <iomanip>

namespace amm {
//! ----------------------------------------------------------------------------
//! timer utility
//! ----------------------------------------------------------------------------

class timer {

public:
    using clock  = std::chrono::high_resolution_clock;
    using tstamp = std::chrono::time_point<clock>;

private:
    static inline tstamp now() {            return clock::now();    }
public:
    static inline size_t now_ms() {
        auto _ = std::chrono::time_point_cast<std::chrono::milliseconds>(timer::now());
        return static_cast<size_t>(_.time_since_epoch().count());
    }

private:
    tstamp m_start, m_end;
    bool m_active;

public:
    timer() {               this->start();                                  }
    inline void start() {   m_start = timer::now();  m_active = true;       }
    inline void stop() {    m_end = timer::now();    m_active = false;      }

    inline size_t elapsed() const {
        auto endTime = m_active ? timer::now() : m_end;
        auto _ = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-m_start);
        return static_cast<size_t>(_.count());
    }

    friend inline std::ostream& operator<<(std::ostream &os, const timer &t);
};

inline std::ostream& operator<<(std::ostream &os, const timer &tmr) {

    auto t = static_cast<double>(tmr.elapsed());    // t is in milliseconds

    if (t < 1)      return os << " [took " << std::setprecision(3) << 1000.0*t << " usec]";
    if (t < 1000)   return os << " [took " << std::setprecision(3) << t        << " msec]";

    t *= 0.001;                                     // now, t is in seconds
    if (t < 60)     return os << " [took " << std::setprecision(3) << t        << " sec]";

    auto m = std::floor(t/60.0);                    // minutes
    t -= (m*60.0);                                  // remaining seconds
    return os << " [took " << m << " mins " << std::setprecision(3) << t       << " sec]";
}
//! ----------------------------------------------------------------------------

}   // end of namespace
#endif
