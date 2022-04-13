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
#ifndef AMM_LOGGER_H
#define AMM_LOGGER_H

//! ----------------------------------------------------------------------------
#include <stdarg.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>

#include "types/strformat.hpp"

//! ----------------------------------------------------------------------------
//! an extremely lightweight logging utility that uses macros to control output
//! ----------------------------------------------------------------------------

namespace amm {

enum struct LogLevel: std::uint8_t {
    None  = 0,
    Error = 1,
    Warn  = 2,
    Info  = 3,
    Debug = 4
};

static LogLevel gloglevel = LogLevel::Info;
inline static void set_loglevel(const LogLevel _) {
    amm::gloglevel = _;
}
}   // end of namespace


//! ----------------------------------------------------------------------------
//! macro to define log format

#define AMM_log_format(_)       ""
//#define AMM_log_format(_)       amm::format("[%s] ", _)
//#define AMM_log_format(_)       amm::format("<%s, %s> [%s] ", __DATE__, __TIME__, _)
//#define AMM_log_format(_)       amm::format("(%s:%d) [%s] ", __FILE__, __LINE__, _)
//#define AMM_log_format(_)       amm::format("<%s, %s> (%s:%d) [%s] ", __DATE__, __TIME__, __FILE__, __LINE__, _)

//! ----------------------------------------------------------------------------
//! macros to define the logging streams

#define AMM_log_debug if (amm::gloglevel < amm::LogLevel::Debug) {} else std::cout << AMM_log_format("Debug")
#define AMM_log_info  if (amm::gloglevel < amm::LogLevel::Info)  {} else std::cout << AMM_log_format("Info")
#define AMM_log_warn  if (amm::gloglevel < amm::LogLevel::Warn)  {} else std::cerr << AMM_log_format("Warning")
#define AMM_log_error if (amm::gloglevel < amm::LogLevel::Error) {} else std::cerr << AMM_log_format("Error")

// if we want to continue a print on the same line, do not print the log type
#define AMM_logc_debug if (amm::gloglevel < amm::LogLevel::Debug) {} else std::cout
#define AMM_logc_info  if (amm::gloglevel < amm::LogLevel::Info)  {} else std::cout
#define AMM_logc_warn  if (amm::gloglevel < amm::LogLevel::Warn)  {} else std::cerr
#define AMM_logc_error if (amm::gloglevel < amm::LogLevel::Error) {} else std::cerr

//! ----------------------------------------------------------------------------

#endif
