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
#ifndef AMM_EXCEPTIONS_H
#define AMM_EXCEPTIONS_H

#include <stdexcept>
#include "types/strformat.hpp"

//! ----------------------------------------------------------------------------
//! macros for light-weight error handling
//! ----------------------------------------------------------------------------

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define AMM_error_msg(fmt,...)  std::string(amm::format(fmt,##__VA_ARGS__)) + std::string("!\n")
#define AMM_error_loc           std::string(amm::format("  where(): %s", __PRETTY_FUNCTION__))

#define AMM_error_runtime(_,fmt,...)     if(_) throw std::runtime_error(AMM_error_msg(fmt,##__VA_ARGS__) + AMM_error_loc)
#define AMM_error_invalid_arg(_,fmt,...) if(_) throw std::invalid_argument(AMM_error_msg(fmt,##__VA_ARGS__) + AMM_error_loc)
#define AMM_error_logic(_,fmt,...)       if(_) throw std::logic_error(AMM_error_msg(fmt,##__VA_ARGS__) + AMM_error_loc)

#define AMM_error_idx_mx(i,I)   AMM_error_invalid_arg(i>=I, "invalid index %d (max = %d)", i, I-1)
#define AMM_error_idx_mn(i,I)   AMM_error_invalid_arg(i<I, "invalid index %d (min = %d)", i, I)

//! ----------------------------------------------------------------------------

#endif
