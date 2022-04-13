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
#ifndef AMM_MACROS_H
#define AMM_MACROS_H

//! ----------------------------------------------------------------------------
//! Macros to control AMM's functionalities
//! ----------------------------------------------------------------------------

#define AMM_USE_VTK
//#define AMM_ENABLE_PRECISION
#define AMM_MAX_DATA_DEPTH 12


//! ----------------------------------------------------------------------------
//! We recommend not changing these macros
//! ----------------------------------------------------------------------------

// stage the coefficients before updating the mesh
// used only for precision streams where it is useful
// to collect all bits for a coefficient before processing
#define AMM_STAGE_WCOEFFS

// stage the nodes within AMM before creating/splitting
// useful to reduce redundant processing
#define AMM_STAGE_NODES

// stage the vertices within AMM before creating/splitting
// useful to reduce redundant processing
    // currently, this must be set!
#define AMM_STAGE_VERTS


// preprocess all wavelet coefficients to ensure we dont
// process those that do not intersect with the actual domain
#define AWR_FILTER_EXTERNAL_COEFFICIENTS


// some specifications for precision handling
#define AMM_MAX_NBYTES 8
#define AMM_MAX_NBITS 64
#define AMM_MAX_PRECISION 8
#define AMM_BLOCK_NVERTS 64


// some macros used in wavelet computation
#define RESTRICT
#define THREAD_LOCAL
#define PARALLEL_LOOP
#define OUT



//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
//! the following are useful only for developers for testing/debugging
//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------

// testing and validation of representation
//#define AWR_VALIDATE_PER_COEFF
//#define AWR_VALIDATE_PER_UPDATE
#ifdef AWR_VALIDATE_PER_COEFF
#ifndef AWR_VALIDATE_PER_UPDATE
#define AWR_VALIDATE_PER_UPDATE
#endif
#endif

//! ----------------------------------------------------------------------------
//! a disabled and partially implemented functonality
//! ----------------------------------------------------------------------------
// can store vertex (index) cache as a dense vector of bools
// takes more memory but may be faster than querying the set
//#define AMM_USE_DENSE_CACHE


//! ----------------------------------------------------------------------------
//! some debugging functionality
//! ----------------------------------------------------------------------------
//#define AMM_DEBUG_LOGIC
#define AMM_DEBUG_VMANAGER_INVALIDS
#define AMM_DEBUG_VMANAGER_ZEROS


//#define AWR_DEBUG_TEST_STREAM
//#define AWR_DEBUG_TEST_ITERATORS
//#define AWR_DEBUG_TEST_STENCILS
//#define VTKCREATOR_TEST_ARBIT_SPLIT


//#define AMM_ENCODE_CFLAGS
//#define AMM_MAPPED_CONFIGS
//#define AMM_USE_NAN_FOR_MISSING_VERTEX
//#define AMM_STORE_IMPROPER_NODES

//#define AMM_DEBUG_TRACK_CONFIGS
#ifdef AMM_DEBUG_TRACK_CONFIGS
#undef AMM_ENCODE_CFLAGS
#endif

// old code for writing some profiles
//#define AWR_WRITE_PROFILING


//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
#endif
