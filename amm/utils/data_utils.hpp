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
#ifndef AMM_UTILS_RW_H_
#define AMM_UTILS_RW_H_

//! ----------------------------------------------------------------------------
#include <cstring>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "macros.hpp"
#include "types/dtypes.hpp"
#include "utils/utils.hpp"
#include "containers/vec.hpp"
#include "utils/timer.hpp"
#include "utils/logger.hpp"

//! ----------------------------------------------------------------------------
namespace amm {
namespace utils {


//! ----------------------------------------------------------------------------
//! read/write files
//! ----------------------------------------------------------------------------

static bool
file_exists(const std::string &filename) {

  std::fstream file (filename.c_str(), std::ios::in | std::ios::binary);
  if(!file.is_open()){
    return false;
  }
  file.close();
  return true;
}


template<typename T>
static bool
load_binary(std::vector<T> &data, const std::string &filename, size_t nvals_to_read = 0) {

    std::fstream file (filename.c_str(), std::ios::in | std::ios::binary);
    if(!file.is_open()){
        AMM_log_error << " Failed to open file (" << filename << ") for reading!\n";
        return false;
    }

    // get the size of the file (number of bytes and number of values)!
    file.seekg(0, std::ios::end);
    const size_t nbytes_in_file = file.tellg();
    file.seekg(0, std::ios::beg);

    const size_t nvals_in_file = nbytes_in_file/sizeof(T);
    AMM_error_invalid_arg(nbytes_in_file != nvals_in_file*sizeof(T),
                           "data_loader::load_binary(%s) got invalid number of values! [file_size = %d, datatype = %d]",
                           filename.c_str(), nbytes_in_file, sizeof(T));

    // how many values to read?
    if (nvals_to_read == 0) {
        nvals_to_read = nvals_in_file;
    }
    else {
        AMM_error_invalid_arg (nvals_in_file < nvals_to_read,
                               "data_loader::load_binary(%s) got invalid request to read %d values! [file_size = %d, datatype = %d]",
                               filename.c_str(), nvals_to_read, nvals_in_file, sizeof (T));
    }

    // allocate the memory
    data.resize(nvals_to_read, T(0));

    // max chunk that can be read depends on the OS
    // ideally, should figure out what the OS limit is
    // for now, just add the limit for mac
#if __APPLE__
    static constexpr size_t read_limit_bytes (2147483647);      // = 2 GB -1 = 2*1024*1024*1024 - 1
#else
    static constexpr size_t read_limit_bytes (nvals_to_read*sizeof(T));
#endif

    // need to read large files in smaller chunks!
    const size_t read_limit_vals = read_limit_bytes / sizeof(T);
    size_t values_read = 0;
    while (values_read < nvals_to_read) {

        // read everything if the limit has not been set
        const size_t chunk_vals = (read_limit_vals==0) ? nvals_to_read
                                                       : std::min(read_limit_vals, nvals_to_read-values_read);

        file.read ((char*)(data.data()+values_read), chunk_vals*sizeof(T));
        values_read += chunk_vals;
    }

    file.close();
    return true;
}


template <typename T>
static bool
write_binary(const std::vector<T> &d, const std::string &fname) {
    std::fstream file (fname.c_str(), std::ios::out | std::ios::binary);
    if(!file.is_open()) {
        AMM_log_error << "Failed to open file (" << fname << ") for writing!\n";
        return false;
    }

    file.write ((char*)d.data(), d.size()*sizeof(T));
    file.close();
    return true;
}


template <typename T>
static bool
write_binary(const std::vector<T> &d, const size_t databounds[3], const size_t writebounds[3], const std::string &fname) {

    if (d.size() != databounds[0]*databounds[1]*databounds[2]) {
        AMM_log_error << "Cannot write [" << databounds[0] <<" x "<<databounds[1]<<" x "<<databounds[2]<<"] data with " << d.size() << " values!\n";
        return false;
    }
    if (databounds[0] < writebounds[0] || databounds[1] < writebounds[1] || databounds[2] < writebounds[2]) {
        AMM_log_error << "Cannot write [" << databounds[0] <<" x "<<databounds[1]<<" x "<<databounds[2]<<"] data to [" << writebounds[0] <<" x "<<writebounds[1]<<" x "<<writebounds[2]<<"]\n";
        return false;
    }

    std::fstream file (fname.c_str(), std::ios::out | std::ios::binary);
    if(!file.is_open()) {
        AMM_log_error << "Failed to open file (" << fname << ") for writing!\n";
        return false;
    }

    AMM_log_debug << "Writing (" << fname << ") with " << d.size()
                  << " (" << databounds[0] << ", " << databounds[1] << ", " << databounds[2] << ") values to "
                  << " (" << writebounds[0] << ", " << writebounds[1] << ", " << writebounds[2] << ")...";

    amm::timer t;
    for(size_t z = 0; z < writebounds[2]; z++){
    for(size_t y = 0; y < writebounds[1]; y++){
    for(size_t x = 0; x < writebounds[0]; x++){
        const size_t idx = AMM_xyz2grid(x,y,z,databounds);
        file.write ((char*)&(d[idx]), sizeof(T));
    }}}
    file.close();
    AMM_logc_debug << " done!" << t << std::endl;
    return true;
}


//! ----------------------------------------------------------------------------
//! extrapolate data (zero padding or linear extrapolation)
//! ----------------------------------------------------------------------------
template<typename Tin, typename Tout=Tin>
static bool
copy_data(const size_t in_dims[3], const Tin *in_data,
          const size_t out_dims[3], std::vector<Tout> &out_data) {

    amm::timer t;
    AMM_log_info << "Copying [" << in_dims[0] << " x " << in_dims[1] << " x " << in_dims[2] << " : " << typeid(Tin).name() << "] data "
                 << "into ["<< out_dims[0] << " x " << out_dims[1] << " x " << out_dims[2] << " : " << typeid(Tout).name() << "]...";
    fflush(stdout);

    // initialize the function with zeros
    const size_t nPoints = out_dims[0]*out_dims[1]*out_dims[2];
    out_data.resize(nPoints, Tout(0));

    // this is the part of the domain where we will loop
    const size_t dims[] { std::min(in_dims[0], out_dims[0]),
                          std::min(in_dims[1], out_dims[1]),
                          std::min(in_dims[2], out_dims[2]) };

    // copy the given data over
    for (size_t z = 0; z < dims[2]; ++z) {
    for (size_t y = 0; y < dims[1]; ++y) {
    for (size_t x = 0; x < dims[0]; ++x) {
        out_data[AMM_xyz2grid(x, y, z, out_dims)] = in_data[AMM_xyz2grid(x, y, z, in_dims)];
    }}}
    t.stop();
    AMM_logc_info << " done!" << t  << std::endl;
    return true;
}


template<typename T>
static bool extrapolate_linearly(const size_t in_dims[3], const size_t out_dims[3], std::vector<T> &out_data) {

    // check if the output grid is larger than the input grid
    bool extrapolation_needed = false;
    for(size_t d = 0; d < 3; d++) {
        extrapolation_needed |= (out_dims[d] > in_dims[d]);
    }

    if (!extrapolation_needed)
        return false;

    // ------------------------------------------------------------------------------
    AMM_log_info << "Linearly extrapolating [" << in_dims[0] << " x " << in_dims[1] << " x " << in_dims[2] << "] data into "
                 << "into ["<< out_dims[0] << " x " << out_dims[1] << " x " << out_dims[2] << "]...";
    fflush(stdout);
    amm::timer t;

    // extrapolate along x
    if (in_dims[0] >= 2) {
      for (size_t z = 0; z < in_dims[2]; ++z) {
      for (size_t y = 0; y < in_dims[1]; ++y) {
          auto a = out_data[AMM_xyz2grid(in_dims[0]-2, y, z, out_dims)];
          auto b = out_data[AMM_xyz2grid(in_dims[0]-1, y, z, out_dims)];
          for (size_t x = in_dims[0]; x < out_dims[0]; ++x)
              out_data[AMM_xyz2grid(x, y, z, out_dims)] = b + (b-a) * (x-in_dims[0]+1);
      }}
    }

    // extrapolate along y
    if (in_dims[1] >= 2) {
        for (size_t z = 0; z < in_dims[2]; ++z) {
        for (size_t x = 0; x < out_dims[0]; ++x) {
            auto a = out_data[AMM_xyz2grid(x, in_dims[1]-2, z, out_dims)];
            auto b = out_data[AMM_xyz2grid(x, in_dims[1]-1, z, out_dims)];
            for (size_t y = in_dims[1]; y < out_dims[1]; ++y)
                out_data[AMM_xyz2grid(x, y, z, out_dims)] = b + (b-a) * (y-in_dims[1]+1);
        }}
    }

    // extrapolate along z
    if (in_dims[2] >= 2) {
        for (size_t y = 0; y < out_dims[1]; ++y) {
        for (size_t x = 0; x < out_dims[0]; ++x) {
            auto a = out_data[AMM_xyz2grid(x, y, in_dims[2]-2, out_dims)];
            auto b = out_data[AMM_xyz2grid(x, y, in_dims[2]-1, out_dims)];
            for (size_t z = in_dims[2]; z < out_dims[2]; ++z)
                out_data[AMM_xyz2grid(x, y, z, out_dims)] = b + (b-a) * (z-in_dims[2]+1);
        }}
    }
    t.stop();
    AMM_logc_info << " done!" << t  << std::endl;
    return true;
}

//! ----------------------------------------------------------------------------
}}   // end of namespace
#endif
