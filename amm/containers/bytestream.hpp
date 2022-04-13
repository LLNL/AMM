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
#ifndef AMM_BYTE_STREAM_H
#define AMM_BYTE_STREAM_H

//! ----------------------------------------------------------------------------
#include <bitset>
#include <stdlib.h>
#include <stddef.h>
#include <climits>
#include <stdexcept>
#include <iostream>
#include <type_traits>

#include "macros.hpp"
#include "types/byte_traits.hpp"
#include "utils/utils.hpp"
#include "utils/exceptions.hpp"


//#define AMM_DEBUG_BYTESTREAM

//! ----------------------------------------------------------------------------
namespace amm {

//! ----------------------------------------------------------------------------
//! a buffer stored as an void*
//! ----------------------------------------------------------------------------
class voidstream {

private:
    void *data = nullptr;

public:
    inline
    void
    realloc(const size_t &nbytes) {
        data = static_cast<void*>(std::realloc(data, nbytes));
        AMM_error_runtime(data == nullptr,
                          "Failed to reallocate buffer of size %d!\n", nbytes);
    }


    template <typename T>
    inline
    T
    get(const size_t i) const {
        static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
        return static_cast<T*>(data)[i];
    }


    template <typename T>
    inline
    T
    add(const size_t i, const T val) {
        static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
        static_cast<T*>(data)[i] += val;
        return static_cast<T*>(data)[i];
    }


    template <typename T>
    inline
    void
    set(const size_t i, const T val) {
        static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
        static_cast<T*>(data)[i] = val;
    }


    template <typename T>
    inline
    void
    insert(const size_t at, const size_t curr_end, const size_t new_end) {
        static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");
        std::move_backward((static_cast<T*>(data)) + at,
                           (static_cast<T*>(data)) + curr_end,
                           (static_cast<T*>(data)) + new_end);
    }


    template<typename T>
    inline
    void
    print(const size_t sz) const {
        static_assert(std::is_unsigned<T>::value, "T is required to be an unsigned type!");

        const size_t nwords = sz / sizeof(T);
        std::cout << "[ ";
        for(size_t i = 0; i < nwords; i++) {
            const T val = static_cast<T*>(data)[i];
            std::cout << std::bitset<CHAR_BIT*sizeof(T)>(val) << " : " << size_t(val) << " ";
        }
        std::cout << " ]\n";
    }
};


//! ----------------------------------------------------------------------------
//! a buffer stored as an unsigned char*
//! ----------------------------------------------------------------------------
class bytestream {

public:
    using wtype = unsigned char;
private:
    static constexpr uint8_t _nbits = CHAR_BIT*sizeof(wtype);
    static constexpr wtype _mask = static_cast<wtype>(-1);

private:
    wtype *data = nullptr;

#ifdef AMM_DEBUG_BYTESTREAM
    size_t _bsz = 0;
    inline void access(const size_t _, const std::string &caller) const {
        if (_bsz == 0) {  throw std::runtime_error("("+caller+") tried to access empty buffer!\n");   }
        if (_ >= _bsz) {  throw std::runtime_error("("+caller+") tried to access byte "+std::to_string(_)+" in a "+std::to_string(_bsz)+"-long buffer!\n");   }
    }
#endif

public:
    inline
    void
    resize(const size_t nbytes) {
        data = static_cast<wtype*>(std::realloc(data, nbytes));
        AMM_error_runtime(data == nullptr,
                          "Failed to reallocate buffer of size %d!\n", nbytes);
#ifdef AMM_DEBUG_BYTESTREAM
        _bsz = nbytes;
#endif
    }


#ifdef AMM_DEBUG_BYTESTREAM
    inline const size_t& size() const {
        return _bsz;
    }
#endif


    inline
    bool
    empty() const { return data == nullptr; }


    template <uint8_t N>
    inline
    typename amm::traits_nbytes<N>::unsigned_t
    get(const size_t _, const size_t offset = 0) const {
        static_assert(AMM_is_valid_nbytes(N), "Invalid number of bytes!");

#ifdef AMM_DEBUG_BYTESTREAM
        access(offset+_*N, "get");
#endif

        // copy from the buffer (left-right traversal) and l-shift the value
        const wtype* p = (data + offset + _*N);
        typename amm::traits_nbytes<N>::unsigned_t val (0);
        for(uint8_t b = 0; b < N; b++){
            val <<= _nbits;
            val |= *(p++);
        }
        return val;
    }


    template <uint8_t N>
    inline
    void
    set(const size_t _, typename amm::traits_nbytes<N>::unsigned_t val, const size_t offset = 0) {
        static_assert(AMM_is_valid_nbytes(N), "Invalid number of bytes!");

#ifdef AMM_DEBUG_BYTESTREAM
        access(offset+_*N, "set");
#endif

        // copy to the buffer (right-left traversal) and r-shift the value
        wtype* p = (data + offset + (_+1)*N);
        for(uint8_t b = 0; b < N; b++){
            *(--p) = wtype(val & _mask);
            val >>= _nbits;
        }
    }


    template <uint8_t N>
    inline
    void
    add(const size_t _, typename amm::traits_nbytes<N>::unsigned_t val, const size_t offset = 0) {
        static_assert(AMM_is_valid_nbytes(N), "Invalid number of bytes!");
        // @TODO: can be improved. perform the task in the same loop
        set<N>(_, val + get<N>(_, offset), offset);
    }


    template <uint8_t N>
    inline
    void
    move(const size_t at, const size_t curr_end, const size_t new_end, const size_t offset = 0) {
        static_assert(AMM_is_valid_nbytes(N), "Invalid number of bytes!");

#ifdef AMM_DEBUG_BYTESTREAM
        //std::cout << " move ("<<at<<", " <<curr_end<<") --> (..., "<<new_end<<") in buffer of size " << _bsz << std::endl;
        access(offset+at*N, "move");
        access(offset+curr_end*N-1, "move");
        access(offset+new_end*N-1, "move");

        if (at == curr_end || curr_end == new_end) {
            std::cerr << "move("<<at<<", " <<curr_end<<") --> (..., "<<new_end<<"): Trying to move zero bytes "
                      << "at word size " << int(N) <<" and buffer size " << _bsz << "!\n";
        }
        if (at > curr_end || curr_end > new_end) {
            throw std::runtime_error("Invalid move( ("+std::to_string(at)+", "+std::to_string(curr_end)+") --> (..., "+std::to_string(new_end)+" ) requested!\n");
        }
#endif
        std::move_backward(data + offset + N*at, data + offset + N*curr_end, data + offset + N*new_end);
    }


    template <uint8_t N>
    inline
    void
    print(const size_t nvals, const size_t offset=0) const {
        static_assert(AMM_is_valid_nbytes(N), "Invalid number of bytes!");

#ifdef AMM_DEBUG_BYTESTREAM
        access(offset+nvals*N-1, "print");
#endif

        using T = typename amm::traits_nbytes<N>::unsigned_t;
        static constexpr uint8_t nbits = CHAR_BIT*N;

        std::cout << "[ ";
        for(size_t i = 0; i < nvals; i++) {
            const T val = get<N>(i, offset);
            std::cout << std::bitset<nbits>(val) << " (" << size_t(val) << ") ";
        }
        std::cout << " ]\n";
    }
};
//! ----------------------------------------------------------------------------


} // end of namespace amm
//! ----------------------------------------------------------------------------

#endif
