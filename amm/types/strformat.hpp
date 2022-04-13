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
#ifndef AMM_STRFORMAT_H
#define AMM_STRFORMAT_H

//! ----------------------------------------------------------------------------
#include <cstdio>
#include <cstdarg>
#include <locale>
#include <sstream>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

namespace amm {

static inline
void
ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}


static inline
void
rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}


static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


inline static
std::vector<std::string>
tokenize(const std::string &_, const char delim = ' ') {

    std::vector<std::string> tokens;
    std::stringstream ss (_);
    std::string tok;

    while (std::getline (ss, tok, delim)) {
        tokens.push_back (tok);
    }

    return tokens;
}


inline static
const char*
format(const char *fmt, ...) {
    
    constexpr size_t bsize = 256;
    
    static char buffer[bsize];
    static va_list args;
    
    va_start(args, fmt);
    std::vsnprintf(buffer, bsize, fmt, args);
    va_end(args);
    
    return buffer;
}


struct thousand_separator : std::numpunct<char> {
protected :
    virtual char do_thousands_sep()   const { return ','; }  // separate with spaces
    virtual std::string do_grouping() const { return "\3"; } // groups of 1 digit
};


inline static void set_locale() {

    auto l = std::locale(std::cout.getloc(), new thousand_separator);
    std::cout.imbue(l);
    std::cerr.imbue(l);
    std::clog.imbue(l);
    //std::cout.precision(10);
    //std::cout.setf (std::cout.fixed , std::cout.floatfield);
    //std::cout.setf (std: .fixed , std::cout.floatfield);
    //std::cout << std::defaultfloat;
}

}   // end of namespace
#endif
