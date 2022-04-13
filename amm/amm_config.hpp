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
#ifndef AMM_CONFIG_H
#define AMM_CONFIG_H

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "types/dtypes.hpp"
#include "types/enums.hpp"
#include "types/strformat.hpp"
#include "wavelets/enums.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"


namespace amm {

//! -----------------------------------------------------------------------------
//! Configuration Manager for AMM
//! -----------------------------------------------------------------------------
struct amm_config {

private:
    // Mandatory input parameters
    std::unordered_map<std::string, bool> amm_hasReq;

    template <typename T>
    bool validate(const std::string &arg, const T& val, bool allow_zero) const {

        if (val > 0)                return true;
        if (allow_zero && val==0)   return true;
        AMM_log_error << "Invalid value [" << val << "] for parameter [" << arg << "]\n";
        usage();
        return false;
    }

    template <typename T>
    bool validate(const std::string &arg, const T& val, const std::vector<T> &valid_vals) const {

        for(auto iter = valid_vals.begin(); iter != valid_vals.end(); iter++) {
            if (*iter == val)   return true;
        }
        AMM_log_error << "Invalid value [" << val << "] for parameter [" << arg << "]\n";
        usage();
        return false;
    }

    bool validate(const std::string &arg, const std::string& val, const std::vector<std::string> &valid_vals) const {

        for(auto iter = valid_vals.begin(); iter != valid_vals.end(); iter++) {
            if (AMM_is_same(val, *iter))   return true;
        }

        AMM_log_error << "Invalid value [" << val << "] for parameter [" << arg << "]\n";
        usage();
        return false;
    }

public:
    std::string amm_command = "unknown";

    // input data (no default.. must be specified)
    std::string amm_inputFilename = "unknown";
    std::string amm_inputType = "unknown";
    std::string amm_inputPrecision = "unknown";
    std::size_t amm_dataDims[3];

    // info about stream (no default.. must be specified)
    std::uint8_t amm_streamType = 0;

    double amm_chunkSize = -1.0;
    std::string amm_chunkUnit = "unknown";

    double amm_streamEnd = -1.0;
    std::string amm_streamEndCriterion = "unknown";

    // amm parameters (default values, can be specified)
    bool amm_enableImproper = false;
    bool amm_enableRectangular = true;

    std::string amm_outPath = "";    // path for the output vtk file
    std::string amm_wavOut = "";     // file to write the wavelet coefficients
    bool amm_validateOutput = true; // whether to validate the output mesh
    bool amm_writeLowres = false;    // whether to write low res function

    std::uint8_t amm_waveletsDepth = 1;
    bool amm_waveletsNormalized = false;
    EnumWavelet amm_waveletsType =  EnumWavelet::Approximating;

    EnumExtrapolation amm_extrapolationType =  EnumExtrapolation::LinearLifting;

    //! -------------------------------------------------------------------------
    amm_config() {
        amm_hasReq = {{"--input", 0},
                      {"--dims", 0},
                      {"--stream", 0}};
    }

    //! -------------------------------------------------------------------------
    void print() const {
        AMM_log_info << " AMMConfig\n"
            << "\t fname = [" << amm_inputFilename << "],"
            << " type = [" << amm_inputType << "],"
            << " precision = [" << amm_inputPrecision << "]\n"
            << "\t dims = [" << amm_dataDims[0] << " x " << amm_dataDims[1] << " x " << amm_dataDims[2] << "]\n"
            << "\t stream = [" << int(amm_streamType) << "], "
            << " chunk = [" << amm_chunkSize << " " << amm_chunkUnit << "], "
            << " end = [" << amm_streamEnd << " " << amm_streamEndCriterion << "]";
    }

    //! -------------------------------------------------------------------------
    void usage() const {

        AMM_log_error << " Usage: " << amm_command << " <args>\n"
            << "\t --input  [file_name] [func/wcoeffs/amm] [dtype]: input filename, type of input, precision of input\n"
            << "                               func:                                input function read from raw binary file of specific datatype (u8, f32, f64)\n"
            << "                               wcoeffs:                             wavelet coefficients read from raw binary file of specific datatype (f32, f64)\n"
            << "                               amm:                                 AMM read from a vtk unstructued mesh (precision not specified)\n"
            << "\t --dims   [X] [Y] [Z=1]:                              dimensions of data [default: Z = 1])\n"
            << "\n"
            << "\t --stream [stream_type]:                              type of stream to create\n"
            << "                               1:                                   by row major\n"
            << "                               2:                                   by subband row major\n"
            << "                               3:                                   by coeff wavelet norm\n"
            << "                               4:                                   by wavelet norm\n"
            << "                               5:                                   by level\n"
            << "                               6:                                   by bitplane\n"
            << "                               7:                                   by magnitude\n"
            << "\t --chunk  [chunk_size=0] [count/kb]:                  size of each chunks, either as number of coefficients or amount of data (in kb) [default: chunk_size = full stream]\n"
            << "\t --end    [end_val=0] [count/kb/val]:                 end of the stream, either as number of coefficients, amount of data (in kb), or threshold value of the coefficients [default: end_val = full stream]\n"
            << "\n"
            << "\t --rect   [0/1 = 1]:                                  enable rectangular nodes in AMM [default: true]\n"
            << "\t --improp [0/1 = 0]:                                  enable improper node in AMM [default: false]\n"
            << "\t --wvlts  [approx/interp = approx]:                   use approximating or interpolating wavelet basis [default: approximating]\n"
            << "\t --extrap [zero/linear/linlift = linlift]:            extrapolate data using zero-padding, linear extrapolation, or linear-lifting method [default: linear-lifting]\n"
            << "\t --wnorm  [0/1 = 0]:                                  normalize wavelet basis [default: false]\n"
            << "\t --wdepth [depth = 1]:                                compute wavelet transform up until the specified depth [default: depth=1]\n"
            << "\n"
            << "\t --novalidate                                         do not validate the output\n"
            << "\t --outpath [path]                                     path for the output vtk file [default: no output]\n"
            << "\t --lowres                                             also write the lowres function (same path as above)\n"
            << "\t --wavout [file_name]                                 output the wavelet coefficients to a file\n";
        exit(EXIT_FAILURE);
    }

    //! -------------------------------------------------------------------------
    void parse(int argc, char **argv) {

        amm_command = argv[0];
        for(int i = 1; i < argc; i++) {

            const std::string _(argv[i]);

            // ignore any stray argument
            if (!AMM_starts_with(_, "--")) {   AMM_log_info << " ignoring [" << _ << "]\n";     continue;   }

            // input file
            if (AMM_is_same(_, "--input")) {
                if (argc <= i+2) {AMM_log_warn << " ignoring [--input] (missing params)\n"; continue;}

                amm_inputFilename = argv[++i];
                amm_inputType = argv[++i];
                validate(_, amm_inputType, {"func", "wcoeffs", "amm"});
                if (AMM_is_same(amm_inputType, "func") || AMM_is_same(amm_inputType, "wcoeffs")) {
                    amm_inputPrecision = argv[++i];
                    if (AMM_is_same(amm_inputType, "func")) {
                        validate(_, amm_inputPrecision, {"u8", "f32", "f64"});
                    }
                    else {
                        validate(_, amm_inputPrecision, {"f32", "f64"});
                    }
                }
                amm_hasReq["--input"] = true;
            }

            // dimensions of data
            if (AMM_is_same(_, "--dims")) {
                if (argc <= i+2) {AMM_log_warn << " ignoring [--dims] (missing params)\n"; continue;}
                amm_dataDims[0] = static_cast<std::size_t>(atoi(argv[++i]));  validate(_, amm_dataDims[0], false);
                amm_dataDims[1] = static_cast<std::size_t>(atoi(argv[++i]));  validate(_, amm_dataDims[1], false);
                if (i < argc-1 && !AMM_starts_with(std::string(argv[i+1]), "--")) {
                    amm_dataDims[2] = static_cast<std::size_t>(atoi(argv[++i]));  validate(_, amm_dataDims[2], false);   }
                else {
                    amm_dataDims[2] = 1;                                                                            }
                amm_hasReq["--dims"] = true;
                continue;
            }

            // stream settings
            if (AMM_is_same(_, "--stream")) {   if (argc <= i+1) {AMM_log_warn << " ignoring [--stream] (missing params)\n";        continue;}
                                            amm_streamType = static_cast<std::uint8_t>(atoi(argv[++i]));
                                            if (amm_streamType < 1 || amm_streamType > 7){
                                                AMM_log_error << " Invalid value [" << int(amm_streamType) << "] for parameter [--stream]\n";
                                                usage();
                                            }
                                            amm_hasReq["--stream"] = true;                                                      continue;}

            if (AMM_is_same(_, "--chunk")) {    if (argc <= i+2) {AMM_log_warn << " ignoring [--chunk] (missing params)\n";         continue;}
                                            amm_chunkSize = atoi(argv[++i]);  validate(_, amm_chunkSize, true);
                                            amm_chunkUnit = argv[++i];        validate(_, amm_chunkUnit, {"count", "kb"});      continue;}

            if (AMM_is_same(_, "--end")) {      if (argc <= i+2) { AMM_log_warn << " ignoring [--end] (missing params)\n"; continue; }
                                                amm_streamEnd = atof(argv[++i]);  validate(_, amm_streamEnd, true);
                                                amm_streamEndCriterion = argv[++i];    validate(_, amm_streamEndCriterion, {"val", "count", "kb"}); continue;}

            // amm settings
            if (AMM_is_same(_, "--wvlts")) {    if (argc <= i+1) {AMM_log_warn << " ignoring [--wvlts] (missing params)\n";         continue;}
                                            std::string p = argv[++i];        validate(_, p, {"approx","interp"});
                amm_waveletsType = p.compare("approx")==0 ?  EnumWavelet::Approximating :  EnumWavelet::Interpolating;          continue;}
            if (AMM_is_same(_, "--extrap")) {   if (argc <= i+1) {AMM_log_warn << " ignoring [--extrap] (missing params)\n";        continue;}
                                            std::string p = argv[++i];        validate(_, p, {"zero","linear", "linlift"});
                amm_extrapolationType = p.compare("zero")==0 ? EnumExtrapolation::Zero :
                                        p.compare("linear")==0 ? EnumExtrapolation::Linear : EnumExtrapolation::LinearLifting;  continue;}

            if (AMM_is_same(_, "--rect")) {     if (argc <= i+1) {AMM_log_warn << " ignoring [--rect] (missing params)\n";          continue;}
                                            int p = atoi(argv[++i]);  validate(_, p, {0,1}); amm_enableRectangular = p;         continue;}
            if (AMM_is_same(_, "--improp")) {   if (argc <= i+1) {AMM_log_warn << " ignoring [--improp] (missing params)\n";        continue;}
                                            int p = atoi(argv[++i]);  validate(_, p, {0,1}); amm_enableImproper = p;            continue;}

            if (AMM_is_same(_, "--wnorm")) {    if (argc <= i+1) {AMM_log_warn << " ignoring [--wnorm] (missing params)\n";         continue;}
                                            int p = atoi(argv[++i]);  validate(_, p, {0,1}); amm_waveletsNormalized = p;        continue;}
            if (AMM_is_same(_, "--wdepth")) {   if (argc <= i+1) {AMM_log_warn << " ignoring [--wdepth] (missing params)\n";        continue;}
                                            int p = atoi(argv[++i]);  validate(_, p, false); amm_waveletsDepth = static_cast<std::uint8_t>(p); continue;}

            if (AMM_is_same(_, "--novalidate")) { amm_validateOutput = false; }
            if (AMM_is_same(_, "--lowres")) {     amm_writeLowres = true; }
            if (AMM_is_same(_, "--outpath")) {
                if (argc <= i + 1) { AMM_log_warn << " ignoring [--outpath] (missing params)\n";         continue; }
                amm_outPath = argv[++i];
            }
            if (AMM_is_same(_, "--wavout")) {
                if (argc <= i + 1) { AMM_log_warn << " ignoring [--wavout] (missing params)\n";         continue; }
                amm_wavOut = argv[++i];
            }
        }

        // catch missing required parameters

        if (!AMM_is_same(amm_inputType, "amm")) {
            auto iter = std::find_if(amm_hasReq.begin(), amm_hasReq.end(), [](const std::pair<std::string,bool> &v){return !v.second;});
            if (iter != amm_hasReq.end()) {
                AMM_log_error << " Missing required parameter " << (*iter).first << "\n";
                usage();
            }
        }
    }

    //! -------------------------------------------------------------------------
};
//! -----------------------------------------------------------------------------
}   // end of namespace
#endif
