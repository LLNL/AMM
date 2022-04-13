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
#ifndef _AMM_HANDLER_H_
#define _AMM_HANDLER_H_

//! -------------------------------------------------------------------------------------
#include <cmath>
#include <bitset>
#include <cstring>
#include <cstdint>
#include <limits>
#include <iostream>
#include <fstream>

#include "types/dtypes.hpp"
#include "containers/vec.hpp"
#include "utils/data_utils.hpp"
#include "tree/amtree.hpp"

#include "amm.hpp"
#include "amm_config.hpp"
#include "amm_profile.hpp"

#include "wavelets/wavelet_cdf53.hpp"
#include "creator/creator_wcdf53.hpp"

#include "streams/enums.hpp"
#include "streams/utils.hpp"
#include "streams/data_stream.hpp"

#ifdef AMM_USE_VTK
#include <vtkCell.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGridWriter.h>

#include "vtk/vtkAMM.hpp"
#endif


//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
//! Master class that manages the construction of the AMM representation
//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------

template <typename _Tp>
class AMMRepresentationHandler {

    using _WvTp  = wavelets::Wavelets_CDF53<_Tp>;
    using _StTp  = streams::dataStream<_Tp>;

    template<TypeDim D>
    using _AMMTp = typename amm::AMM<_Tp, D, AMM_MAX_DATA_DEPTH>;

    template<TypeDim D>
    using _CrtTp = typename amm::amm_creator_wcdf53<_Tp, D, AMM_MAX_DATA_DEPTH>;

    template<TypeDim D>
    using _VertexTypeD = typename _AMMTp<D>::TypeVertex;

    using _VertexType2 = typename _AMMTp<2>::TypeVertex;
    using _VertexType3 = typename _AMMTp<3>::TypeVertex;

    //! AMM configuration
    const amm::amm_config m_config;

    //! we maintain two types of domain sizes
    size_t m_fileDims[3] = {0, 0, 0};           //! input dimensions of the data
    size_t m_treeDims[3] = {0, 0, 0};           //! internal dimensions of the octree representation (2^L + 1, and square)
    size_t m_dataDims[3] = {0, 0, 0};           //! dimensions of the (cropped, if needed) data

    TypeScale m_dataDepth = 0;                  //! L for the given data

    //! corresponding number of points
    size_t m_nDataPoints = 0;
    size_t m_nTreePoints = 0;

    //! member variables
    _AMMTp<2> *m_octree2 = nullptr;        //! octree representation
    _AMMTp<3> *m_octree3 = nullptr;        //! octree representation

    _StTp *m_dataStream = nullptr;          //! class to compute data stream
    _WvTp *m_wcomputer = nullptr;           //! class to compute wavelet coefficients

    std::vector<_Tp> m_func;                //! function
    std::vector<_Tp> m_wcoeffs;             //! wavelet coefficients
    std::vector<_Tp> m_wcoeffs_filtered;    //! filtered wavelet coefficients
    std::vector<_Tp> m_lowres_func;         //! reconstructed function from wavelets

    EnumStream m_streamType;
    size_t m_stream_chunk_count = ~(size_t)0;    //! size of each chunk of the stream in terms of count
    size_t m_stream_chunk_bits = ~(size_t)0;    //! size of each chunk of the stream in terms of bytes

    size_t m_stream_end_coeffs_wanted = ~(size_t)0;
    size_t m_stream_end_bits_wanted = ~(size_t)0;
    double m_stream_end_threshold = 0;        //! threshold corresponding to the end of the stream (<0 = full stream)

    size_t m_stream_total_bits = 0;
    size_t m_stream_total_count = 0;
    bool   m_stream_done = false;

    EnumArrangement m_wcoeff_arrangement_type;
    //std::vector<std::tuple<size_t, size_t, uint8_t, EnumWCoefficient>> m_wcoeff_uindices;
    size_t m_stream_time = 0;

public:
    //! ---------------------------------------------------------------------------------
    //! constructor
    //! ---------------------------------------------------------------------------------
    AMMRepresentationHandler(const amm::amm_config &config) :
        m_config(config),
        m_wcomputer(nullptr), m_octree2(nullptr), m_octree3(nullptr) {

        m_fileDims[0] = 0;    m_fileDims[1] = 0;    m_fileDims[2] = 0;
        m_treeDims[0] = 0;    m_treeDims[1] = 0;    m_treeDims[2] = 0;
        m_dataDims[0] = 0;    m_dataDims[1] = 0;    m_dataDims[2] = 0;

        if (m_config.amm_inputType.compare("vtk") == 0)
            return;

        // TODO: how to read the arrangement in
        m_streamType = as_enum<EnumStream>(static_cast<std::uint8_t>(m_config.amm_streamType));
        m_wcoeff_arrangement_type = streams::get_stream_suitable_arrangement(m_streamType);
    }

    //! ---------------------------------------------------------------------------------
    //! destructor
    //! ---------------------------------------------------------------------------------
    ~AMMRepresentationHandler() {
        reset();
        delete m_wcomputer;
        delete m_dataStream;
    }

    //! ---------------------------------------------------------------------------------
    //! initialize the representation from function
    //! ---------------------------------------------------------------------------------
    template <typename TypeInputFile>
    void init_with_function(const size_t &X, const size_t &Y, const size_t &Z = 1) {

        this->init_sizes(X,Y,Z);

        // function could have a different data type
        amm::timer t;
        AMM_log_info << "Loading [" << m_fileDims[0] << " x " << m_fileDims[1] << " x " << m_fileDims[2] << "] "
                     << "function from (" << m_config.amm_inputFilename << " : " << typeid(TypeInputFile).name() << ")...";
        fflush(stdout);

        const size_t _nFilePoints = m_fileDims[0]*m_fileDims[1]*m_fileDims[2];
        std::vector<TypeInputFile> _data;
        amm::utils::load_binary<TypeInputFile>(_data, m_config.amm_inputFilename, _nFilePoints);

        t.stop();
        AMM_logc_info << " done!" << t  << std::endl;

        AMM_error_invalid_arg(_data.empty() || _data.data() == nullptr,
                              "Failed to load function!");

        AMM_error_invalid_arg(_data.size() != _nFilePoints,
                              "Invalid size for function! Given = (%lu), Expected (%lu)",
                              _data.size(), _nFilePoints);

        auto frng0 = amm::utils::frange(_data);
        AMM_log_info << "\tgiven data:   ["<<m_fileDims[0]<<" x "<<m_fileDims[1]<<" x "<<m_fileDims[2]<<"], "
                     << "range = ("<<_Tp(frng0.first) << ", " << _Tp(frng0.second)<<")\n";

        this->preprocess<TypeInputFile>(_data.data());

        auto frng = amm::utils::frange(m_func);
        AMM_log_info << "\tpreprocessed data: ["<<m_treeDims[0]<<" x "<<m_treeDims[1]<<" x "<<m_treeDims[2]<<"], range = "<<frng<<"\n";

        this->compute_wavelet_transform(m_func, m_wcoeffs, +1);

        // no need to carry the memory for the function
        // if we wont compute the quality later
        if (!m_config.amm_validateOutput) {
            m_func.clear();
        }

        this->filter_external_wcoeffs();
        if (!m_config.amm_wavOut.empty()) {
            write_wavelets(m_config.amm_wavOut);
        }
    }

    //! ---------------------------------------------------------------------------------
    //! initialize the representation from wavelet coefficients
    //! ---------------------------------------------------------------------------------
    void init_with_wcoeffs(size_t X, size_t Y, size_t Z = 1) {

        this->init_sizes(X,Y,Z);

        amm::timer t;
        AMM_log_info << "Loading [" << m_treeDims[0] << " x " << m_treeDims[1] << " x " << m_treeDims[2] << "] "
                     << "wavelets from (" << m_config.amm_inputFilename << " : " << typeid(_Tp).name() << ")...";
        fflush(stdout);

        amm::utils::load_binary<_Tp> (m_wcoeffs, m_config.amm_inputFilename, m_nTreePoints);

        t.stop();
        AMM_logc_info << " done!" << t  << std::endl;

        AMM_error_invalid_arg(m_wcoeffs.empty() || m_wcoeffs.data() == nullptr,
                              "Failed to load wavelet coefficients!");

        AMM_error_invalid_arg(m_wcoeffs.size() != m_nTreePoints,
                              "Invalid size for wavelet coeffs! Given = (%lu), Expected (%lu)",
                              m_wcoeffs.size(), m_nTreePoints);

        std::pair<_Tp,_Tp> frng = amm::utils::frange(m_wcoeffs);
        AMM_log_info << "\tgiven wavelet coefficients: "
                     << "("<<m_treeDims[0]<<","<<m_treeDims[1]<<","<<m_treeDims[2]<<"), "
                     << "range = "<<frng<<"\n";

        this->filter_external_wcoeffs();
    }

    //! ---------------------------------------------------------------------------------
    //! initialize with a precomputed amm
    //! ---------------------------------------------------------------------------------
#ifdef AMM_USE_VTK
    void init_with_amm(vtkAMM &vtk_mesh) {
        this->load_vtkamm(vtk_mesh);
    }
#endif

    //! ---------------------------------------------------------------------------------
    //! save amm
    //! ---------------------------------------------------------------------------------
    void write_inverse(std::string filename = "") const {
        filename = get_outfilename(filename).append(".inverse.raw");
        amm::timer t;
        AMM_log_info << "Writing lowres function to (" << filename << ")...";
        fflush(stdout);
        amm::utils::write_binary<_Tp>(m_lowres_func, filename);
        t.stop();
        AMM_logc_info << " done!" << t  << std::endl;
    }
    bool read_inverse(std::string filename = "") {
        filename = get_outfilename().append(".inverse.raw");
        if (!amm::utils::file_exists(filename)) {
          return false;
        }
        amm::timer t;
        AMM_log_info << "Loading lowres function from (" << filename << ")...";
        fflush(stdout);
        amm::utils::load_binary<_Tp>(m_lowres_func, filename, m_nTreePoints);
        t.stop();
        AMM_logc_info << " done!" << t  << std::endl;
        return true;
    }

    // the caller should delete the pointer to vtk_mesh
    auto write_vtkamm(std::string filename = "") const {
#ifdef AMM_USE_VTK
        amm::AMM_field_data field_data (m_stream_end_threshold, m_stream_chunk_count);

        vtkAMM *vtk_mesh = new vtkAMM();
        if (m_treeDims[2] == 1) vtk_mesh->amm2vtk(*this->m_octree2, field_data);
        else                    vtk_mesh->amm2vtk(*this->m_octree3, field_data);

        filename = get_outfilename(filename).append(".vtu");
        vtk_mesh->write(filename);
        return vtk_mesh;
#endif
    }

    //! ---------------------------------------------------------------------------------
    //! compute the representation
    //! ---------------------------------------------------------------------------------
    void compute_amm() {

        // initialize the stream parameters
        init_stream();

        // only use m_wcoeffs_filtered if we want to validate the mesh or write lowres
        if (m_config.amm_validateOutput || m_config.amm_writeLowres) {
            AMM_log_info << "Preparing to capture filtered coefficients... ";
            fflush(stdout);
            if (m_wcoeffs_filtered.size() != m_nTreePoints) {
                m_wcoeffs_filtered.resize(m_nTreePoints, _Tp(0));
            }
            else {
                std::fill(m_wcoeffs_filtered.begin(), m_wcoeffs_filtered.end(), _Tp(0));
            }
            AMM_logc_info << " done!\n";
        }

        //-------------------------------------------------------------------------------
        // now, actually compute the representation
        print_memory();
        amm::timer t;
        t.start();
#ifdef AMM_ENABLE_PRECISION
        const bool precision_enabled = true;
#else
        const bool precision_enabled = false;
#endif

        AMM_log_info << "Creating AMM ["
                     << "precision="<<precision_enabled<<", "
                     << "rect="<< m_config.amm_enableRectangular<<", "
                     << "vacc="<< m_config.amm_enableImproper<<", "
                     << "wvlets="<<wavelets::get_wavelets_name(m_config.amm_waveletsType)<<"]";

        if (m_config.amm_streamEnd >= 0) {
            AMM_logc_info << " for threshold " << m_config.amm_streamEnd << " " << m_config.amm_streamEndCriterion;
        }
        AMM_logc_info << "...\n";

        //-------------------------------------------------------------------------------
        while(update_amm()) {
            print_memory();
        }
        print_memory();

        //-------------------------------------------------------------------------------
        AMM_log_info << "Created AMM using "<< m_stream_total_count <<" wavelet coefficients "
                     << "("  << (m_stream_total_bits+CHAR_BIT-1)/CHAR_BIT << " bytes)\n";

#ifdef AMM_DEBUG_TRACK_CONFIGS
        if (m_dataDims[2] == 1) m_octree2->dump_configs();
        else                    m_octree3->dump_configs();
#endif
        t.stop();

        //AMM_log_info << "TOTAL TIME = " << t << "\n";
        //AMM_log_info << "STREAM TIME = " << m_stream_time * 0.001 << " sec \n";

#ifndef AWR_VALIDATE_PER_UPDATE
        this->validation();
#endif
    }

    void print_memory() const {
        return;

        AMM_log_info << std::setprecision(8) << "\n---- AWR memory ----           \n";
        AMM_log_info << "\tm_func               = " << m_func.size() << "    [ " << float(m_func.size()*sizeof(_Tp))/1024.0 << " KB]\n";
        AMM_log_info << "\tm_wcoeffs            = " << m_wcoeffs.size() << "    [ " << float(m_wcoeffs.size()*sizeof(_Tp))/1024.0 << " KB]\n";
        AMM_log_info << "\tm_wcoeffs_filtered   = " << m_wcoeffs_filtered.size() << "    [ " << float(m_wcoeffs_filtered.size()*sizeof(_Tp))/1024.0 << " KB]\n";
        AMM_log_info << "\tm_lowres_func        = " << m_lowres_func.size() << "    [ " << float(m_lowres_func.size()*sizeof(_Tp))/1024.0 << " KB]\n";
        //AMM_log_info << "\tm_wcoeff_uindices       = " << m_wcoeff_uindices.size() << "    [ " << float(m_wcoeff_uindices.size()*sizeof(_Tp))/1024.0 << " KB]\n";

        if (m_octree2)      m_octree2->print_memory();
        if (m_octree3)      m_octree3->print_memory();
        if (m_dataStream)   m_dataStream->print_memory();
        AMM_log_info << "---- end AWR memory ----           \n\n";
    }

private:
    //! ---------------------------------------------------------------------------------
    //! For the given data size, this function determines how to expand the data
    //! to make it a square of 2^L + 1
    //! ---------------------------------------------------------------------------------
    void init_sizes(const size_t &X, const size_t &Y, const size_t &Z = 1) {

        // input dimensions
        m_dataDims[0] = X;
        m_dataDims[1] = Y;
        m_dataDims[2] = Z;

        m_fileDims[0] = X;
        m_fileDims[1] = Y;
        m_fileDims[2] = Z;

        // internal dimensions
        if (Z == 1) {
            _VertexType2 tsize = amm::octree_utils<2>::size_tree(m_dataDims, AMM_MAX_DATA_DEPTH);
            _VertexType2 osize = amm::octree_utils<2>::size_domain(m_dataDims, AMM_MAX_DATA_DEPTH);

            m_treeDims[0] = tsize[0];   m_dataDims[0] = osize[0];
            m_treeDims[1] = tsize[1];   m_dataDims[1] = osize[1];
            m_treeDims[2] = 1;          m_dataDims[2] = 1;
        }
        else {
            _VertexType3 tsize = amm::octree_utils<3>::size_tree(m_dataDims, AMM_MAX_DATA_DEPTH);
            _VertexType3 osize = amm::octree_utils<3>::size_domain(m_dataDims, AMM_MAX_DATA_DEPTH);

            m_treeDims[0] = tsize[0];   m_dataDims[0] = osize[0];
            m_treeDims[1] = tsize[1];   m_dataDims[1] = osize[1];
            m_treeDims[2] = tsize[2];   m_dataDims[2] = osize[2];
        }

        size_t n = m_treeDims[0];
        m_dataDepth = 0;
        while (n >>= 1) ++m_dataDepth;

        m_nDataPoints = m_dataDims[0]*m_dataDims[1]*m_dataDims[2];
        m_nTreePoints = m_treeDims[0]*m_treeDims[1]*m_treeDims[2];

        AMM_log_info << "\tfile dims: ["<< m_fileDims[0] << " x " << m_fileDims[1] << " x " << m_fileDims[2] << "] : " << m_fileDims[0]*m_fileDims[1]*m_fileDims[2] << "\n";
        AMM_log_info << "\ttree dims: ["<< m_treeDims[0] << " x " << m_treeDims[1] << " x " << m_treeDims[2] << "] : " << m_nTreePoints << "\n";
        AMM_log_info << "\tdomain dims: ["<< m_dataDims[0] << " x " << m_dataDims[1] << " x " << m_dataDims[2] << "] : " << m_nDataPoints << "\n";
        AMM_log_info << "\tdata depth: ["<< int(m_dataDepth) << "]\n";
    }

    //! ---------------------------------------------------------------------------------
    //! reset the representation
    //! ---------------------------------------------------------------------------------
    void reset() {
        delete m_octree2;               m_octree2 = nullptr;
        delete m_octree3;               m_octree3 = nullptr;
    }

    //! ---------------------------------------------------------------------------------
    //! compute wavelet transforms
    //! ---------------------------------------------------------------------------------
    void compute_wavelet_transform(const std::vector<_Tp> &input, std::vector<_Tp> &output, int8_t direction) {

        // direction is forward or inverse!
        if (direction != 1 && direction != -1)
            return;

        AMM_error_invalid_arg(input.size()!= m_nTreePoints, "Invalid input of size %lu; expected %lu", input.size(), m_nTreePoints);

        // new assertion after duong disabled the subband arrangement
        const bool in_place = (m_wcoeff_arrangement_type == EnumArrangement::Original);

        AMM_error_invalid_arg(!in_place, "Only \"Original\" arrangement of wavelet coefficients is supported!");
        /*if (!in_place && m_wcoeff_uindices.empty()) {
            // compute the mapping between wavelet and spatial domains
            AMM_log_info << "Computing wavelet2spatial mapping...";
            fflush(stdout);
            amm::timer t;
            wavelets::wavelet2spatial(itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]), waveletMaxLevels, m_wcoeff_uindices);
            t.stop();
            AMM_logc_info << " done!" << t  << std::endl;
        }*/


        const TypeScale waveletMaxLevels = m_dataDepth - m_config.amm_waveletsDepth;
        TypeScale lx = waveletMaxLevels;
        TypeScale ly = waveletMaxLevels;
        TypeScale lz = (m_treeDims[2] == 1) ? 0 : waveletMaxLevels;

        // initiailze the computer if needed
        if (m_wcomputer == nullptr) {
            m_wcomputer = new _WvTp(m_config.amm_waveletsType, waveletMaxLevels, m_config.amm_waveletsNormalized, in_place);
        }

        // and go!
        AMM_log_info << "Computing " << ((direction == 1) ? "forward " : "inverse ")
                     << "wavelet transform "
                     << "for ["<< m_treeDims[0] << " x " << m_treeDims[1] << " x " << m_treeDims[2] << "] data "
                      << "and [" << int(lx) << " x " << int(ly) << " x " << int(lz) << "] levels "
                     << "(normalized basis = " << m_config.amm_waveletsNormalized << ")... ",
        fflush(stdout);




        amm::timer t;
        output.clear();
        output.assign(input.begin(), input.end());
        if (direction == 1) {
            m_wcomputer->forward_transform(output.data(),
                                           itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]),
                                           lx, ly, lz);
        }
        else {
            m_wcomputer->inverse_transform(output.data(),
                                           itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]),
                                           lx, ly, lz);
        }

        t.stop();
        AMM_logc_info << " done!" << t  << "\n";
        AMM_log_info << " \t" << ((direction == 1) ? "wcoeffs " : "lowres_func ")
                     << "range = " << amm::utils::frange(m_wcoeffs) << "\n";
    }

/*
    void compute_wavelets() {

        const bool in_place = (m_wcoeff_arrangement_type == EnumArrangement::Original);
        const TypeScale waveletMaxLevels = m_dataDepth - m_config.amm_waveletsDepth;

        //if (!in_place) {
        //    // compute the mapping between wavelet and spatial domains
        //    AMM_log_info << "Computing wavelet2spatial mapping...";
        //    fflush(stdout);

        //    amm::timer t;
        //    //wavelets::wavelet2spatial(itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]), waveletMaxLevels, m_wcoeff_uindices);
        //    t.stop();
        //    AMM_logc_info << " done!" << t  << std::endl;
        //}

        // compute wavelets
        AMM_log_info << "Computing forward wavelet transform for " << int(waveletMaxLevels) << " levels "
                     << "(normalized basis = " << m_config.amm_waveletsNormalized << ")... ",
        fflush(stdout);

        amm::timer t;
        m_wcomputer = new _WvTp(m_config.amm_waveletsType, waveletMaxLevels, m_config.amm_waveletsNormalized, in_place);
        m_wcoeffs.assign(m_func.begin(), m_func.end());
        m_wcomputer->forward_transform(m_wcoeffs.data(), itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]),
                                       waveletMaxLevels, waveletMaxLevels, (m_treeDims[2] == 1) ? 0 : waveletMaxLevels);

        // no need to carry the memory for the function
        // if we wont compute the quality later
        if (!m_config.amm_validateOutput)
            m_func.clear();

        t.stop();
        AMM_logc_info << " done!" << t  << "\n";
        AMM_log_info << " \twcoeff range = " << amm::utils::frange(m_wcoeffs) << "\n";
    }
*/
    //! ---------------------------------------------------------------------------------
    //! filter the coefficients outside the true data domain
    //! ---------------------------------------------------------------------------------
    void filter_external_wcoeffs() {
#ifndef AWR_FILTER_EXTERNAL_COEFFICIENTS
        return;
#endif
        wavelets::filter_external_wavelets(m_wcoeffs, m_dataDims, m_treeDims,
                                           m_dataDepth - m_config.amm_waveletsDepth,
                                           m_config.amm_waveletsType);

        auto frng = amm::utils::frange(m_wcoeffs);
        AMM_log_info << "\twcoeff range: "
                     << "("<<m_treeDims[0]<<","<<m_treeDims[1]<<","<<m_treeDims[2]<<"), "
                     << "range = "<<frng<<"\n";
    }

    //! ---------------------------------------------------------------------------------
    //! initialize the stream using config
    //! ---------------------------------------------------------------------------------
    void init_stream() {

        AMM_error_invalid_arg((streams::is_precision_stream(m_streamType) &&
                               m_config.amm_streamEndCriterion.compare("kb") != 0),
                               "Precision streams support \"kb\" end criterion only!\n");


        AMM_log_info << "Initializing stream ("
                     << streams::get_stream_name(m_streamType).c_str() << ", "
                     << wavelets::get_arrangement_name(m_wcoeff_arrangement_type).c_str() <<")...";
        fflush(stdout);

        //-------------------------------------------------------------------------------
        if (m_config.amm_streamEnd > 0) {
            if (m_config.amm_streamEndCriterion.compare("val") == 0) {
                m_stream_end_threshold = m_config.amm_streamEnd;
                //m_stream_end_coeffs_wanted = std::count_if(m_wcoeffs.begin(), m_wcoeffs.end(),
                //                                           [this](const _Tp v) { return fabs(v)>=this->m_stream_end_threshold; });
                //m_stream_end_bits_wanted = m_stream_end_coeffs_wanted * sizeof(_Tp) * CHAR_BIT;
            }
            else if (m_config.amm_streamEndCriterion.compare("count") == 0) {
                m_stream_end_coeffs_wanted = size_t(m_config.amm_streamEnd);
                //m_stream_end_bits_wanted = m_stream_end_coeffs_wanted * sizeof(_Tp) * CHAR_BIT;
            }
            else if (m_config.amm_streamEndCriterion.compare("kb") == 0) {
                m_stream_end_bits_wanted = size_t(m_config.amm_streamEnd * 1024. * double(CHAR_BIT));
                //m_stream_end_coeffs_wanted = m_stream_end_bits_wanted / sizeof(_Tp) / CHAR_BIT;
            }
            else {
                AMM_error_invalid_arg(true, "Invalid stream end criterion. Must be one of val/count/kb");
            }
        }

        if (m_streamType == EnumStream::By_Coeff_Wavelet_Norm || m_streamType == EnumStream::By_Magnitude) {
            if (!m_config.amm_validateOutput) {
                m_wcoeffs.clear();
                m_wcoeffs_filtered.clear();
            }
        }

        // set the chunk size
        if (m_config.amm_chunkUnit == "kb") {
            m_stream_chunk_bits = size_t(m_config.amm_chunkSize * 1024. * double(CHAR_BIT));
            //m_stream_chunk_count = m_stream_chunk_bits / sizeof(_Tp) / CHAR_BIT;
        }
        else if (m_config.amm_chunkUnit == "count") {
            m_stream_chunk_count = m_config.amm_chunkSize;
            //m_stream_chunk_bits = m_stream_chunk_count * sizeof(_Tp) * CHAR_BIT;
        }

        AMM_logc_info << " done!" << std::endl;
        AMM_log_info << "\tchunk = " << int64_t(m_stream_chunk_count) << " count"
                     << " / " << int64_t(m_stream_chunk_bits) << " bits!\n"
                     << "\tend   = " << int64_t(m_stream_end_coeffs_wanted) << " count"
                     << " / " << int64_t(m_stream_end_bits_wanted) << " bits"
                     << " / " <<  m_stream_end_threshold << " value!\n";

        //-------------------------------------------------------------------------------
        AMM_log_info << "Building stream...";
        fflush(stdout);
        amm::timer t;

        // initialize the actual data stream
        m_dataStream = new _StTp(m_wcoeffs.data(), m_treeDims,
                                 m_streamType, m_wcoeff_arrangement_type,
                                 m_dataDepth - m_config.amm_waveletsDepth,
                                 m_stream_end_threshold);
        t.stop();
        AMM_logc_info << " done!" << t  << std::endl;
    }

    //! ---------------------------------------------------------------------------------
    //! extrapolate the data to fit the tree's domain
    //! ---------------------------------------------------------------------------------
    template <typename TypeInputFile = _Tp>
    void preprocess(const TypeInputFile *data) {

        // copy input data onto the output grid
        amm::utils::copy_data<TypeInputFile, _Tp>(m_fileDims, data, m_treeDims, m_func);

        // if output grid is larger than the input grid
        bool extrapolation_needed = false;
        for(size_t d = 0; d < 3; d++) {
            extrapolation_needed |= (m_treeDims[d] > m_dataDims[d]);
        }

        if (extrapolation_needed) {
            switch(m_config.amm_extrapolationType) {
                case (EnumExtrapolation::Zero):             break;      // do nothing! m_func is initialized with zero
                case (EnumExtrapolation::Linear):           amm::utils::extrapolate_linearly(m_dataDims, m_treeDims, m_func);  break;
                case (EnumExtrapolation::LinearLifting):    _WvTp::extrapolate(m_dataDims, m_treeDims, m_func);       break;
                default:                                    AMM_error_invalid_arg(true, "Invalid extrapolation type %d!", m_config.amm_extrapolationType);
            }
        }
    }

    //! ---------------------------------------------------------------------------------
    //! perform all validation
    //! ---------------------------------------------------------------------------------
    void validation() {

        // having floating point overflow for 2014 f32 miranda dataset!
        //const bool bypass_error = (m_treeDims[0] == 2048);
        const bool bypass_error = sizeof(_Tp) == 4;

        if (m_config.amm_validateOutput || m_config.amm_writeLowres) {

          // if a low-res file exists with the exact same parameters
          const bool compute_needed = !read_inverse();
          if (compute_needed) {
             compute_wavelet_transform(m_wcoeffs_filtered, m_lowres_func, -1);
             if (m_config.amm_writeLowres) {
               write_inverse();
             }
           }
        }

        if (m_config.amm_validateOutput) {

            double psnr, rmse, mse;
            this->compute_quality(psnr, rmse, mse, true);

            if (!this->validate_mesh_vertices(false)) {
                if (!bypass_error) exit(1);
            }
            if (!this->validate_mesh_function(true)) {
              if (!bypass_error) exit(1);
            }
        }
    }


    //! ---------------------------------------------------------------------------------
    //! compute the reconstruction quality
    //! ---------------------------------------------------------------------------------
/*
    void reconstruct_from_wcoeffs() {

        const TypeScale waveletMaxLevels = m_dataDepth - m_config.amm_waveletsDepth;

        // initialize the wavelet computer for inverse transform later on
        if (m_wcomputer == nullptr) {
            const bool in_place = (m_wcoeff_arrangement_type == EnumArrangement::Original);
            m_wcomputer = new _WvTp(m_config.amm_waveletsType, waveletMaxLevels, m_config.amm_waveletsNormalized, in_place);
        }

        AMM_log_debug << "Computing inverse wavelet transform for " << int(waveletMaxLevels)
                      << " levels (normalized basis = " << m_config.amm_waveletsNormalized << ")...";
        fflush(stdout);
        amm::timer t;

        m_lowres_func.assign(m_wcoeffs_filtered.begin(), m_wcoeffs_filtered.end());
        m_wcomputer->inverse_transform(m_lowres_func.data(), itype(m_treeDims[0]), itype(m_treeDims[1]), itype(m_treeDims[2]),
                                       waveletMaxLevels, waveletMaxLevels, (m_treeDims[2] == 1) ? 0 : waveletMaxLevels);

        AMM_logc_debug << " done!" << t  << std::endl;
    }
*/
    void compute_quality(double &psnr, double &rmse, double &mse, bool verbose = false) {

        if (m_func.empty() || m_lowres_func.empty()) {
            return;
        }

        AMM_log_debug << "Computing reconstruction quality...";
        fflush(stdout);
        psnr = amm::utils::psnr(m_lowres_func.data(), m_func.data(), m_dataDims);
        rmse = amm::utils::rmse(m_lowres_func.data(), m_func.data(), m_dataDims);
        mse = amm::utils::mse(m_lowres_func.data(), m_func.data(), m_dataDims);
        AMM_log_debug << "\tPSNR = " << psnr << "; RMSE = " << rmse << "; MSE = " << mse << "\n";
    }

    //! ---------------------------------------------------------------------------------
    //! validate the vertices of the mesh
    //! ---------------------------------------------------------------------------------
    bool validate_mesh_vertices(bool verbose=false) const {

        if (this->m_lowres_func.empty())
            return true;

        const bool success = (m_treeDims[2] == 1)
              ? m_octree2->test_vertices_against_function(this->m_lowres_func, verbose)
              : m_octree3->test_vertices_against_function(this->m_lowres_func, verbose);

        if (!success) {
            write_vtkamm("amm_with_verts_error");
        }
        return success;

    }

    //! ---------------------------------------------------------------------------------
    //! valdate the complete function
    //! ---------------------------------------------------------------------------------
    bool validate_mesh_function(bool verbose=false) const {

        if (this->m_lowres_func.empty())
            return true;

        const bool success = (m_treeDims[2] == 1)
              ? m_octree2->test_function_against_function(this->m_lowres_func, m_octree2->dbox1(), verbose)
              : m_octree3->test_function_against_function(this->m_lowres_func, m_octree3->dbox1(), verbose);

        if (!success) {
            write_vtkamm("amm_with_func_error");
        }
        return success;
    }



    //! ---------------------------------------------------------------------------------
    //! create a tag to represent the config (used for output filename)
    //! ---------------------------------------------------------------------------------
    std::string get_tag() const {
#ifdef AMM_ENABLE_PRECISION
        const bool precision_enabled = true;
#else
        const bool precision_enabled = false;
#endif
        std::string s ("");
        s = s.append("_r").append(std::to_string(m_config.amm_enableRectangular))
             .append("_v").append(std::to_string(m_config.amm_enableImproper))
             .append("_p").append(std::to_string(precision_enabled))
             .append("_s").append(streams::get_stream_name(m_streamType))
             .append("_e").append(std::to_string(m_config.amm_streamEnd)).append(m_config.amm_streamEndCriterion);

        if (m_config.amm_chunkSize < 0) {
            s = s.append("_n0");
        }
        else {
            s = s.append("_n").append(std::to_string(m_config.amm_chunkSize)).append(m_config.amm_chunkUnit);
        }
        return s;
    }

    std::string get_outfilename(std::string filename = "") const {

        // if filename is empty, use filename = basename(input_filename)
        if (filename.empty()) {
            filename = m_config.amm_inputFilename;
            filename = filename.substr(1+filename.find_last_of("/\\"));
            filename = filename.append(get_tag());
        }
        // if outpath is given, then prepend it
        if (!m_config.amm_outPath.empty()) {
            filename.insert(0, "/");
            filename.insert(0, m_config.amm_outPath);
        }
        return filename;
    }

    //! ---------------------------------------------------------------------------------
    //! update amm for one chunk
    //! ---------------------------------------------------------------------------------
    template <uint8_t Dim>
    bool update_amm(_CrtTp<Dim> &tcreator, _AMMTp<Dim> &octree) {

        bool verbose=false;

#ifdef AMM_STAGE_WCOEFFS
        bool do_staging = true;//false;//streams::is_precision_stream(m_streamType);
#else
        bool do_staging = false;
#endif

        if (verbose){
            AMM_log_info << "\n----------------------- \n";
        }

        static size_t update_cnt = 0;   // number of times the stream has been fetched
        typename streams::dataStream<_Tp>::coeff c;

        AMM_logc_info << "\n";
        AMM_log_info << (do_staging ? "Fetching" : "Processing") << " chunk [" << update_cnt++ << "]:";
        if (int(m_stream_chunk_count) > 0) {
            AMM_logc_info << " adding " << int64_t(m_stream_chunk_count) << " coefficients"
                          << " (" << int64_t(m_stream_chunk_bits) << " bits)...";
        }
        else if (int(m_stream_chunk_bits) > 0) {
            AMM_logc_info << " adding " << int64_t(m_stream_chunk_bits) << " bits"
                          << " (" << int64_t(m_stream_chunk_count) << " coefficients)...";
        }
        fflush(stdout);
        if (verbose){
            std::cout << "\n";
        }
        amm::timer t;

        // for precision stream, convert bytes to bits
        size_t chunk_ncoeffs = 0, chunk_nbits = 0;
        while (chunk_ncoeffs < m_stream_chunk_count && chunk_nbits < m_stream_chunk_bits) {

            int ncoeffs = 0, nbits = 0;

            // no more bits to stream
            amm::timer s;
            if (!m_dataStream->FetchNextCoefficient(c, &ncoeffs, &nbits)) {
                m_stream_done = true;
                break;
            }
            s.stop();
            m_stream_time += s.elapsed();

            const TypeScale olvl = m_dataDepth - c.wlvl;
            _VertexTypeD<Dim> up = (2 == Dim) ? _VertexTypeD<Dim>(TypeCoord(c.x), TypeCoord(c.y))
                                              : _VertexTypeD<Dim>(TypeCoord(c.x), TypeCoord(c.y), TypeCoord(c.z));
            if (verbose){
                // already incremented the chunk above
                std::cout << "["<<chunk_ncoeffs<<"] adding stencil: level " << int(olvl) << ", type = " << int(c.wtype) << " :: "
                          <<  up << " = " << std::fixed << std::setprecision (12) << c.val << "\n";
            }

            if (do_staging)     tcreator.add_stencil(up, c.val, olvl, c.wtype);
            else                tcreator.create_stencil(up, c.val, olvl, c.wtype);

            if (m_config.amm_validateOutput)
                m_wcoeffs_filtered[c.widx] += c.val;

#ifdef AWR_VALIDATE_PER_COEFF
            if (ncounter > 1) {
                this->validation();
            }
#endif

            chunk_ncoeffs += ncoeffs;
            chunk_nbits += nbits;
            m_stream_total_bits += nbits;
            m_stream_total_count += ncoeffs;

            // done, based on threshold
            if (m_stream_total_bits >= m_stream_end_bits_wanted || m_stream_total_count >= m_stream_end_coeffs_wanted) {
                m_stream_done = true;
                break;
            }
        }

        t.stop();
        AMM_logc_info << " done!" << t;
        /*if (int(m_stream_chunk_count) > 0) {
            AMM_logc_info << " added " << chunk_ncoeffs << " coefficients"
                          << " (" << chunk_nbits << " bits)\n";
        }
        else if (int(m_stream_chunk_bits) > 0) {*/
            AMM_logc_info << " added " << chunk_nbits << " bits"
                          << " (" << chunk_ncoeffs << " coefficients)\n";
        //}

        if (chunk_ncoeffs == 0 || chunk_nbits == 0)
            return false;

        // ----------------------------------------------------------------------------------------
        if (do_staging) {
            AMM_log_info << "Processing chunk [" << update_cnt-1 << "]...";
            fflush(stdout);
            amm::timer t2;
            tcreator.update();
            t2.stop();
            AMM_logc_info << " done!" << t2 << "\n";
        }

        // ----------------------------------------------------------------------------------------
        // do profiling
        if (verbose){
            AMM_log_info << "\n----------------------- \n";
        }

#ifdef AWR_WRITE_PROFILING
        std::string profile_name = this->m_config.amm_inputFilename;
        profile_name = profile_name.append(get_tag()).append("_profile.csv");
        amm::amm_profile pp;
        octree.profile_prefinalize(pp);
#endif

        octree.finalize();


        if (verbose){
            AMM_log_info << "\n----------------------- \n";
        }

#ifdef AWR_VALIDATE_PER_UPDATE
        this->validation();
#endif

#ifdef AWR_WRITE_PROFILING
#ifndef AWR_VALIDATE_PER_UPDATE
        this->validation();
#endif

        octree.profile_postfinalize(pp);

        if (m_stream_chunkUnit.compare("bytes") == 0) {
            octree.profile_quality(pp, m_current_chunkCount/sizeof(_Tp), m_current_chunkCount, psnr);
        }
        else {
            octree.profile_quality(pp, m_current_chunkCount, m_current_chunkCount*sizeof(_Tp), psnr);
        }
        pp.write_csv(profile_name, update_cnt==1);
#endif

        // ----------------------------------------------------------------------------------------
        return !m_stream_done;
    }


    bool update_amm() {

        if (m_treeDims[2] == 1) {

            static _CrtTp<2> tcreator (m_config.amm_waveletsType,
                                       m_dataDims, m_config.amm_enableRectangular, m_config.amm_enableImproper);
            m_octree2 = const_cast<_AMMTp<2>*>(tcreator.output());

            return update_amm<2>(tcreator, *m_octree2);
        }
        else {
            static _CrtTp<3> tcreator (m_config.amm_waveletsType,
                                       m_dataDims, m_config.amm_enableRectangular, m_config.amm_enableImproper);
            m_octree3 = const_cast<_AMMTp<3>*>(tcreator.output());

            return update_amm<3>(tcreator, *m_octree3);
        }
    }



#ifdef AMM_USE_VTK
    //! ---------------------------------------------------------------------------------
    //! construct representation from unstructured vtk data
    //! (data assumed to obey AWR mesh constraints for now)
    //! ---------------------------------------------------------------------------------
    void load_vtkamm(vtkAMM &vtk_mesh) {

        // check the bounds and dimensionality
        size_t dsize[3] = {0, 0, 0};
        const TypeDim dim = vtk_mesh.dim(dsize);
        init_sizes(dsize[0], dsize[1], dsize[2]);

        // use the mesh creator to create the tree
        if(dim == 2){
            m_octree2 = vtk_mesh.vtk2amm<_Tp, 2, AMM_MAX_DATA_DEPTH>(m_config.amm_enableRectangular, m_config.amm_enableImproper);
        }
        else {
            m_octree3 = vtk_mesh.vtk2amm<_Tp, 3, AMM_MAX_DATA_DEPTH>(m_config.amm_enableRectangular, m_config.amm_enableImproper);
        }
    }
    void load_vtkamm(const std::string &filename) {

        vtkAMM vtk_mesh;
        vtk_mesh.read(filename);
        this->load_vtkamm(vtk_mesh);
    }

    // the caller should delete the pointer to vtk_mesh
    auto save_vtkamm(std::string filename = "") const {
        // DUONG: TODO: add back some field data

        AMM_log_warn << "Adding incorrect field data to vtk!\n";
        amm::AMM_field_data field_data (0,0);
        //amm::AMM_field_data field_data (m_stream_end_threshold, m_current_chunkCount);

        vtkAMM *vtk_mesh = new vtkAMM();
        if (m_treeDims[2] == 1) vtk_mesh->amm2vtk(*this->m_octree2, field_data);
        else                    vtk_mesh->amm2vtk(*this->m_octree3, field_data);

        if (filename.length() == 0) {
            filename = m_config.amm_inputFilename;
            filename = filename.append(get_tag()).append(".vtu");
        }

        vtk_mesh->write(filename);
        return vtk_mesh;
    }

    //! ---------------------------------------------------------------------------------
    //! initialize the representation from structured vtk data
    //! ---------------------------------------------------------------------------------
    void load_vtkDataArray(vtkDataArray *vdata, const size_t &X, const size_t &Y, const size_t &Z = 1) {

        _Tp *data = (_Tp *)vdata->GetVoidPointer(0);

        auto frng = amm::utils::frange(data, X*Y*Z);
        AMM_log_info << "\tgiven data: ("<<X<<", " <<Y<<", "<< Z<<"), range = ("<<frng<<")\n";
        AMM_error_invalid_arg(!AMM_is_zero(fabs(frng.second-frng.first)), "Appears to be incorrect data!\n");

        // this is the code that should be fixed
        init_sizes(X,Y,Z);
        this->preprocess<_Tp>(data);

        frng = amm::utils::frange(m_func);
        AMM_log_info << "\tpreprocessed data: ("<<m_treeDims[0]<<", " <<m_treeDims[1]<<", "<<m_treeDims[2]<<"), range = ("<<frng<<")\n";

        this->compute_wavelet_transform(m_func, m_wcoeffs, +1);

        // no need to carry the memory for the function
        // if we wont compute the quality later
        if (!m_config.amm_validateOutput) {
            m_func.clear();
        }

        this->filter_external_wcoeffs();
        if (!m_config.amm_wavOut.empty()) {
            write_wavelets(m_config.amm_wavOut);
        }
    }
#endif

    //! ---------------------------------------------------------------------------------
    //! some helper utilities for IO for debugging
    //! ---------------------------------------------------------------------------------
    //! write wavelet coefficients
    void write_wavelets(const std::string &filename) const {

        AMM_log_debug << "Writing " << m_nTreePoints << " wavelet coefficients to ["<<filename<<"]...";
        fflush(stdout);
        amm::utils::write_binary(m_wcoeffs, filename);
        AMM_logc_debug << " done!\n";
    }


#if 0
    //! ---------------------------------------------------------------------------------
    //! initialize the representation from streaming wavelet coefficients
    void load_swcoeffs(_Tp *swav, size_t* widx, size_t num_SW, size_t X, size_t Y, size_t Z = 1) {

        init_sizes(X,Y,Z);
        m_nSWavs = num_SW;
        X = std::min(m_inDims[0], m_dims[0]);
        Y = std::min(m_inDims[1], m_dims[1]);
        Z = std::min(m_inDims[2], m_dims[2]);

        std::pair<_Tp,_Tp> frng;
        frng = utils::frange(swav, m_nSWavs);
        AMM_log_info << "\tgiven %d wavelet coefficients: (%d, %d, %d), range = %f,%f\n", m_nSWavs, X, Y, Z, frng.first, frng.second);

        // copy the wavelet coefficients
        m_wcoeffs = new _Tp[m_nSWavs];
        for (size_t idx = 0; idx < m_nSWavs; idx++)
            m_wcoeffs[idx] = swav[idx];

        delete swav;
        swav = nullptr;

        // copy the wavelet indices
        m_swidx = new size_t[m_nSWavs];
        for (size_t idx = 0; idx < m_nSWavs; idx++)
            m_swidx[idx] = widx[idx];

        delete widx;
        widx = nullptr;
    }
#endif

    //! ---------------------------------------------------------------------------------
    //! some testing functions
    //! ---------------------------------------------------------------------------------
public:
#ifdef AWR_DEBUG_TEST_ITERATORS
    //! ---------------------------------------------------------------------------------
    void test_iterator() {

        char a;


        std::cout << " \n\n ----------cell iterator------- \n\n";
        {
            auto citer = m_octree2->iterator_cell(true);
            size_t cnt = 0;
            for(auto iter = citer.begin(); iter != citer.end(); iter++) {
                std::cout << " cell " << cnt++ << " of " << citer.size() << " :: "; m_octree2->print_node(*iter);

                auto cviter = m_octree2->iterator_cellvertex(*iter);
                for (auto citer = cviter.begin(); citer != cviter.end(); citer++){
                    std::cout << " --- " << m_octree2->idx2p(*citer) << " : " << "\n";
                }
            }
            std::cout << "prompt: "; std::cin >> a;
        }

        std::cout << " \n\n ----------vertex iterator------- \n\n";
        {
            auto citer = m_octree2->iterator_vertex(true);
            size_t cnt = 0;
            for(auto iter = citer.begin(); iter != citer.end(); iter++) {
                std::cout << " cell " << cnt++ << " of " << citer.size() << " :: " << m_octree2->idx2p(iter->first) << " = " << iter->second << "\n";

                auto cviter = m_octree2->iterator_vertexcell(iter->first);
                for (auto citer = cviter.begin(); citer != cviter.end(); citer++){
                    std::cout << " --- " << *citer << " : " << "\n";
                }
                exit(1);
            }
            std::cout << "prompt: "; std::cin >> a;
        }

        /*
#if 0
        {
            auto ibegin = m_octree2->sbegin_vertex();
            auto iend = m_octree2->send_vertex();
            size_t cnt = 0;
            for(auto iter = ibegin; iter != iend; iter++) {
                std::cout << " : " << cnt++ << " : " << iter->first << " , " << iter->second << std::endl;
            }
            std::cout << cnt << ", " << std::distance(ibegin, iend) << std::endl;
            std::cin >> a;
        }

        std::cout << " \n\n ----------leaf iterator------- \n\n";
        {
            auto ibegin = m_octree2->begin_leaf();
            auto iend = m_octree2->end_leaf();
            size_t cnt = 0;
            for(auto iter = ibegin; iter != iend; iter++) {
                std::cout << " : " << cnt++ << " : " << *iter << std::endl;
            }
            std::cout << cnt << ", " << std::distance(ibegin, iend) << std::endl;
            std::cin >> a;
        }
        {
            auto ibegin = m_octree2->sbegin_leaf();
            auto iend = m_octree2->send_leaf();
            size_t cnt = 0;
            for(auto iter = ibegin; iter != iend; iter++) {
                std::cout << " : " << cnt++ << " : " << *iter << std::endl;
            }
            std::cout << cnt << ", " << std::distance(ibegin, iend) << std::endl;
            std::cin >> a;
        }

        std::cout << " \n\n ----------penultimate iterator------- \n\n";
        {
            auto ibegin = m_octree2->begin_penultimate_node();
            auto iend = m_octree2->end_penultimate_node();
            size_t cnt = 0;
            for(auto iter = ibegin; iter != iend; iter++) {
                std::cout << " : " << cnt++ << " : " << *iter << std::endl;
            }
            std::cout << cnt << ", " << std::distance(ibegin, iend) << std::endl;
            std::cin >> a;
        }
        {
            auto ibegin = m_octree2->sbegin_penultimate_node();
            auto iend = m_octree2->send_penultimate_node();
            size_t cnt = 0;
            for(auto iter = ibegin; iter != iend; iter++) {
                std::cout << " : " << cnt++ << " : " << *iter << std::endl;
            }
            std::cout << cnt << ", " << std::distance(ibegin, iend) << std::endl;
            std::cin >> a;
        }
#endif
        //exit(1);
*/
    }
#endif

#ifdef AWR_DEBUG_TEST_STENCILS
    void create_single_2dstencil(const TypeWaveletCoefficient wtype, const bool use_rect) {

        const TypeScale olvl = 2;
        TypeTCreator2 tcreator2 = TypeTCreator2(m_inDims, m_waveletType, use_rect, 0);
        m_octree2 = tcreator2.output();

        if (wtype == TypeWaveletCoefficient::SS) {
            _VertexType2 up((TypeCoord)2);
            tcreator2.create_stencil(up, 1, olvl-1, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::WW) {
            _VertexType2 up((TypeCoord)3);
            tcreator2.create_stencil(up, 1, olvl, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::SW) {
            _VertexType2 up((TypeCoord)3, (TypeCoord)2);
            tcreator2.create_stencil(up, 1, olvl, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::WS) {
            _VertexType2 up((TypeCoord)2, (TypeCoord)3);
            tcreator2.create_stencil(up, 1, olvl, wtype);
        }

        std::string filename = "stencil_2d";
        filename = filename.append("_wtype_").append(std::to_string(int(wtype)))
                           .append("_rect_").append(std::to_string(use_rect))
                           .append(".vtk");
        this->write_vtk(filename);
    }
    void create_single_3dstencil(const TypeWaveletCoefficient wtype, const bool use_rect) {

        const TypeScale olvl = 2;
        TypeTCreator3 tcreator3 = TypeTCreator3(m_inDims, m_waveletType, use_rect, 0);
        m_octree3 = tcreator3.output();

        // todo: write rest of the stencils!
        if (wtype == TypeWaveletCoefficient::SSS) {
            _VertexType3 up((TypeCoord)2);
            tcreator3.create_stencil(up, 1, olvl-1, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::WWW) {
            _VertexType3 up((TypeCoord)3);
            tcreator3.create_stencil(up, 1, olvl, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::SSW) {
            _VertexType3 up((TypeCoord)2, (TypeCoord)2, (TypeCoord)3);
            tcreator3.create_stencil(up, 1, olvl, wtype);
        }
        else if (wtype == TypeWaveletCoefficient::WWS) {
            _VertexType3 up((TypeCoord)3, (TypeCoord)3, (TypeCoord)2);
            tcreator3.create_stencil(up, 1, olvl, wtype);
        }

        std::string filename = "stencil_3d";
        filename = filename.append("_wtype_").append(std::to_string(int(wtype)))
                           .append("_rect_").append(std::to_string(use_rect))
                           .append(".vtk");
        this->write_vtk(filename);
    }

    void dump_stencils() {

        if (m_inDims[2] == 1) {
            if (m_inDims[0] == 7 && m_inDims[1] == 7) {
                create_single_2dstencil(TypeWaveletCoefficient::WW, 1);
                create_single_2dstencil(TypeWaveletCoefficient::WW, 0);
            }
            else if (m_inDims[0] == 5 && m_inDims[1] == 5) {
                create_single_2dstencil(TypeWaveletCoefficient::SS, 1);
                create_single_2dstencil(TypeWaveletCoefficient::SS, 0);
            }
            else if (m_inDims[0] == 5 && m_inDims[1] == 7) {
                create_single_2dstencil(TypeWaveletCoefficient::WS, 1);
                create_single_2dstencil(TypeWaveletCoefficient::WS, 0);
            }
            else if (m_inDims[0] == 7 && m_inDims[1] == 5) {
                create_single_2dstencil(TypeWaveletCoefficient::SW, 1);
                create_single_2dstencil(TypeWaveletCoefficient::SW, 0);
            }
        }
        else {
            if (m_inDims[0] == 7 && m_inDims[1] == 7 && m_inDims[2] == 7) {
                create_single_3dstencil(TypeWaveletCoefficient::WWW, 0);
                create_single_3dstencil(TypeWaveletCoefficient::WWW, 1);
            }
            else if (m_inDims[0] == 5 && m_inDims[1] == 5 && m_inDims[2] == 5) {
                create_single_3dstencil(TypeWaveletCoefficient::SSS, 0);
                create_single_3dstencil(TypeWaveletCoefficient::SSS, 1);
            }
            else if (m_inDims[0] == 5 && m_inDims[1] == 5 && m_inDims[2] == 7) {
                create_single_3dstencil(TypeWaveletCoefficient::SSW, 0);
                create_single_3dstencil(TypeWaveletCoefficient::SSW, 1);
            }
            else if (m_inDims[0] == 7 && m_inDims[1] == 7 && m_inDims[2] == 5) {
                create_single_3dstencil(TypeWaveletCoefficient::WWS, 0);
                create_single_3dstencil(TypeWaveletCoefficient::WWS, 1);
            }
        }

        exit(1);
    }
#endif
};

//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
#endif
