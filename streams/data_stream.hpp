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

#ifndef DATASTREAM_H
#define DATASTREAM_H
#pragma once

#include "amm/types/dtypes.hpp"
#include "amm/utils/exceptions.hpp"
#include "wavelets/utils.hpp"
#include "streams/utils.hpp"


namespace streams {

//! -------------------------------------------------------------------------------------
//! Encapsulator for data streams
//!
//! @tparam t: data type of input stream
//!
//! -------------------------------------------------------------------------------------
template <typename t>
class dataStream {

private:
    const EnumStream m_type;
    const EnumArrangement m_arrangement;
    const size_t X, Y, Z, N;

    // internal state of the data streams
    state<t> m_state;

    // state for spatial stream
    const t* m_data;

    double m_threshold = 0;
    uint8_t m_waveletMaxLevels;
    size_t m_x, m_y, m_z, m_idx;
    size_t m_coeffs_read = 0;
    size_t m_bits_read = 0;
    bool m_done = false; // we have streamed up to the --end criterion
    std::vector<coeff_lite<t>> m_coeffs_sorted;

    // HB added a new container for magn stream (score=magnitude, so save 8 bytes/coefficients)
    std::vector<coeff_litest<t>> m_coeffs_sorted_for_magn;

public:
    struct coeff {
        t val;
        size_t x, y, z;
        size_t widx;
        TypeScale wlvl;
        EnumWCoefficient wtype;
    };
    //! -------------------------------------------------------------------------------------
    //! constructor for spatial and magnitude streams
    //! threshold = number of bits for precision streams
    //! -------------------------------------------------------------------------------------
    dataStream(const t* data, size_t dims[], EnumStream Type, EnumArrangement Arrangement,
               uint8_t NumLevels, double end_threshold)
        : m_type(Type), m_arrangement(Arrangement), X(dims[0]), Y(dims[1]), Z(dims[2]),
          N(dims[0]*dims[1]*dims[2]), m_waveletMaxLevels(NumLevels),
          m_data(data), m_x(0), m_y(0), m_z(0), m_idx(0), m_threshold(end_threshold)
    {
        if (Type == EnumStream::By_RowMajor) {
            // do nothing
        }
        else if (Type == EnumStream::By_Subband_Rowmajor || Type == EnumStream::By_Coeff_Wavelet_Norm) {
            bool SkipLeadingZero = false;
            BuildStream(data, &m_coeffs_sorted, int(X), int(Y), int(Z), m_waveletMaxLevels,
                        m_arrangement, m_type, sizeof(t) * CHAR_BIT, SkipLeadingZero, &m_state);
        }
        else if (Type == EnumStream::By_Level || Type == EnumStream::By_BitPlane || Type == EnumStream::By_Wavelet_Norm) {
            bool SkipLeadingZero = true;
            BuildStream(data, &m_coeffs_sorted, int(X), int(Y), int(Z), m_waveletMaxLevels,
                        m_arrangement, m_type, sizeof(t) * CHAR_BIT, SkipLeadingZero, &m_state);
        }
        // sort the coefficients by magnitude and filter out ones smaller than the threshold
        else if (Type == EnumStream::By_Magnitude) {
            const size_t n = std::count_if(m_data, m_data+(X*Y*Z),
                                           [end_threshold](const t v)
                                           { return std::abs(v) > end_threshold; });
            m_coeffs_sorted_for_magn.reserve(n);
            for (size_t i = 0; i < N; ++i) {
                if (std::abs(data[i]) > end_threshold) {
                    m_coeffs_sorted_for_magn.emplace_back(data[i], i);
                }
            }
            // HB changed from stable_sort to sort
            std::sort(m_coeffs_sorted_for_magn.rbegin(), m_coeffs_sorted_for_magn.rend(),
                             [](const auto& c1, const auto& c2) {
                return std::abs(c1.val) < std::abs(c2.val);
            });
        }
        else {
            AMM_error_invalid_arg(true, "Invalid stream type!");
        }

        if (m_arrangement == EnumArrangement::Subband) {
            std::cerr << "Subband arrangement is unsupported\n";
            exit(1);
        }
    }

    //! -------------------------------------------------------------------------------------
    //! public funtion to extract the next coefficient from the stream
    //! -------------------------------------------------------------------------------------
    bool FetchNextCoefficient(coeff& c, int* ncoeffs, int* nbits) {

        if (m_type == EnumStream::By_RowMajor) {
            return this->GetNextCoefficient_rowmajor(c, ncoeffs, nbits);
        }
        else if (m_type == EnumStream::By_Magnitude) {
            return this->GetNextCoefficient_magnitude(c, ncoeffs, nbits);
        }
        else if (m_type == EnumStream::By_Subband_Rowmajor || m_type == EnumStream::By_Coeff_Wavelet_Norm ||
                 m_type == EnumStream::By_Level || m_type == EnumStream::By_BitPlane || m_type == EnumStream::By_Wavelet_Norm) {
            return this->GetNextCoefficient(c, ncoeffs, nbits);
        }
        else {
            std::cerr << " dataStream::GetNextCoefficients() invalid stream type!\n";
            exit(1);
        }
        return false;
    }

    void print_memory() const {
        AMM_log_info << "---- Data stream memory ---- \n";
        AMM_log_info << "\tm_data (same as m_wcoeffs)    = " << N                                   << "\n";
        AMM_log_info << "\tm_coeffs_sorted               = " << m_coeffs_sorted.size()  << " ("<<sizeof(coeff_lite<t>) << ")\n";
        AMM_log_info << "\tm_coeffs_sorted_for_magn      = " << m_coeffs_sorted_for_magn.size()  << " ("<<sizeof(coeff_litest<t>) << ")\n";
        AMM_log_info << "\tstate->DataCopy               = " << m_state.DataCopy.size()               << "\n";
        AMM_log_info << "\tstate->Subbands               = " << m_state.Subbands.size()               << "\n";
        AMM_log_info << "\tstate->Bitplanes              = " << m_state.Bitplanes.size()              << "\n";
        AMM_log_info << "\tstate->FirstOnes              = " << m_state.FirstOnes.size()              << "\n";
        AMM_log_info << "\tstate->Order                  = " << m_state.Order.size()                  << "\n";
        AMM_log_info << "\tstate->CoeffOrder             = " << m_state.CoeffOrder.size()             << "\n";
        AMM_log_info << "---- end Data stream memory ----\n";
    }


private:
    //! -------------------------------------------------------------------------------------
    //! get next coefficient
    //!
    //! @param val:     the coefficient value
    //! @param widx:    the index of the coefficient with respect to the arrangement
    //! @param sx:      the spatial x coordinates of the coefficient
    //! @param sy:      the spatial y coordinates of the coefficient
    //! @param sz:      the spatial z coordinates of the coefficient
    //! @param widx:    TODO: description
    //! @param wscale:  TODO: description
    //! @param wtype:   TODO: description
    //!
    //! -------------------------------------------------------------------------------------

    bool NextCoefficientWaveletNorm(coeff& c, int* ncoeffs, int* nbits) {
        const size_t xy = X * Y;
        for (; m_idx < m_coeffs_sorted.size();) {
            // wavelets are laid out in the spatial domain
            c.widx = m_coeffs_sorted[m_idx].widx;
            c.val = m_coeffs_sorted[m_idx].val;
            c.z = c.widx / xy;
            c.y = (c.widx % xy) / X;
            c.x = (c.widx % xy) % X;
            c.wtype = wavelets::get_wcoeff_lvl_type((itype)c.x, (itype)c.y, (itype)c.z, X, m_waveletMaxLevels, c.wlvl);
            ++(*ncoeffs);
            *nbits += sizeof(t) * CHAR_BIT;
            ++m_idx;
            return true;
        }
        return false;
    }

    //! -------------------------------------------------------------------------------------
    bool GetNextCoefficient(coeff& c, int* ncoeffs, int* nbits) {

        bool is_valid = m_type == EnumStream::By_Subband_Rowmajor ||
                        m_type == EnumStream::By_Coeff_Wavelet_Norm ||
                        m_type == EnumStream::By_Level ||
                        m_type == EnumStream::By_BitPlane ||
                        m_type == EnumStream::By_Wavelet_Norm;

        if (!is_valid) {
            std::cerr << " dataStream::GetNextCoefficients() invalid stream type!\n";
            exit(1);
        }
        if (m_type == EnumStream::By_Coeff_Wavelet_Norm) {
            return NextCoefficientWaveletNorm(c, ncoeffs, nbits);
        }
        *nbits = *ncoeffs = 0;
        if (!m_done && is_valid) {

            static std::vector<t> Coeff(1);
            static std::vector<std::array<int, 3>> CoeffPos3(1);

            if (m_type == EnumStream::By_Subband_Rowmajor) {
                if (!NextCoefficientsFullPrecision(&m_state, &Coeff, &CoeffPos3, m_threshold, ncoeffs, nbits)) {
                    m_done = true;
                    return false;
                }
            }
            else if (m_type == EnumStream::By_Level || m_type == EnumStream::By_BitPlane || m_type == EnumStream::By_Wavelet_Norm) {
                // DUONG: PERFORMANCE: getting one bit at a time is inefficient
                if (!NextCoefficients(&m_state, &Coeff, &CoeffPos3, ncoeffs, nbits)) {
                    m_done = true;
                    return false;
                }
            }
            m_bits_read += *nbits;
            m_coeffs_read += *ncoeffs;

            c.val = Coeff[0];
            c.x = CoeffPos3[0][0];
            c.y = CoeffPos3[0][1];
            c.z = CoeffPos3[0][2];

            const size_t xy = X * Y;
            c.widx = c.z * xy + c.y * X + c.x;
            if (m_arrangement == EnumArrangement::Original) {
                c.wtype = wavelets::get_wcoeff_lvl_type((itype)c.x,(itype)c.y,(itype)c.z, X, m_waveletMaxLevels, c.wlvl);
            } else if (m_arrangement == EnumArrangement::Subband) {
                std::cerr << "Subband arrangement is unsupported\n";
                exit(1);
            }
            //m_done = (m_coeffs_read >= m_coeffs_wanted) || (m_bits_read >= m_bits_wanted);
            return true;
        }

        return false;
    }

    bool GetNextCoefficient_magnitude(coeff& c, int* ncoeffs, int* nbits) {
        AMM_error_invalid_arg(m_type != EnumStream::By_Magnitude, "Invalid stream type");
        if (m_arrangement == EnumArrangement::Original) {
            const size_t xy = X * Y;
            for(; m_idx < m_coeffs_sorted_for_magn.size();) {
                // wavelets are laid out in the spatial domain
                c.widx = m_coeffs_sorted_for_magn[m_idx].widx;
                c.val = m_coeffs_sorted_for_magn[m_idx].val;
                c.z = c.widx / xy;
                c.y = (c.widx % xy) / X;
                c.x = (c.widx % xy) % X;
                c.wtype = wavelets::get_wcoeff_lvl_type((itype)c.x,(itype)c.y,(itype)c.z, X, m_waveletMaxLevels, c.wlvl);
                ++(*ncoeffs);
                *nbits += sizeof(t) * CHAR_BIT;
                ++m_idx;
                return true;
            }
            return false;
        }
        else { // subband arrangement
            std::cerr << "Subband arrangement is unsupported\n";
            exit(1);
        }
    }

    //! -------------------------------------------------------------------------------------
    //! get next coefficient from row major stream
    //! -------------------------------------------------------------------------------------
    bool GetNextCoefficient_rowmajor(coeff& c, int* ncoeffs, int* nbits) {
        AMM_error_invalid_arg(m_type != EnumStream::By_RowMajor, "Invalid stream type");
        const size_t xy = X * Y;
        if (m_arrangement == EnumArrangement::Original) {
            for(; m_idx < N;) {
                // wavelets are laid out in the spatial domain
                c.widx = m_idx;
                c.val = m_data[c.widx];
                if (fabs(c.val) <= m_threshold) {
                    m_idx++;
                    continue;
                }
                c.z = c.widx / xy;
                c.y = (c.widx % xy) / X;
                c.x = (c.widx % xy) % X;
                c.wtype = wavelets::get_wcoeff_lvl_type((itype)c.x,(itype)c.y,(itype)c.z, X, m_waveletMaxLevels, c.wlvl);
                ++(*ncoeffs);
                *nbits += sizeof(t) * CHAR_BIT;
                m_idx++;
                return true;
            }
            return false;
        }
        else { // subband arrangement
            std::cerr << "Subband arrangement is unsupported\n";
            exit(1);
        }
    }


    //! -------------------------------------------------------------------------------------
};
}
#endif
