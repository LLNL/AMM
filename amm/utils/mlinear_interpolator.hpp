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
#ifndef AMM_MULTILINEAR_INTERPOLATOR_H
#define AMM_MULTILINEAR_INTERPOLATOR_H

//! ----------------------------------------------------------------------------
#include <vector>
#include <sstream>
#include <stdexcept>

#include "types/dtypes.hpp"
#include "utils/utils.hpp"
#include "containers/vec.hpp"
#include "macros.hpp"

//! ------------------------------------------------------------------------
#define AMM_MLINTERP_invalid_val(T) AMM_nan(T)

namespace amm {

//! ------------------------------------------------------------------------
//! static multilinear interpolation
template<typename TypeValue, typename TypeLambda>
static inline TypeValue lerp (const TypeLambda l, const TypeValue f0, const TypeValue f1) {
    static constexpr TypeLambda o = 1.0;
    return (o-l)*f0 + l*f1;
}
template<typename TypeValue, typename TypeLambda>
static inline TypeValue lerp2(const TypeLambda lx, const TypeLambda ly,
                              const TypeValue f00, const TypeValue f01, const TypeValue f10, const TypeValue f11) {
    return lerp(ly, lerp(lx, f00, f01),
                    lerp(lx, f10, f11));
}
template<typename TypeValue, typename TypeLambda>
static inline TypeValue lerp3(const TypeLambda lx, const TypeLambda ly, const TypeLambda lz,
                              const TypeValue f000, const TypeValue f010, const TypeValue f100, const TypeValue f110,
                              const TypeValue f001, const TypeValue f011, const TypeValue f101, const TypeValue f111) {
    return lerp(lz, lerp2(lx, ly, f000, f010, f100, f110),
                    lerp2(lx, ly, f001, f011, f101, f111));
}


//! ----------------------------------------------------------------------------
template<TypeDim Dim, typename TypeValue>
class MultilinearInterpolator {

    static_assert((Dim == 1 || Dim == 2 || Dim == 3), "MultilinearInterpolator works for 1D, 2D, and 3D only!");

    using TypeVertex = Vec<Dim, TypeCoord>;

    // store values and gradients in double
    // to perform all calculations in double
    using typel = double;
    std::vector<TypeValue> m_cvals;
    typel m_gradient[Dim];

    static constexpr typel o = 1.0;
    static constexpr TypeCornerId snCorners = AMM_pow2(Dim);
    std::pair<TypeCoord, TypeCoord> m_bounds[Dim];

public:

    //! ------------------------------------------------------------------------
    //! default constructor
    MultilinearInterpolator() {}

    //! define only using bounds (values to supplied when needed)
    MultilinearInterpolator(const TypeVertex &bbox0, const TypeVertex &bbox1) {

        for(TypeDim d = 0; d < Dim; d++) {
            m_bounds[d].first = bbox0[d];
            m_bounds[d].second = bbox1[d];
            m_gradient[d] = o / typel(bbox1[d] - bbox0[d]);
        }
    }

    //! define using bounds and corner values (in row-major order)
    MultilinearInterpolator(const TypeVertex &bbox0, const TypeVertex &bbox1, const std::vector<TypeValue> &cornerValues) {

        AMM_error_invalid_arg(snCorners != cornerValues.size(),
                               "MultilinearInterpolator<D=%d> requires %d corner values, but got %d!\n",
                               int(Dim), int(snCorners), cornerValues.size());

        for(TypeDim d = 0; d < Dim; d++) {
            m_bounds[d].first = bbox0[d];
            m_bounds[d].second = bbox1[d];
            m_gradient[d] = o / typel(bbox1[d] - bbox0[d]);
        }
        m_cvals = cornerValues;

        AMM_replace_missing_with_zeros(TypeValue, m_cvals);
    }

    //! ------------------------------------------------------------------------
    //! compute the multilinear interpolation at a point 'p'
    //! test for point "inside" if requested
    //! ------------------------------------------------------------------------
    TypeValue compute(const TypeVertex &_, const std::vector<TypeValue> &cornerValues, bool inside_only = false) const {

        AMM_error_logic(snCorners != cornerValues.size(),
                                                 "MultilinearInterpolator<D=%d>.compute() requires %d corner values, but got %d!\n",
                                                 int(Dim), int(snCorners), cornerValues.size());

        // return invalids if this point lies outside
        if (inside_only) {
            for(TypeDim d = 0; d < Dim; d++){
                if (_[d] < m_bounds[d].first || _[d] > m_bounds[d].second)
                    return AMM_MLINTERP_invalid_val(TypeValue);
            }
        }

        static Vec<Dim, typel> lambda;
        for(TypeDim d = 0; d < Dim; d++) {
            lambda[d] = typel(_[d] - m_bounds[d].first) * m_gradient[d];
        }

        TypeValue v = 0;
        switch (Dim) {

        case 1:    {
                    const typel &x = lambda[0];
                    v = (o-x)*typel(cornerValues[0]) + x*typel(cornerValues[1]);
                   }
                   break;

        case 2:    {
                    const typel &x = lambda[0];
                    const typel &y = lambda[1];
                    const typel ox = (o-x);

                    v = (o-y)*(ox*typel(cornerValues[0]) + x*typel(cornerValues[1])) +
                            y*(ox*typel(cornerValues[2]) + x*typel(cornerValues[3]));
                   }
                   break;

        case 3:   {
                    const typel &x = lambda[0];
                    const typel &y = lambda[1];
                    const typel &z = lambda[2];
                    const typel ox = (o-x);
                    const typel oy = (o-y);

                    v = (o-z)*(oy*(ox*typel(cornerValues[0]) + x*typel(cornerValues[1])) +
                                y*(ox*typel(cornerValues[2]) + x*typel(cornerValues[3]))) +
                            z*(oy*(ox*typel(cornerValues[4]) + x*typel(cornerValues[5])) +
                                y*(ox*typel(cornerValues[6]) + x*typel(cornerValues[7])));
                   }
                   break;
        }
        return v;
    }

    //! compute the multilinear interpolation at a point 'p'
    //! test for point "inside" if requested
    TypeValue compute(const TypeVertex &_, bool inside_only = false) const {

        AMM_error_logic(m_cvals.empty(), "MultilinearInterpolator does not have corner values!\n");
        return this->compute(_, m_cvals, inside_only);
    }

    TypeValue compute_using_lerp(const TypeVertex &p, bool inside_only = false) const {

        AMM_error_logic(m_cvals.empty(), "MultilinearInterpolator does not have corner values!\n");

        // return invalids if this point lies outside
        if (inside_only) {
            for(TypeDim d = 0; d < Dim; d++){
                if (p[d] < m_bounds[d].first || p[d] > m_bounds[d].second)
                    return AMM_MLINTERP_invalid_val(TypeValue);
            }
        }

        static Vec<Dim, typel> lambda;
        for(TypeDim d = 0; d < Dim; d++) {
            lambda[d] = typel(p[d] - m_bounds[d].first) * m_gradient[d];
        }

        TypeValue v = 0;
        switch (Dim) {
            case 1:     v = lerp (lambda[0],
                                  m_cvals[0], m_cvals[1]);    break;

            case 2:     v = lerp2(lambda[0], lambda[1],
                                  m_cvals[0], m_cvals[1], m_cvals[2], m_cvals[3]);    break;

            case 3:     v = lerp3(lambda[0], lambda[1], lambda[2],
                                  m_cvals[0], m_cvals[1], m_cvals[2], m_cvals[3],
                                  m_cvals[4], m_cvals[5], m_cvals[6], m_cvals[7] );   break;
        }

        return v;
    }
};
//! ----------------------------------------------------------------------------

}   // end of namespace
#endif
