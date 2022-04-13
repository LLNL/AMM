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
#ifndef AMM_BLOCK_PRECISION_H
#define AMM_BLOCK_PRECISION_H
//! ----------------------------------------------------------------------------

#include <cstdint>
#include <type_traits>
#include <iomanip>
#include <unordered_map>

#include "macros.hpp"
#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"
#include "containers/vec.hpp"
#include "containers/bitmask.hpp"
#include "containers/bytestream.hpp"
#include "containers/unordered_map.hpp"
#include "containers/unordered_set.hpp"
#include "utils/utils.hpp"
#include "precision/codec.hpp"
#include "precision/precision.hpp"
#include "tree/block_abstract.hpp"


//! ----------------------------------------------------------------------------
// 3 bits to precision (and remaining to exponent)
#define AMM_BLOCKP_PRECISION_NBITS 3
#define AMM_BLOCKP_EXPONENT_NBITS (2*CHAR_BIT - AMM_BLOCKP_PRECISION_NBITS)

//! ----------------------------------------------------------------------------
//#define AMM_DEBUG_BLOCKP_FULLVALS

#define AMM_BLOCKP_OFFSET 0
struct QuantInfo {

    TypePrecision m_precision : AMM_BLOCKP_PRECISION_NBITS;  // 3 bits can represent upto precision = 8
    TypeExponent m_exponent   : AMM_BLOCKP_EXPONENT_NBITS;   // 11 bits are needed to represent exponents of double

public:
    TypePrecision get_precision() const {   return m_precision;     }
    void set_precision(TypePrecision _) {   m_precision = _;        }

    TypeExponent get_exponent() const {     return m_exponent;      }
    void set_exponent(TypeExponent _) {     m_exponent = _;         }
};

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------

namespace amm {

//! ----------------------------------------------------------------------------
//! A block that uses mixed-precision uchar* to represent data
//! ----------------------------------------------------------------------------
template <typename _Float>
class block_precision : public amm::block<_Float> {

private:
    static_assert(std::is_floating_point<_Float>::value, "_Float is required to be a floating-point type!");

    using BaseBlock = amm::block<_Float>;
    using TypeMask  = amm::bitmask<AMM_BLOCK_NVERTS>;

    //TODO: should not have to use 64 bits here!
    using _Quant    = uint64_t; //typename TraitsInt<sizeof(_Float)>::unsigned_t;

    //! ------------------------------------------------------------------------
    amm::bytestream m_data;                 //! quantized bitstream
    TypeMask m_exists {0};                  //! bitmask to represent which values are stored

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
    _Float* m_fdata = nullptr;              //! actual bitstream of full doubles
#endif

    QuantInfo m_qinfo = {0, 0};             //! precision and exponent

public:
    block_precision() {}
    block_precision(const TypePrecision p) {

        AMM_error_invalid_arg(!AMM_is_valid_precision(p), "invalid precision (%d)!\n", p);
        precision(p);
    }
    ~block_precision() {}

    //! ------------------------------------------------------------------------
    //! query the underlying mask
    //! ------------------------------------------------------------------------
    inline size_t size() const {
        return m_exists.count();
    }

    // TODO: should properly clear these
    virtual inline void clear() {}

    inline bool contains(const TypeLocalIdx _) const {
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS, "invalid vertex %d\n", _);
        return m_exists[_];
    }

    inline float memory_in_kb() const {

        const size_t bytes = sizeof(TypeMask) + sizeof(QuantInfo) +
                             sizeof(amm::bytestream::wtype) * size();
        return float(bytes)/1024.0;

    }
    //! ------------------------------------------------------------------------
    //! get/set the precision and exponent
    //! ------------------------------------------------------------------------
    inline TypePrecision precision() const {    return 1 + m_qinfo.get_precision(); }
    inline void precision(TypePrecision _) {    return m_qinfo.set_precision(_-1);  }
    inline TypeExponent exponent() const {      return m_qinfo.get_exponent();      }
    inline void exponent(TypeExponent _) {      return m_qinfo.set_exponent(_);     }

    inline _Float get(const TypeLocalIdx _, bool &exists) const {
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS, "invalid vertex %d\n", _);
        exists = m_exists[_];
        return (m_exists[_]) ? _Float(_get(m_exists.count(_))) : AMM_missing_vertex(_Float);
    }

    //! ------------------------------------------------------------------------
    //! get/set values
    //! ------------------------------------------------------------------------
    inline
    _Float
    get(const TypeLocalIdx _) const {
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS, "invalid vertex %d\n", _);
        return (m_exists[_]) ? _Float(_get(m_exists.count(_))) : AMM_missing_vertex(_Float);
    }

    inline
    void
    set(const TypeLocalIdx _, const _Float v) {
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS, "invalid vertex %d\n", _);
        std::unordered_map<TypeLocalIdx, _Float> vals = {{_, v}};
        this->update(vals, false);
    }

    inline
    void
    add(const TypeLocalIdx _, const _Float v) {
        AMM_error_runtime(_ >= AMM_BLOCK_NVERTS, "invalid vertex %d\n", _);
        std::unordered_map<TypeLocalIdx, _Float> vals = {{_, v}};
        if (this->contains(_)) {
            vals[_] += this->get(_);
        }
        this->update(vals, false);
    }

    //! ------------------------------------------------------------------------
    //! update a block
    void
    update(const std::unordered_map<TypeLocalIdx, _Float>& mvals, bool add) {

        const size_t n_inputvals = mvals.size();

        // ignore empty blocks
        if (n_inputvals == 0)
            return;

        if (n_inputvals > AMM_BLOCK_NVERTS) {
            throw std::invalid_argument("BlockPrecision.update() received "+std::to_string(n_inputvals)+"values!\n");
        }

        // find the max index!
        TypeLocalIdx mx = 0;
        std::for_each(mvals.begin(), mvals.end(),
                      [&mx](auto _){ mx = std::max(mx, _.first); });

        if (mx >= AMM_BLOCK_NVERTS) {
            throw std::invalid_argument("BlockPrecision.update() received invalid vertex ("+std::to_string(n_inputvals)+")!\n");
        }

        // ---------------------------------------------------------------------
        // convert the map to vectors
        // TODO: fix this! shouldnt be creating vectors here
        std::vector<TypeLocalIdx> vidxs = {};
        std::vector<_Float> vvals = {};

        vidxs.reserve(mvals.size());
        vvals.reserve(mvals.size());

        for (auto iter = mvals.begin(); iter != mvals.end(); iter++) {

            const TypeIndex &i = iter->first;

            vidxs.push_back(i);
            if (add && this->contains(i)) { vvals.push_back(iter->second + this->get(i));   }
            else {                          vvals.push_back(iter->second);                  }
        }

        // ---------------------------------------------------------------------
        // mask for the vertices to be updated
        const TypeMask to_input {vidxs};
        //const TypeMask to_update = to_input & m_exists;
        const TypeMask to_insert = to_input.not_in(m_exists);

        const size_t n_existvals = m_exists.count();
        //const size_t n_updatevals = to_update.count();
        //const size_t n_insertvals = to_insert.count();
        const size_t n_finalvals = (to_input | m_exists).count();


        // ---------------------------------------------------------------------
        // create a list of updated values
        // to find new exponent and precision

        std::vector<_Float> updated_values (n_finalvals, _Float(0));

        // grab all the existing values
        for(TypeLocalIdx v = 0; v < AMM_BLOCK_NVERTS; v++) {
            if (m_exists[v]){
                const TypeLocalIdx i = m_exists.count(v);
                updated_values[i] = _get(i);
            }
        }

        // now, update the values
        TypeLocalIdx curr_end = n_existvals;
        for(TypeLocalIdx j = 0; j < n_inputvals; j++) {
            const TypeLocalIdx &v = vidxs[j];
            if (m_exists[v]) {  updated_values[m_exists.count(v)] = vvals[j];   }
            else {              updated_values[curr_end++] = vvals[j];          }
        }

        // ---------------------------------------------------------------------
        // find the new exponent and precision

        TypeExponent new_exp = amm::exp<_Float>(updated_values);
        new_exp = std::max(new_exp, exponent());

        const TypePrecision new_prec = amm::precisionf<_Float>(updated_values, new_exp);

        bool update_exponent = new_exp > exponent();
        bool update_precision = new_prec > precision();

        // ---------------------------------------------------------------------
        // reallocate the buffer if needed

        if (n_finalvals > n_existvals) {    _realloc(n_finalvals);              }
        if (update_precision) {             _realloc(n_finalvals, new_prec);    }

        // ---------------------------------------------------------------------
        // update the precision and exponent

        if (update_precision) {             _update_precision(new_prec);        }
        if (update_exponent) {              _update_exponent(new_exp);          }

        // ---------------------------------------------------------------------
        // now do the actual update
        // by now, precision and exponent have been updated!
        // ---------------------------------------------------------------------

        // if block is empty, initialize it
        if (m_exists.none()) {

            m_exists = to_input;
            for (size_t i = 0; i < n_inputvals; i++){
                _set(m_exists.count(vidxs[i]), vvals[i]);
            }
            _validate();
            return;
        }

        // ---------------------------------------------------------------------
        // if new vertices not needed, just update vertices
        if (to_insert.none()) {
            for (size_t i = 0; i < n_inputvals; i++) {
                _set(m_exists.count(vidxs[i]), vvals[i]);
            }
            _validate();
            return;
        }

        // ---------------------------------------------------------------------
        // otherwise, reallocate memory and shift values as necessary

        // a & b are indices in old buffer
        // c & d are indices in reallocated buffer)
        int b = n_existvals;    int a = b;
        int d = n_finalvals;    int c = d;

        // v is the index in the new data that needs to be added
        int v = n_inputvals;

        // in the first loop, simply insert invalids where needed!
        for(int bidx = AMM_BLOCK_NVERTS-1; bidx >= 0; --bidx) {

            // nothing to do for this bit!
            if (!m_exists[bidx] && !to_input[bidx]) {
                continue;
            }

            // insert a invalids for this bit!
            if (to_insert[bidx]) {

                // move [a,b] to [c,d], where b-a = d-c
                c = d + a - b;

                _move(a, b, d);
                _set(--c, vvals[--v]);

                b = a;
                d = c;
                continue;
            }

            // this bit does not need insertion
            --a;

            // this bit has a new value!
            if (to_input[bidx]) {
                _set(a, vvals[--v]);
            }
        }

        m_exists |= to_input;
        _validate();
    }


    //! ------------------------------------------------------------------------
    //! print the block
    //! ------------------------------------------------------------------------
    // TODO: should not use a print function. change into the ostream operator.
    void print(const int idx = -1) const {

        std::cout << " > printing ---- block (" << idx << ") at"
                  << " precision (" << int(precision()) << ")"
                  << " exponent = (" << int(exponent()) << ")"
                  << " exists = (" << m_exists << ")\n";

        size_t offset = 0;
        for(size_t i = 0; i < AMM_BLOCK_NVERTS; i++) {

            if (!m_exists[i]) {
                continue;
            }

            std::cout << "   > ["<<i<<"] = ";
            fflush(stdout);

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
            const _Float &v1 = m_fdata[offset];
            std::cout << "(( " << v1 << " )) ";

            //const _Quant &q2 = m_idata[offset];
            //const _Float &v2 = amm::__decode<_Float, _Quant>(q2, exponent());
            //          << "(( " << v2 << " : " << q2 << " [" << std::bitset<8*sizeof(_Quant)>(q2) << "] )) ";
            fflush(stdout);
#endif

            switch(precision()) {

            case 1:{    auto q3 = m_data.get<1>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 1>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 2:{    auto q3 = m_data.get<2>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 2>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 3: {   auto q3 = m_data.get<3>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 3>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 4:{    auto q3 = m_data.get<4>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 4>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 5:{    auto q3 = m_data.get<5>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 5>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 6:{    auto q3 = m_data.get<6>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 6>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 7:{    auto q3 = m_data.get<7>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 7>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            case 8:{    auto q3 = m_data.get<8>  (offset, AMM_BLOCKP_OFFSET);
                        _Float v3 = amm::decoden<_Float, _Quant, 8>(q3, exponent());
                        std::cout << " (( " << v3 << " : " << q3 << " [" << std::bitset<CHAR_BIT*sizeof(q3)>(q3) << "] )) ";
                        fflush(stdout);
                        break;
                    }
            default: std::cerr << "Block.print() -- Invalid precision " << precision() << std::endl;    exit(1);
            }
            std::cout << std::endl;
            offset++;
        }
    }

private:
    //! ------------------------------------------------------------------------
    //! Reallocate memory to adjust number of vertex value slots of a new
    //! precision level
    //! ------------------------------------------------------------------------
    inline void _realloc(const size_t sz, TypePrecision p=0) {

        BaseBlock::validate(sz-1, "BlockPrecision.realloc");

        if (p > 0) {
            //! Cannot allocate slots with invalid precisions
            AMM_error_invalid_arg(!AMM_is_valid_precision(p),
                                   "Block::realloc(): Invalid precision %d!\n", p);
        }
        else {  p = precision();    }

        bool is_first = (m_data.empty());
        m_data.resize(AMM_BLOCKP_OFFSET + p*sz);

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
        m_fdata = std::realloc(m_fdata, sizeof(_Float)*sz);
#endif
    }

    inline void _move(const TypeLocalIdx at, const TypeLocalIdx curr_end, const TypeLocalIdx new_end) {

        // TODO: figure out why this is happening!
        if (at == curr_end || curr_end == new_end) {
            //std::cerr << "move("<<at<<", " <<curr_end<<") --> (..., "<<new_end<<"): Trying to move zero values!\n";
            return;
        }

        if (at > curr_end || curr_end > new_end) {
            throw std::runtime_error("Invalid move( ("+std::to_string(at)+", "+std::to_string(curr_end)+") --> (..., "+std::to_string(new_end)+" ) requested!\n");
        }

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
        std::move_backward(m_fdata + at, m_fdata + curr_end, m_fdata + new_end);
#endif
        switch(precision()) {
        case 1: {   m_data.move<1> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 2: {   m_data.move<2> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 3: {   m_data.move<3> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 4: {   m_data.move<4> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 5: {   m_data.move<5> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 6: {   m_data.move<6> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 7: {   m_data.move<7> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        case 8: {   m_data.move<8> (at, curr_end, new_end, AMM_BLOCKP_OFFSET);  break;  }
        }
    }

    //! ------------------------------------------------------------------------
    inline _Float _get(const TypeLocalIdx i) const {

        _Float val_p = 0;
        switch(precision()) {
        case 1: {   val_p = amm::decoden<_Float, _Quant, 1>(m_data.get<1> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 2: {   val_p = amm::decoden<_Float, _Quant, 2>(m_data.get<2> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 3: {   val_p = amm::decoden<_Float, _Quant, 3>(m_data.get<3> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 4: {   val_p = amm::decoden<_Float, _Quant, 4>(m_data.get<4> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 5: {   val_p = amm::decoden<_Float, _Quant, 5>(m_data.get<5> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 6: {   val_p = amm::decoden<_Float, _Quant, 6>(m_data.get<6> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 7: {   val_p = amm::decoden<_Float, _Quant, 7>(m_data.get<7> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        case 8: {   val_p = amm::decoden<_Float, _Quant, 8>(m_data.get<8> (i, AMM_BLOCKP_OFFSET), exponent());   break; }
        }
        return val_p;
    }

    inline void _set(const TypeLocalIdx i, const _Float fv) {

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
        m_fdata[i]  = fv;
#endif
        switch(precision()) {
        case 1: { m_data.set<1> (i, amm::encoden<_Float, _Quant, 1>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 2: { m_data.set<2> (i, amm::encoden<_Float, _Quant, 2>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 3: { m_data.set<3> (i, amm::encoden<_Float, _Quant, 3>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 4: { m_data.set<4> (i, amm::encoden<_Float, _Quant, 4>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 5: { m_data.set<5> (i, amm::encoden<_Float, _Quant, 5>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 6: { m_data.set<6> (i, amm::encoden<_Float, _Quant, 6>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 7: { m_data.set<7> (i, amm::encoden<_Float, _Quant, 7>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 8: { m_data.set<8> (i, amm::encoden<_Float, _Quant, 8>(fv, exponent()), AMM_BLOCKP_OFFSET);   break;   }
        }
    }

    inline void _add(const TypeLocalIdx i, const _Float fv) {

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
        m_fdata[i]  += fv;
#endif
        //TODO: i should be able to directly add the quantizated values!
        switch(precision()) {
        case 1: { m_data.set<1> (i, amm::encoden<_Float, _Quant, 1>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 2: { m_data.set<2> (i, amm::encoden<_Float, _Quant, 2>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 3: { m_data.set<3> (i, amm::encoden<_Float, _Quant, 3>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 4: { m_data.set<4> (i, amm::encoden<_Float, _Quant, 4>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 5: { m_data.set<5> (i, amm::encoden<_Float, _Quant, 5>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 6: { m_data.set<6> (i, amm::encoden<_Float, _Quant, 6>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 7: { m_data.set<7> (i, amm::encoden<_Float, _Quant, 7>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        case 8: { m_data.set<8> (i, amm::encoden<_Float, _Quant, 8>(fv + _get(i), exponent()), AMM_BLOCKP_OFFSET);   break;   }
        }

        /*
        switch(precision()) {
        case 1: { m_data.add<1> (i, amm::encoden<_Float, _Quant, 1>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 2: { m_data.add<2> (i, amm::encoden<_Float, _Quant, 2>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 3: { m_data.add<3> (i, amm::encoden<_Float, _Quant, 3>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 4: { m_data.add<4> (i, amm::encoden<_Float, _Quant, 4>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 5: { m_data.add<5> (i, amm::encoden<_Float, _Quant, 5>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 6: { m_data.add<6> (i, amm::encoden<_Float, _Quant, 6>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 7: { m_data.add<7> (i, amm::encoden<_Float, _Quant, 7>(fv, exponent()), BLOCK_OFFSET);   break;   }
        case 8: { m_data.add<8> (i, amm::encoden<_Float, _Quant, 8>(fv, exponent()), BLOCK_OFFSET);   break;   }
        }*/
    }

    //! ------------------------------------------------------------------------
    inline void _update_exponent(const TypeExponent new_exp) {

        const TypeExponent old_exp = exponent();
        if (new_exp == old_exp) return;
        exponent(new_exp);

        const size_t sz = m_exists.count();
        for (size_t i = 0; i < sz; i++) {

            _Quant qv = 0;
            switch(precision()) {
            case 1: {   qv = amm::inflate<1, sizeof(_Quant)>(m_data.get<1> (i, AMM_BLOCKP_OFFSET));   break; }
            case 2: {   qv = amm::inflate<2, sizeof(_Quant)>(m_data.get<2> (i, AMM_BLOCKP_OFFSET));   break; }
            case 3: {   qv = amm::inflate<3, sizeof(_Quant)>(m_data.get<3> (i, AMM_BLOCKP_OFFSET));   break; }
            case 4: {   qv = amm::inflate<4, sizeof(_Quant)>(m_data.get<4> (i, AMM_BLOCKP_OFFSET));   break; }
            case 5: {   qv = amm::inflate<5, sizeof(_Quant)>(m_data.get<5> (i, AMM_BLOCKP_OFFSET));   break; }
            case 6: {   qv = amm::inflate<6, sizeof(_Quant)>(m_data.get<6> (i, AMM_BLOCKP_OFFSET));   break; }
            case 7: {   qv = amm::inflate<7, sizeof(_Quant)>(m_data.get<7> (i, AMM_BLOCKP_OFFSET));   break; }
            case 8: {   qv = amm::inflate<8, sizeof(_Quant)>(m_data.get<8> (i, AMM_BLOCKP_OFFSET));   break; }
            }

            _set(i, amm::decode<_Float, _Quant>(qv, old_exp));
        }
    }

    //! ------------------------------------------------------------------------
    template <uint8_t Nlow, uint8_t Nhigh>
    inline void _increase_precision() {

        static_assert(AMM_is_valid_precision(Nlow), "Invalid Nlow!");
        static_assert(AMM_is_valid_precision(Nhigh), "Invalid Nhigh!");
        static_assert(Nlow < Nhigh, "Nlow should be smaller than Nhigh!");

        // update precision (Nin --> Nhigh) of all existing values
        const size_t sz = m_exists.count();
        for (int i = sz-1; i >= 0; i--) {
            m_data.set<Nhigh>(i, amm::inflate<Nlow, Nhigh>(m_data.get<Nlow>(i, AMM_BLOCKP_OFFSET)), AMM_BLOCKP_OFFSET);
        }
    }

    inline void _update_precision(const TypePrecision new_prec) {

        if (!AMM_is_valid_precision(new_prec)) {
            std::cerr << "_update_precision("<<int(new_prec)<<" : invalid precision!\n";
            exit(1);
        }

        const TypePrecision old_prec = precision();

        if (new_prec <= old_prec)   return;
        precision(new_prec);

        // new precision is greater than old precision
        switch(old_prec) {

            case 1:
            {
                switch(new_prec) {
                    case 2: return _increase_precision<1, 2>();
                    case 3: return _increase_precision<1, 3>();
                    case 4: return _increase_precision<1, 4>();
                    case 5: return _increase_precision<1, 5>();
                    case 6: return _increase_precision<1, 6>();
                    case 7: return _increase_precision<1, 7>();
                    case 8: return _increase_precision<1, 8>();
                }
                break;
            }
            case 2:
            {
                switch(new_prec) {
                    case 3: return _increase_precision<2, 3>();
                    case 4: return _increase_precision<2, 4>();
                    case 5: return _increase_precision<2, 5>();
                    case 6: return _increase_precision<2, 6>();
                    case 7: return _increase_precision<2, 7>();
                    case 8: return _increase_precision<2, 8>();
                }
                break;
            }
            case 3:
            {
                switch(new_prec) {
                    case 4: return _increase_precision<3, 4>();
                    case 5: return _increase_precision<3, 5>();
                    case 6: return _increase_precision<3, 6>();
                    case 7: return _increase_precision<3, 7>();
                    case 8: return _increase_precision<3, 8>();
                }
                break;
            }
            case 4:
            {
                switch(new_prec) {
                    case 5: return _increase_precision<4, 5>();
                    case 6: return _increase_precision<4, 6>();
                    case 7: return _increase_precision<4, 7>();
                    case 8: return _increase_precision<4, 8>();
                }
                break;
            }
            case 5:
            {
                switch(new_prec) {
                    case 6: return _increase_precision<5, 6>();
                    case 7: return _increase_precision<5, 7>();
                    case 8: return _increase_precision<5, 8>();
                }
                break;
            }
            case 6:
            {
                switch(new_prec) {
                    case 7: return _increase_precision<6, 7>();
                    case 8: return _increase_precision<6, 8>();
                }
                break;
            }
            case 7:
            {
                switch(new_prec) {
                    case 8: return _increase_precision<7, 8>();
                }
                break;
            }
        }

        std::cerr << "_update_precision ("<<int(old_prec) << " ---> " << int(new_prec) <<") : invalid request!\n";
        exit(1);
    }

    //! ------------------------------------------------------------------------
    //! Compare all double data values against dequantized values to verify
    //! representation correctness
    //! ------------------------------------------------------------------------
    void _validate(const bool verbose = false) const {

#ifdef AMM_DEBUG_BLOCKP_FULLVALS
        if (verbose)
            printf(LOG::ERROR, "\nValidating Block: \n");

        for (TypeLocalIdx i = 0; i < AMM_BLOCK_NVERTS; i++) {
            if (!m_exists[i])   continue;

                const _Float v1 = _get(m_exists.count(i));
                const _Float v2 = m_fdata[m_exists.count(i)];

                if (!AMM_is_zero(v1-v2)) {
                    std::cerr << " BlockPrecision::validate(): mismatch in vertex " << i <<
                                std::fixed << std::setprecision (std::numeric_limits<_Float>::digits10 + 1) <<
                                 " : full-res " << v2 <<
                                 " : block " << v1 <<
                                 " : diff = " << fabs(v1-v2) << std::endl;
                    std::cerr << " BlockPrecision::validate(): mismatch detected in vertex " << i << "!\n";
                    this->print();
                    exit(1);
                }
        }

        if (verbose)
            printf(LOG::ERROR, "Done\n");
#endif
    }

public:
    //! ------------------------------------------------------------------------
    //! const iterator
    //! ------------------------------------------------------------------------
    class const_iterator {

    public:
        typedef std::pair<TypeLocalIdx, _Float> value_type;

    private:

        typedef const value_type& reference;
        typedef const value_type* pointer;
        typedef const_iterator self_type;
        typedef int difference_type;
        typedef std::forward_iterator_tag iterator_category;

        value_type _vertex;
        const block_precision *_block;

    public:
        const_iterator() :
            _vertex(std::make_pair(AMM_BLOCK_NVERTS, 0)),
            _block(nullptr)
        {}

        const_iterator(const block_precision &block, const bool &is_begin) :
                _vertex(std::make_pair(AMM_BLOCK_NVERTS, 0)),
                _block(&block) {

            if (is_begin) {
                TypeLocalIdx vidx = _block->m_exists.first();
                if (vidx < AMM_BLOCK_NVERTS) {
                    _vertex = std::make_pair(vidx, _block->get(vidx));
                }
            }
        }

        self_type operator++()      { self_type i = *this; this->operator ++(0); return i;  }
        self_type operator++(int) {

            TypeLocalIdx vidx = _block->m_exists.next(_vertex.first);
            if (vidx < AMM_BLOCK_NVERTS) {  _vertex = std::make_pair(vidx, _block->get(vidx));  }
            else {                          _vertex = std::make_pair(AMM_BLOCK_NVERTS, 0);          }
            return *this;
        }

        reference operator*() const             { return _vertex;       }
        pointer operator->() const              { return &(_vertex);    }
        bool operator==(const self_type& rhs) {
            return _block == rhs._block && _vertex.first == rhs._vertex.first;
        }
        bool operator!=(const self_type& rhs) {
            return _block != rhs._block || _vertex.first != rhs._vertex.first;
        }

        void operator=(const self_type& rhs) {
            this->_vertex = std::make_pair(rhs._vertex.first, rhs._vertex.second);
            this->_block = rhs._block;
        }
    };

    //! ------------------------------------------------------------------------
    const_iterator begin() const    { return const_iterator(*this, true);   }
    const_iterator end() const      { return const_iterator(*this, false);  }
};

//! ------------------------------------------------------------------------
}   // end of namespace
#endif
