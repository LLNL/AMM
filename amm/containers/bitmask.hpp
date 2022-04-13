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
#ifndef AMM_BITMASK_H
#define AMM_BITMASK_H

//! ----------------------------------------------------------------------------
#include <bitset>
#include <cassert>
#include <climits>
#include <vector>
#include <ostream>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

#include "macros.hpp"
#include "types/byte_traits.hpp"
#include "utils/utils.hpp"

//! ----------------------------------------------------------------------------
// max number of bits supported by this container
#define AMM_BM_MAX_NBITS            AMM_MAX_NBITS
#define AMM_BM_is_valid_nbits(_)    AMM_is_valid_nbits(_)

// create a bitmask of n ones
#define AMM_BM_create_mask(T,n) (static_cast<T>(-((n)!=0)))&(static_cast<T>(-1)>>((sizeof(T)*CHAR_BIT)-(n)))

namespace amm {

//! ----------------------------------------------------------------------------
//! A container to manage a bitmask
//! ----------------------------------------------------------------------------
template<uint8_t NBITS>
class bitmask {

    static_assert (AMM_BM_is_valid_nbits(NBITS), "bitmask currently works for upto 64 bits only!");

public:
    // datatype of the word that is being stored
    using word_type = typename traits_nbits<NBITS>::unsigned_t;

private:
    // datatype to represent the number of bits (8 bit type can represent 256 bits)
    using size_type = uint8_t;

    // self type
    using self_type = bitmask<NBITS>;

    // static constants to support bit manipulations
    static const word_type _one = 1;
    static const word_type _full = AMM_BM_create_mask(word_type, NBITS);

    // actual data is stored as a single word
    word_type m_data;

    inline void _mask() {                      m_data &= _full;         }
    inline void _set(const size_type &_) {     m_data |= _one<<_;       }
    inline void _unset(const size_type &_) {   m_data &=  ~(_one<<_);   }

public:
    //! ------------------------------------------------------------------------
    //! initialization methods
    //! ------------------------------------------------------------------------
    bitmask() : m_data(0) {}
    bitmask(const word_type _) : m_data(_&_full) {}


    template <typename T>
    bitmask(const std::vector<T> &_) : m_data(0){
        static_assert (std::is_unsigned<T>::value, "BitMask<T>() reads vectors of unsigned types!");
        init_vec(_);
    }


    //! initialize using the value
    inline
    void
    init_val(const word_type _) {
        m_data = _&_full;
    }


    //! initialize using a vector
    template <typename T>
    inline
    void
    init_vec(const std::vector<T> &_){

        static_assert (std::is_unsigned<T>::value, "BitMask::from_vector() reads vectors of unsigned types!");
        assert(*std::max_element(_.begin(), _.end()) < NBITS);

        m_data = 0;
        for(auto i = _.begin(); i != _.end(); ++i)  _set(*i);
        _mask();
    }

    //! initialize using an initializer list
        //! this is a convenience function, so not templating it
        //! because c++ numeric literals are integers by default
    inline
    void
    init_list(std::initializer_list<int> _) {
        m_data = create_word(_);
    }

    //! ------------------------------------------------------------------------
    //! query functionality on the bitmask
    //!     all query functions work directly on raw m_data
    //!     assuming that any invalid bits will be 0
    //! ------------------------------------------------------------------------
    inline bool any() const {           return m_data != 0;     }
    inline bool none() const {          return m_data == 0;     }
    inline bool all() const {           return m_data == _full; }
    inline word_type value() const {    return m_data;          }


    //! count the number of set bits in the data
    inline
    size_type
    count() const {
        return utils::bcount_ones(m_data);
    }


    //! count the number of set bits that come before a given bit
    inline
    size_type
    count(const size_type &_) const {
        assert(_ < NBITS);
        return utils::bcount_ones(m_data & ((_one<<_)-1));
    }


    //! count the number of set bits that come between two bits
    inline
    size_type
    count(const size_type &a, const size_type &b) const {
        assert(a < NBITS && b < NBITS && a < b);
        return static_cast<size_type>(utils::bcount_ones(b)-utils::bcount_ones(a));
    }


    inline
    size_type
    first() const {
        return (m_data == 0) ? AMM_BM_MAX_NBITS : utils::bcount_tz<word_type>(m_data);
    }


    inline
    size_type
    next(size_type _) const {
        if (_ == NBITS-1) return AMM_BM_MAX_NBITS;
        const word_type masked = m_data & (bitmask::_full << (_+1));
        return (masked == 0) ? AMM_BM_MAX_NBITS : utils::bcount_tz<word_type>(masked);
    }



    //! ------------------------------------------------------------------------
    //! modification functionality on the bitmask
    //!     setters are responsible to work on valid bits only
    //! ------------------------------------------------------------------------
    inline void set() {     m_data = _full; }
    inline void reset() {   m_data = 0; }


    inline
    void
    flip() {
        m_data = ~m_data;
        _mask();
    }


    inline
    void
    set(const size_type &_) {
        assert(_ < NBITS);
        _set(_);
        _mask();
    }


    inline
    void
    unset(const size_type &_) {
        assert(_ < NBITS);
        _unset(_);
        _mask();
    }


    inline
    void
    flip(const size_type &_) {
        assert(_ < NBITS);
        (*this)[_] ? _unset(_) : _set(_);
        _mask();
    }


    //! ------------------------------------------------------------------------
    //! comparison with other bitmasks
    //! ------------------------------------------------------------------------

    //! get the bits that are set in (this) but not set in (_)
    inline
    self_type
    not_in(const self_type &_) const {
        return self_type(m_data & ~_.m_data);
    }

    //! get a list of bits that are set
    template <typename T>
    inline
    std::vector<T>
    as_vector() const {
        static_assert (std::is_unsigned<T>::value, "BitMask::to_vector() returns vectors of unsigned types!");
        std::vector<T> _;
        _.reserve(this->count());
        for(size_type i = 0; i < NBITS; i++) {
            if ((*this)[i]) _.push_back(i);
        }
        return _;
    }


    //! ------------------------------------------------------------------------
    //! static methods
    //! ------------------------------------------------------------------------
    inline static size_type size() {    return sizeof(word_type);   }
    inline static bitmask get_full() {  return bitmask(_full);      }


    inline static
    bitmask
    create_mask(std::initializer_list<int> _) {
        return bitmask(create_word(_));
    }

    //! populate the underlying word using an initilizer list
    inline static
    word_type
    create_word(std::initializer_list<int> _) {

        assert(*std::max_element(_.begin(), _.end()) < NBITS);
        assert(*std::min_element(_.begin(), _.end()) >= 0);

        // process the initializer list
        word_type rval {0};
        for (int i: _)  rval ^= (_one<<i);
        return rval & _full;
    }



    //! ------------------------------------------------------------------------
    //! operators
    //! ------------------------------------------------------------------------
    //! access operator
    inline
    bool
    operator[](const size_type &_) const {
        assert(_ < NBITS);
        return (m_data & (_one<<_)) != 0;
    }

    //! comparison operators
    inline bool operator==(const self_type& _) const { return m_data == _.m_data; }
    inline bool operator<(const self_type& _) const {  return m_data < _.m_data;  }

    //! bitwise operators
    inline
    self_type&
    operator&=(const self_type& _) {
        m_data &= _.m_data;
        _mask();
        return *this;
    }


    inline
    self_type&
    operator|=(const self_type& _) {
        m_data |= _.m_data;
        _mask();
        return *this;
    }


    inline
    self_type&
    operator^=(const self_type& _) {
        m_data ^= _.m_data;
        _mask();
        return *this;
    }


    inline
    self_type&
    operator<<=(const size_type& _) {
        assert(_ < NBITS);
        m_data <<= _;
        _mask();
        return *this;
    }


    inline
    self_type&
    operator>>=(const size_type& _) {
        assert(_ < NBITS);
        m_data >>= _;
        _mask();
        return *this;
    }


    //! ------------------------------------------------------------------------
    //! friend operators
    //! ------------------------------------------------------------------------
    template<uint8_t N>
    friend inline bitmask<N> operator&(const bitmask<N> &a, const bitmask<N> &b);


    template<uint8_t N>
    friend inline bitmask<N> operator|(const bitmask<N> &a, const bitmask<N> &b);


    template<uint8_t N>
    friend inline bitmask<N> operator^(const bitmask<N> &a, const bitmask<N> &b);


    template<uint8_t N>
    friend inline bitmask<N> operator~(const bitmask<N> &a);


    template<uint8_t N>
    friend inline std::ostream& operator<<(std::ostream &os, const bitmask<N> &_);

    //! ------------------------------------------------------------------------
    //! bitmask operator
    //! ------------------------------------------------------------------------
#if 0
    // @TODO: this is not used anywhere, and is also not tested!
    class const_iterator {

        typedef size_type value_type;

        const bitmask &data;
        value_type i;

    public:

        typedef const value_type& reference;
        typedef const value_type* pointer;
        typedef const_iterator self_type;
        typedef int difference_type;
        typedef std::forward_iterator_tag iterator_category;

        const_iterator(){}
        const_iterator(const bitmask &_, const bool &is_begin) : data(_) {
            i = (is_begin) ? data.first() : AMM_BM_MAX_NBITS;
        }

        self_type operator++()      { self_type k = *this; this->operator ++(0); return k;  }
        self_type operator++(int) {   i = data.next(i);   return *this;                     }

        reference operator*() const             { return i;       }
        pointer operator->() const              { return &i;      }
        bool operator==(const self_type& rhs)   { return this->data == rhs.data && this->i == rhs.i;  }
        bool operator!=(const self_type& rhs)   { return this->data != rhs.data || this->i != rhs.i;  }
        void operator=(const self_type& rhs) {    this->data = rhs.data;  this->i = rhs.i;            }
    };

    const_iterator begin() const    { return const_iterator(*this, true);   }
    const_iterator end() const      { return const_iterator(*this, false);  }
#endif
};


//! ----------------------------------------------------------------------------
//! friend operators
//! ----------------------------------------------------------------------------
template<uint8_t N>
static inline
bitmask<N>
operator&(const bitmask<N> &a, const bitmask<N> &b) {
    return bitmask<N> (a.m_data & b.m_data);
}


template<uint8_t N>
static inline
bitmask<N>
operator|(const bitmask<N> &a, const bitmask<N> &b) {
    return bitmask<N> (a.m_data | b.m_data);
}


template<uint8_t N>
static inline
bitmask<N>
operator^(const bitmask<N> &a, const bitmask<N> &b) {
    return bitmask<N> (a.m_data ^ b.m_data);
}


template<uint8_t N>
static inline
bitmask<N>
operator~(const bitmask<N> &a) {
    return bitmask<N> (~a.m_data);
}


template<uint8_t N>
static inline
std::ostream&
operator<<(std::ostream &os, const bitmask<N> &_) {
    return os << std::size_t(_.m_data) << " [" << std::bitset<N>(_.m_data) << "]";
}

//! ----------------------------------------------------------------------------
} // end of namespace
#endif
