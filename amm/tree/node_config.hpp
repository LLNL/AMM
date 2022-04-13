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
#ifndef AMM_NODECONFIG_H
#define AMM_NODECONFIG_H

#include <set>
#include <vector>
#include <fstream>
#include <unordered_set>

#include "types/dtypes.hpp"
#include "types/enums.hpp"
#include "utils/utils.hpp"
#include "utils/logger.hpp"
#include "utils/exceptions.hpp"
#include "containers/bitmask.hpp"

#include "node_config_core.hpp"

namespace amm {


/// ------------------------------------------------------------------------------------
/// a utility structure to manipulate child flags
/// the "OctreeNodeConfiguration" responds to user queries
/// it uses "OctreeNodeCoreConfiguration" to create static collections for fast response
/// ------------------------------------------------------------------------------------

template <TypeDim Dim>
struct OctreeNodeConfiguration : private OctreeNodeCoreConfiguration<Dim> {

    static_assert((Dim == 2 || Dim == 3), "OctreeNodeConfiguration works for 2D and 3D only!");

    using CoreConfig = OctreeNodeCoreConfiguration<Dim>;

    using TypeConfig = typename CoreConfig::TypeConfig;
    using TypeChildMask = typename CoreConfig::TypeChildMask;
    using TypeChildFlag = typename CoreConfig::TypeChildFlag;
    using TypeOccupancyMask = typename CoreConfig::TypeOccupancyMask;

    using VecOccMask = typename CoreConfig::VecOccMask;
    using VecChildMask = typename CoreConfig::VecChildMask;
    using VecVecChildMask = typename CoreConfig::VecVecChildMask;

    using TypeConfigSet = typename std::set<TypeChildFlag>;
    using TypeMappingMap = typename std::unordered_map<TypeChildFlag, TypeChildFlag>;

public:

    //! number of possible types of axes
    static constexpr EnumAxes snAxes = CoreConfig::snAxes;

    //! number of possible types of standard and non-standard children
    static const  TypeChildId snChildren = CoreConfig::snChildren;
    static const  TypeChildId snChildren_nonstandard = CoreConfig::snChildren_nonstandard;

    //! config id for a fully refined node
    static const TypeChildFlag sCflags_full = CoreConfig::sCflags_full;


    /// --------------------------------------------------------------------------------
    //! query for empty and full childflags
    static inline TypeChildFlag full() {                        return sCflags_full;                }
    static inline bool is_full(const TypeChildMask &_) {        return _.value() == sCflags_full;   }


    //! query for a particular child
    static inline bool child_exists(const TypeChildMask &_, const  TypeChildId &child_id) {
        return _[child_id];
    }


    //! check if a node has vacuum (using occupancy flags)
    static inline bool has_vacuum(const TypeChildMask &_) {
        return !get_occupancy_flags(_).all();
    }

    //! check if a node is valid (using occupancy flags)
    static inline bool is_valid_node(const TypeChildMask &_) {

        std::vector<uint8_t> occupancy_counts(snChildren, 0);
        for( TypeChildId c = 0; c < snChildren_nonstandard; c++) {

            if (!_[c]){
                continue;
            }

            const TypeOccupancyMask &co = get_occupancy_of_child(c);
            for( TypeChildId k = 0; k < snChildren; k++) {
                if (co[k]) {
                    occupancy_counts[k]++;
                }
            }
        }

        bool is_invalid = false;
        for(uint8_t i=0; i < snChildren; i++){
            if (occupancy_counts[i] > 1){  is_invalid = true;  break;  }
        }


        if (is_invalid) {
            AMM_log_error << "invalid occupancy: [";
            for(int i=8-1; i >= 0; i--)
                AMM_log_error  << " " << int(occupancy_counts[i]) ;
            AMM_log_error << " ] -- for node";
            print(_);
            return false;
        }

        return true;
    }


    //! get the type of the child node (0: square, 1: rect in 1 dim, 2: rect in 2 dims)
    static inline uint8_t node_type(const  TypeChildId &child_id) {
        static const uint8_t node_types[7] = {0,1,1,2,1,2,2};
        return node_types[size_t(get_long_axes(child_id))];
    }


    /// --------------------------------------------------------------------------------
    //! based on the child_id, return the axes in which the child is long
    static inline
    EnumAxes
    get_long_axes(const  TypeChildId &_) {

        AMM_error_invalid_arg(_ > snChildren_nonstandard,
                               "OctreeChildNodes<%d>::get_long_axes(%d) got invalid child_id!\n",
                               int(Dim), int(_));

        static std::vector<EnumAxes> data = CoreConfig::long_axes();
        return data[_];
    }


    //! based on the child_id, return the axis that the parent should be split at
    static inline
    EnumAxes
    get_split_axes(const  TypeChildId &_) {

        AMM_error_invalid_arg(_ > snChildren_nonstandard,
                               "OctreeChildNodes<%d>::get_split_axes(%d) got invalid child_id!\n",
                               int(Dim), int(_));

        static std::vector<EnumAxes> data = CoreConfig::split_axes();
        return data[_];
    }


    //! based on the child_id, return a bitmap for what axes needs to be scaled and shifted
    static inline
    TypeDim
    get_bounds_conversion_flag_for_child_id(const TypeChildId &_) {

        AMM_error_invalid_arg(_ > snChildren_nonstandard,
                               "OctreeChildNodes<%d>::get_bounds_conversion_flags(%d) got invalid child_id!\n",
                               int(Dim), int(_));

        static std::vector<TypeDim> data = CoreConfig::bounds_conversion_flag_for_child_id();
        return data[_];
    }


    //! based on the child_id, return a bitmap for what axes needs to be scaled and shifted
    static inline
    TypeChildId
    get_child_id_for_bounds_conversion_flag(const TypeDim &_) {
        return CoreConfig::child_id_for_bounds_conversion_flag(_);
    }


    //! get occupancy flags for each child
    static inline
    TypeOccupancyMask
    get_occupancy_of_child(const  TypeChildId &_) {

        static VecOccMask data = CoreConfig::occupancy_flags();
        return data[_];
    }


    //! get occupancy flags for each node
    static inline
    TypeOccupancyMask
    get_occupancy_flags(const TypeChildMask &_) {

        TypeOccupancyMask occupancy = 0;
        for( TypeChildId c = 0; c < snChildren_nonstandard; c++) {
            if (_[c])
                occupancy |= get_occupancy_of_child(c);
        }
        return occupancy;
    }


    //! get child ids invalid with respect to dimensionality
    static inline
    TypeChildMask
    get_conflicts_for_axes(const  EnumAxes &_) {

        AMM_error_invalid_arg(_ >= AMM_pow2(Dim),
                               "OctreeChildNodes<%d>::get_directional_invalid(%d) got invalid axes!\n",
                               int(Dim), int(_));

        static VecChildMask data = CoreConfig::conflicts_for_axes();
        return data[as_utype(_)];
    }


    //! get components of a child respect to dimensionality
    static inline
    TypeChildMask
    get_components_for_axes(const  TypeChildId &_, const  EnumAxes &axes) {


        static const uint8_t naxes = AMM_pow2(Dim);

        AMM_error_invalid_arg(_ > snChildren_nonstandard,
                               "OctreeChildNodes<%d>::get_child_components(%d) got invalid child_id!\n",
                               int(Dim), int(_));

        AMM_error_invalid_arg(axes >= naxes,
                               "OctreeChildNodes<%d>::get_child_components(%d) got invalid axes!\n",
                               int(Dim), int(axes));

        static VecVecChildMask data = CoreConfig::components_for_axes();
        return data[_][as_utype(axes)];
    }


    //! get conflicts for each child node as a vector of childflags
    static inline
    TypeChildMask
    get_conflicts_for_child(const  TypeChildId &_) {

        AMM_error_invalid_arg(_ > snChildren_nonstandard,
                               "OctreeChildNodes<%d>::get_node_conflicts(%d) got invalid child_id!\n",
                               int(Dim), int(_));

        static VecChildMask data = CoreConfig::conflicts_for_child();
        return data[_];
    }


    //! how to split a child, when another has been requested
    static inline
    TypeChildMask
    get_components_for_child(const  TypeChildId &to_split, const  TypeChildId &to_create) {
        return CoreConfig::components_for_child(to_split, to_create);
    }


    //! components of a node
    static inline
    TypeChildMask
    get_node_components(const  TypeChildId &cid) {

        static VecChildMask data = CoreConfig::node_components();
        return data[cid];
    }


#ifdef UNUSED
    //! get merge components of all child nodes
    static VecBitMask& get_merge_components(const  TypeChildId &_) {

        static VecVecBitMask data = merge_components();
        return data[_];
    }
#endif


    /// ------------------------------------------------------------------------------------
    //! given a childflag, compute the corresponding list of children
    static inline
    void
    create_childlist(const TypeChildMask &_, std::vector<TypeChildId> &childlist) {

        childlist.clear();
        childlist.reserve(snChildren_nonstandard);
        for( TypeChildId c = 0; c < snChildren_nonstandard; c++) {
            if (_[c])
                childlist.push_back(c);
        }
    }






    /// -----------------------------------------------------------------------------------
    /// -----------------------------------------------------------------------------------
    //! encode and decode the child flags into config
    /// -----------------------------------------------------------------------------------
    /// -----------------------------------------------------------------------------------

    static inline TypeConfig encode(const TypeChildMask &_) {

        #ifndef AMM_ENCODE_CFLAGS
            return _.value();
        #endif

        static const std::string filename = fname_configs(0);
        static TypeConfigSet encodings = read_configs(filename);
        static typename TypeConfigSet::const_iterator ebegin = encodings.begin();
        static typename TypeConfigSet::const_iterator eend   = encodings.end();

        typename TypeConfigSet::const_iterator iter = encodings.find(_.value());
        AMM_error_invalid_arg(iter == eend,
                              "Got unknown child flags (%lu)!\n", static_cast<long long>(_.value()));

        return (TypeConfig)std::distance(ebegin, iter);
    }
    static inline TypeChildMask decode(const TypeConfig &_) {

        #ifndef AMM_ENCODE_CFLAGS
            return TypeChildMask(_);
        #endif

        static const std::string filename = fname_configs(0);
        static TypeConfigSet encodings = read_configs(filename);
        static typename TypeConfigSet::const_iterator ebegin = encodings.begin();
        static size_t nencodings = encodings.size();

        AMM_error_invalid_arg(_ >= nencodings,
                              "Got unknown config!\n", int(_));


        typename TypeConfigSet::const_iterator iter = ebegin;
        std::advance(iter, _);
        return TypeChildMask(*iter);
    }

    /// --------------------------------------------------------------------------------
    /// --------------------------------------------------------------------------------
    /// key functionality to manipulate configs
    /// --------------------------------------------------------------------------------
    /// --------------------------------------------------------------------------------


    /// -----------------------------------------------------------------------------------
    //! create a standard child
    /// -----------------------------------------------------------------------------------
    static inline
    TypeChildMask
    create_standard_child(TypeChildMask _, const  TypeChildId &child_id) {

        AMM_error_invalid_arg(child_id >= snChildren,
                              "got invalid child_id %d!\n", int(child_id));

 #ifdef AMM_MAPPED_CONFIGS
        /// use mappings!
        static std::vector<std::unordered_map<TypeChildFlag, TypeChildFlag>> core_map;
        if (core_map.empty()) {
            core_mappings(core_map, "create");
        }

        auto iter = core_map[child_id].find(_);
        if (iter != core_map[child_id].end()) {
            return iter->second;
        }
#endif

        /// functional algorithm
        // if this child already exists
        if (_[child_id]) {                      return _;   }

        // if only regular children exist, simply add the new child
        if (_.value() < sCflags_full) {         _.set(child_id);   return _;   }

        // break any larger child that conflict with the request
        TypeChildMask conflicts = get_conflicts_for_child(child_id);

        // node conflicts contain self as well. we don't want that!
        conflicts.unset(child_id);

        // not check for conflicting children
        for( TypeChildId cid = 0; cid < snChildren_nonstandard; cid++) {

            if (!conflicts[cid])    continue;   // not a conflict
            if (!_[cid])            continue;   // not existing

            const TypeChildMask ccomps = get_components_for_child(cid, child_id);

            _ |= ccomps;            // set all its components
            _.unset(cid);           // unset the child to break
        }

        // due to vacuum, no larger child containing the requested child may exist
        _.set(child_id);
        return _;
    }

    /// -----------------------------------------------------------------------------------
    //! split a node along given axes
    /// -----------------------------------------------------------------------------------
    static inline
    TypeChildMask
    split_along_axes(TypeChildMask _, const  EnumAxes &split_axes) {

        AMM_error_invalid_arg(split_axes >= snAxes,
                              "got invalid split_axes %d!\n",int(split_axes));

        if (_.all())                        return _;                       // already full split
        if (split_axes ==  EnumAxes::None)   return _;                      // no split needed!
        if (split_axes == static_cast<int>(snAxes)-1) return sCflags_full;  // full split is needed!

#ifdef AMM_MAPPED_CONFIGS
        /// use mappings!
        static std::vector<std::unordered_map<TypeChildFlag, TypeChildFlag>> core_map;
        if (core_map.empty()) {
            core_mappings(core_map, "split");
        }

        auto iter = core_map[split_axes].find(_);
        if (iter != core_map[split_axes].end()) {
            return iter->second;
        }
#endif

        /// functional algorithm
        // if this is a leaf node, do a trivial split
        if(_.none()) {
             TypeChildId fakeid = snChildren_nonstandard;
            return get_components_for_axes(fakeid, split_axes);
        }

        // otherwise, decompose the children that cannot exist across these axes
        const TypeChildMask conflicts = get_conflicts_for_axes(split_axes);

        for( TypeChildId cid = snChildren_nonstandard-1; cid >= snChildren; cid--) {

            // if this is an invalid child that exists
            if (!conflicts[cid])    continue;   // not a conflict
            if (!_[cid])            continue;   // not existing

                const TypeChildMask ccomps = get_components_for_axes(cid, split_axes);

                _ |= ccomps;            // set all its components
                _.unset(cid);           // unset the child to break
        }

        // let the split along axes create vacuum if needed
        return fill_vacuum_along_axes(_, split_axes, conflicts);
    }

    /// -----------------------------------------------------------------------------------
    //! fix vacuum in a given node
    /// -----------------------------------------------------------------------------------
    //! fix the vaccum, but disallow the forbidden children
    static inline
    TypeChildMask
    fill_vacuum_complete(TypeChildMask _, const TypeChildMask forbidden = 0) {

        if (!has_vacuum(_))
            return _;

        // try to fill vacuum in the reverse order of children
        for(int c = snChildren_nonstandard-1; c >= 0; c--) {

            // ignore the existing children!
            if (_[c])            continue;

            // ignore the forbidden children!
            if (forbidden[c])    continue;

            // if none of the conficts exist
            if ((get_conflicts_for_child(c) & _).none()) {
                _.set(c);
            }
        }
#ifdef AMM_DEBUG_LOGIC
        // now, check if we did the right job!
        print(_)
        amm::utils::get_exception<std::logic_error>(has_vacuum(_), "OctreeChildNodes<%d>::fill_vacuum() failed to fill vacuum! created ", int(Dim));
#endif
        return _;
    }


    //! fix the vaccum, after splitting along axes!
    static inline
    TypeChildMask
    fill_vacuum_along_axes(TypeChildMask _, const  EnumAxes &split_axes, const TypeChildMask &forbidden) {

        if (!has_vacuum(_))
            return _;

        // for 2d, use the normal vacuum filling
        // since after splitting along any axes, there cannot be any vacuum left!
        if (2==Dim)     return fill_vacuum_complete(_, forbidden);

        // for 3d, if we're splitting along two or more axes,
            // there cannot be any vacuum left
        switch(split_axes) {
        case  EnumAxes::XY:
        case  EnumAxes::XZ:
        case  EnumAxes::YZ:
        case  EnumAxes::XYZ:  return fill_vacuum_complete(_, forbidden);
        default:              AMM_error_invalid_arg(true, "Invalid split_axes %d!", split_axes);
        }

        // if we're splitting along one axis only,
        // there is a degree of freedom along the other two dimensions
        // so we need to maintain vacuum!
        // this degree of freedom is present only when we only a single occupancy
        const TypeOccupancyMask occup = get_occupancy_flags(_);
        if (occup.count() > 1) {
            return fill_vacuum_complete(_, forbidden);
        }

        // try to fill vacuum in the reverse order of children
        // only create the children long in 2 dimensions!
        for(int c = snChildren_nonstandard-1; c >= 20; c--) {

            // ignore the existing children!
            if (_[c])            continue;

            // ignore the forbidden children!
            if (forbidden[c])    continue;

            // if none of the conficts exist
            if ((get_conflicts_for_child(c) & _).none()) {
                _.set(c);
            }
        }

        return _;
    }






    /// -----------------------------------------------------------------------------------
    /// i/o functionalities and temporary recording of all configs
    /// -----------------------------------------------------------------------------------

    //! print the children present in a configuration
    static inline
    void
    print(const TypeChildMask &_, bool end_line = true) {

        AMM_log_info << " " << _ << " [";
        for( TypeChildId c = 0; c < snChildren_nonstandard; c++)
            if (_[c]){
                AMM_log_info << " " << int(c);
            }
        if (end_line)   AMM_log_info << " ]\n";
        else            AMM_log_info <<  " ]";
    }

    static inline
    void
    print_configs(const TypeConfigSet &_) {
        size_t i = 0;
        for(auto iter = _.begin(); iter != _.end(); ++iter)
            std::cout << " : " << i++ << " === " << *iter << std::endl;
    }

    static inline
    std::string
    fname_configs(bool do_vacuum) {

        std::string filename = (2 == Dim) ? "configs_2d" : "configs_3d";
        filename = filename.append("_v")
                           .append(std::to_string(int(do_vacuum)))
                           .append(".conf");
        return filename;
    }

    static inline
    TypeConfigSet
    read_configs(const std::string &filename) {

        TypeConfigSet _;

        std::ifstream infile(filename);
        if (!infile.is_open()) {
            AMM_log_error << " Could not open config file (" << filename << ")\n";
            return _;
        }

        AMM_log_info << " Reading configs from file (" << filename << ")...";
        fflush(stdout);

        int x;
        while(infile >> x) {
            _.insert(TypeChildFlag(x));
        }

        infile.close();
        AMM_logc_info << " Done! Read " << _.size() << " values!\n";
        return _;
    }

    static inline
    void
    write_configs(const std::string &filename, const TypeConfigSet &_) {

        std::string filename2 = filename;
        filename2 = filename.substr(0, filename.length()-5).append(".expanded.conf");

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            AMM_log_error << " Could not open config file (" << filename << ")\n";
            return;
        }

        std::ofstream outfile2(filename2);
        if (!outfile2.is_open()) {
            AMM_log_error << " Could not open config file (" << filename2 << ")\n";
            return;
        }

        AMM_log_error << " Writing " << _.size() << " configs to (" << filename << ")...";
        fflush(stdout);

        for(auto iter = _.begin(); iter != _.end(); iter++) {
            outfile << *iter << "\n";

            outfile2 << *iter << " [";
            for( TypeChildId c = 0; c < snChildren_nonstandard; c++)
                if (AMM_is_set_bit32(*iter, c))
                    outfile2 << c << " ";
            outfile2 << "]\n";
        }

        outfile.close();
        outfile2.close();
        AMM_logc_info << " Done!\n";
    }

    static inline
    std::vector<TypeMappingMap>
    read_mappings(const std::string &filename) {

        std::vector<TypeMappingMap> _ (snChildren);

        std::ifstream infile(filename);
        if (!infile.is_open()) {
            AMM_log_error << " Could not open config file (" << filename << ")\n";
            return _;
        }

        AMM_log_info << " Reading configs from file (" << filename << ")...";
        fflush(stdout);

        size_t cnt = 0;
        int x;
        size_t y, z;
        while(infile >> x >> y >> z) {
            _[x][TypeChildFlag(y)] = TypeChildFlag(z);
            cnt++;
        }

        infile.close();
        AMM_logc_info << " Done! Read " << _.size() << " values!\n";
        return _;
    }

    static inline
    void
    write_configs(const std::string &filename, const std::vector<TypeMappingMap> &_) {

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            AMM_log_error << " Could not open config file (" << filename << ")\n";
            return;
        }

        AMM_log_error << " Writing " << _.size() << " configs to (" << filename << ")...";
        fflush(stdout);

        size_t cnt = 0;
        size_t i = 0;
        for(auto iter = _.begin(); iter != _.end(); iter++, i++) {
            const TypeMappingMap &sc = *iter;
            for(auto siter = sc.begin(); siter != sc.end(); siter++) {
                outfile << i << " " << siter->first << " " << siter->second << "\n";
                cnt++;
            }
        }

        outfile.close();
        AMM_logc_info << " Done! Wrote " << cnt << " values!\n";
    }

    static inline
    void
    insert_config(TypeConfigSet &configs, const TypeChildFlag &v) {
        configs.insert(v);
    }

    // add: config[a][b] = v
    template<typename T>
    static inline
    void
    insert_config(std::vector<TypeMappingMap> &_, const T &a, const TypeChildFlag &b, const TypeChildFlag &v) {
        TypeMappingMap &c = _[size_t(a)];
        if (c.find(b) == c.end())
            c[b] = v;
    }

    static inline
    void
    update_configs(const TypeConfigSet &_, bool do_vacuum) {

        const std::string filename = fname_configs(do_vacuum);
        TypeConfigSet configs = read_configs(filename);
        configs.insert(_.begin(), _.end());
        write_configs(filename, configs);
    }

    static inline
    void
    update_encodings(const std::vector<TypeMappingMap> &_, const std::string type) {

        return;
        const std::string filename = (2 == Dim) ? type+"_2d.conf" : type+"_3d.conf";

        // merge with existing ones!
        std::vector<std::unordered_map<TypeChildFlag, TypeChildFlag>> configs;
        read_configs(filename, configs);

        for( TypeChildId i = 0; i < _.size(); i++) {
            for(auto iter = _[i].begin(); iter != _[i].end(); iter++) {
                configs[i][iter->first] = iter->second;
            }
        }
        write_configs(filename, configs);
    }

    /// -----------------------------------------------------------------------------------
    /// -----------------------------------------------------------------------------------
};


}       // end of namespace
#endif
