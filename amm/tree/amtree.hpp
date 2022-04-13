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
#ifndef AMM_AMTREE_H
#define AMM_AMTREE_H

//! --------------------------------------------------------------------------------
#include <set>
#include <limits>
#include <string>
#include <bitset>
#include <utility>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "macros.hpp"
#include "types/dtypes.hpp"
#include "types/byte_traits.hpp"
#include "containers/vec.hpp"
#include "containers/unordered_set.hpp"
#include "containers/ordered_unique.hpp"
#include "containers/cache_manager.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"
#include "utils/data_utils.hpp"
#include "utils/mlinear_interpolator.hpp"
#include "tree/node_config.hpp"
#include "tree/octree_cell_iterators.hpp"
#include "tree/lcode.hpp"
#include "tree/octree_utils.hpp"
#include "tree/node_config.hpp"
#include "tree/node_splitter.hpp"
#include "tree/vmanager_umap.hpp"
#include "tree/vmanager_blocks.hpp"
#include "tree/block_umap.hpp"
#include "tree/block_precision.hpp"


#include <map>

//! -------------------------------------------------------------------------------------
#define AMM_AMTREE_INVALID_VAL(T)   AMM_nan(T)
#define AMM_AMTREE_IS_INVALID(T)    AMM_is_nan(T)


//! -------------------------------------------------------------------------------------
namespace amm {

//! -------------------------------------------------------------------------------------
template <typename TypeValue>
using AMM_vertex_data = std::pair<TypeValue, TypePrecision>;
std::string AMM_vertex_data_names[2] = {"value", "vertex_precision"};

using AMM_cell_data = std::tuple<std::vector<TypeIndex>, TypeScale, TypeDim, TypeChildId>;
std::string AMM_cell_data_names[4] = {"corner_idxs", "cell_level", "cell_type", "child_id"};

using AMM_field_data = std::tuple<double, size_t>;
std::string AMM_field_data_names[2] = {"wavelet_threshold", "wavelet_count"};



//! -------------------------------------------------------------------------------------
namespace amtree {

//! -------------------------------------------------------------------------------------
//! An adaptive tree datastructure for AMM
//!
//! This tree supports non-square leavf nodes.
//!
//! Total bits needed per location code: (2*Dim-1)*MaxLevels + 1
//! 2D: 3 bits per level can support the required 8 children
//! 3D: 5 bits per level can support the required 26 children
//!
//! -------------------------------------------------------------------------------------
template <typename TypeValue, TypeDim Dim, TypeScale MaxLevels>
class AMTree {

    static_assert((Dim == 2 || Dim == 3), "amTree works for 2D and 3D only!");


    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! static variables
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
public:
    static constexpr TypeCornerId snCorners = Dim==2 ? 4 : 8;   // number of corners of a node
    static constexpr TypeCornerId snSplits = Dim==2 ? 5 : 19;   // number of split points of a node
    static constexpr TypeChildId snChildren = Dim==2 ? 4 : 8;   // number of children in an octree
    static constexpr TypeChildId snChildren_nonstandard = (Dim==2) ? 8 : 26;


    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! typedefs
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
protected:
    // data type needed to represent the location code of a node
    using LocCode           = location_code<MaxLevels, TypeDim(2*Dim-1)>;
    using TypeLocCode       = typename LocCode::dtype;
    using TypeLocCodeBitset = std::bitset<CHAR_BIT*sizeof(TypeLocCode)>;

    // data type to represent D-dimensional vertices and vectors
    using TypeVertex        = Vec<Dim, TypeCoord>;
    using OctUtils          = amm::octree_utils<Dim>;

    // multilinear interpolator and splitter
    using TypeInterpolator  = amm::MultilinearInterpolator<Dim, TypeValue>;
    using TypeSplitter      = amm::MultilinearNodeSplitter<Dim, TypeValue>;
    using TypeSplitFlag     = typename TypeSplitter::TypeSplitFlag;

    // node configuration
    using NodeConfig        = amm::OctreeNodeConfiguration<Dim>;
    using TypeChildFlag     = typename NodeConfig::TypeChildFlag;
    using TypeChildMask     = typename NodeConfig::TypeChildMask;
    using TypeConfig        = typename NodeConfig::TypeConfig;

    // list of nodes and vertices
    using VertexIdCache     = amm::cache_manager<TypeIndex>;
    using NodeSet           = amm::unordered_set<TypeLocCode>;
    using NodeMap           = amm::unordered_map<TypeLocCode, TypeConfig>;
    using NodeMap_Stg       = amm::unordered_map<TypeLocCode, EnumAxes>;

    // different vertex managers
    using VManager          = amm::vertex_manager<Dim, TypeValue>;
    using VMngr_Simple      = amm::vertex_manager_umap<Dim, TypeValue>;
    using VMngr_DeltaV      = amm::vertex_manager_umap<Dim, TypeValue>;
    using VMngr_BoundaryV   = amm::vertex_manager_umap<Dim, TypeValue>;

#ifndef AMM_ENABLE_PRECISION
    using VMngr_AggregatedV = amm::vertex_manager_umap<Dim, TypeValue>;
    using VMngr_FinalV      = amm::vertex_manager_umap<Dim, TypeValue>;
    using VMap_ImproperV    = amm::vertex_manager_umap<Dim, TypeValue>;
#else
    using VMngr_AggregatedV = amm::vertex_manager_blocks<Dim, TypeValue, amm::block_umap<TypeValue>>;
    using VMngr_FinalV      = amm::vertex_manager_blocks<Dim, TypeValue, amm::block_precision<TypeValue>>;
    using VMap_ImproperV    = amm::vertex_manager_blocks<Dim, TypeValue, amm::block_precision<TypeValue>>;
#endif

    // datatrucrures to provide iteration over octree cells
    using DualIterator          = typename amm::octree::dual_cell_iterator<Dim>;
    using NeighborAllIterator   = typename amm::octree::neighboring_cell_all_iterator<Dim>;
    using PrimalDualIterator    = typename amm::octree::primal_dual_cell_iterator<Dim>;
    using PrimalIterator        = typename amm::octree::primal_cell_iterator<Dim>;

    // datastructures to represent the "final" mesh
    using TypeFinalCell     = AMM_cell_data;
    using TypeFinalVert     = AMM_vertex_data<TypeValue>;
    using ListFinalCells    = std::vector<TypeFinalCell>;
    using ListFinalVerts    = std::unordered_map<TypeIndex, TypeFinalVert>;

    // time stamp object
    using tstamp            = amm::timer::tstamp;


    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! member variables
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
private:
    const bool m_enableRectangularNodes;// whether nodes can be rectangular
    const bool m_enableImproperNodes;   // whether vacuum nodes are allowed

    const TypeVertex mOrigin;           // = 0
    const TypeVertex mTreeBox1;         // = 2^L
    const TypeVertex mTreeSize;         // = 2^L + 1
    const TypeVertex mTreeCenter;       // = 2^(L-1)+1
    const TypeScale  mTreeDepth;        // = L

    NodeMap mNodes;                     // a map of internal nodes and their child flags

    VMngr_FinalV mVertices;             // "final" vertices
    VMngr_AggregatedV mVertices_agg;    // "aggregated" vertices
    VMap_ImproperV mVertices_Improper;  // "improper" vertices

    NodeSet mLeaves_at_bdry;            // nodes at boundary
    VMngr_BoundaryV mVertices_at_bdry;  // verts at boundary

    // temporary staging of nodes and vertices
    NodeMap_Stg _nstaged;
    VMngr_DeltaV _vstaged[1+MaxLevels];
    VertexIdCache _vcache;
    VertexIdCache _vcache_bylvl[1+MaxLevels];

    bool m_is_nodeStaging_phase;            // flag for node staging phase
    size_t time_of_last_update;             // to identify stale iterators


#ifdef AMM_STORE_IMPROPER_NODES
    //! a set of improper nodes
    NodeSet mNodes_Improper;
#endif

#ifdef AMM_DEBUG_TRACK_CONFIGS
    typename NodeConfig::TypeConfigSet mconfigs;
    std::vector<typename NodeConfig::TypeMappingMap> mconfigs_create;
    std::vector<typename NodeConfig::TypeMappingMap> mconfigs_split;
#endif


public:
    //! --------------------------------------------------------------------------------
    //! accessors and simple queries
    //! --------------------------------------------------------------------------------
    inline const TypeVertex& origin() const {  return mOrigin;      }
    inline const TypeVertex& size() const {    return mTreeSize;    }
    inline const TypeVertex& mbox1() const {   return mTreeBox1;    }
    inline const TypeScale&  depth() const {   return mTreeDepth;   }


    //! --------------------------------------------------------------------------------
    //! maintain time of last update
    //! --------------------------------------------------------------------------------
    inline const size_t& get_utime() const {    return time_of_last_update;                 }
    inline void set_utime() {                   time_of_last_update = amm::timer::now_ms(); }


    //! --------------------------------------------------------------------------------
    //! idx to point conversion
    //! --------------------------------------------------------------------------------
    inline TypeIndex p2idx(const TypeVertex &_) const { return OctUtils::p2idx(_, mTreeSize); }
    inline TypeVertex idx2p(const TypeIndex &_) const { return OctUtils::idx2p(_, mTreeSize); }


    //! --------------------------------------------------------------------------------
    //! public functions called by the mesh creator
    //! --------------------------------------------------------------------------------
    //! add a vertex update
    inline
    void
    stage_vertex(const TypeVertex &p, const TypeScale lvl, const TypeValue val) {
        _vstaged[lvl].add(p2idx(p), val);
    }


    //! create a node at a given vertex
    inline
    TypeLocCode
    create_node_at(const TypeVertex &_) {

        //std::cout << " : create_node_at("<<_<<") : staging = " << m_is_nodeStaging_phase << "\n";
        const TypeLocCode ncode = ncenter_to_lcode(_);
        create_node(ncode);
        return ncode;
    }


    //! split a given node along requested axes
    inline
    void
    split_node(const TypeLocCode &_, const EnumAxes &split_axes) {

        auto usplit_axes = as_utype(split_axes);
        AMM_error_invalid_arg(!(usplit_axes >= 0 && usplit_axes < snChildren),
                               "Got invalid axes (%u(!\n",
                               std::bitset<Dim>(usplit_axes).to_string().c_str());

        //std::cout << " : split_node("<<_<<", " << size_t(as_utype(split_axes)) << ") : staging = " << m_is_nodeStaging_phase << "\n";
        if (split_axes == EnumAxes::None) {
            return;
        }

        if (m_is_nodeStaging_phase) {
            stage_node(_, split_axes);
            return;
        }
        if (!m_enableRectangularNodes) {
            split_node_basic(_);
        }
        else {
            split_node_adaptive_for_axes(_, split_axes);
        }
    }


protected:
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! constructor and initialization
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    AMTree(const size_t size[], bool allow_rectangular, bool allow_vacuum) :
                    mOrigin(TypeCoord(0)),
                    mTreeSize(OctUtils::size_tree(size, MaxLevels)),
                    mTreeBox1(OctUtils::os2bb(mOrigin, mTreeSize)),
                    mTreeCenter(OctUtils::os2c(mOrigin, mTreeSize)),
                    mTreeDepth(AMM_log2(mTreeSize[0])),
                    m_enableRectangularNodes(allow_rectangular),
                    m_enableImproperNodes(allow_vacuum) {
        init();
    }


    AMTree(const TypeVertex &size, bool allow_rectangular, bool allow_vacuum) :
                    mOrigin(TypeCoord(0)),
                    mTreeSize(OctUtils::size_tree(size, MaxLevels)),
                    mTreeBox1(OctUtils::os2bb(mOrigin, mTreeSize)),
                    mTreeCenter(OctUtils::os2c(mOrigin, mTreeSize)),
                    mTreeDepth(AMM_log2(mTreeSize[0])),
                    m_enableRectangularNodes(allow_rectangular),
                    m_enableImproperNodes(allow_vacuum) {
        init();
    }


    ~AMTree() {}


    //! --------------------------------------------------------------------------------
    //! lcode to point conversion
    //! --------------------------------------------------------------------------------
    inline
    TypeLocCode
    ncenter_to_lcode(const TypeVertex &_) const {
        return (2 == Dim) ?
                LocCode::from_coords(_[0], _[1], mTreeDepth) :
                LocCode::from_coords(_[0], _[1], _[2], mTreeDepth);
    }


    inline
    TypeVertex
    lcode_to_ncenter(const TypeLocCode &_) const {
        TypeVertex p;
        if (2 == Dim) { LocCode::to_coords(_, mTreeDepth, p[0], p[1]);        }
        else {          LocCode::to_coords(_, mTreeDepth, p[0], p[1], p[2]);  }
        return p;
    }


    //! --------------------------------------------------------------------------------
    //! queries in the tree
    //! --------------------------------------------------------------------------------
    inline bool is_node(const TypeLocCode &_) const {   return mNodes.contains(_);    }
    inline bool is_vertex(const TypeIndex &_) const {  return mVertices.contains(_);  }


    //! whether this vertex falls within tree bounds
    inline
    bool
    is_valid_vertex(const TypeVertex &_) const {
        return OctUtils::contains_os(_, mOrigin, mTreeSize);
    }

    //! check for the leaf
    inline
    bool
    is_leaf(const TypeLocCode &_) const {

        // if this is an internal node, then clearly not a leaf
        if (is_node(_)) {
            return false;
        }

        // if the tree is empty, root is a leaf
        if (mNodes.empty() && 1 == _) {
            return true;
        }

        // if the parent does not exist, this is not even a node
        auto piter = mNodes.find(LocCode::parent(_));
        if (piter == mNodes.end()) {
            return false;
        }

        // check if this particular child of the parent exists!
        auto pconfig = NodeConfig::decode(piter->second);
        auto cid =  LocCode::childId(_);
        return NodeConfig::child_exists(pconfig, cid);
    }


    //! find the closest ancestor node that exists
    inline
    TypeLocCode
    find_closest_ancestor(TypeLocCode ncode) const {
        while (ncode > 1) {
            if (is_node(ncode))
                break;
            ncode = LocCode::parent(ncode);
        }
        return ncode;
    }


    //! whether a node intersects with a given bound
    inline
    bool
    is_node_intersects(const TypeLocCode &_, const TypeVertex &box0, const TypeVertex &box1) const {
        static TypeVertex nbox0, nbox1, nsize;
        get_bounds_node(_, nbox0, nsize);
        OctUtils::os2bb(nbox0, nsize, nbox1);
        return !OctUtils::is_outside(nbox0, nbox1, box0, box1);
    }

    //! --------------------------------------------------------------------------------
    //! find bounds of a node by traversing the tree
    //! --------------------------------------------------------------------------------
    inline
    void
    get_bounds_node(const TypeLocCode &ncode, TypeVertex &norigin, TypeVertex &nsize) const {

        const TypeChildId cid = LocCode::childId(ncode);

        // for a regular child, simply use the basic traversal
        if (cid < snChildren) {
            const TypeCoord node_sz = AMM_pow2(mTreeDepth-LocCode::level(ncode));
            norigin = lcode_to_ncenter(ncode) - TypeVertex(node_sz>>1);
            nsize = TypeVertex(node_sz+1);
        }

        // for a non-square child, use the basic traversal to get to its parent
        else {
            const TypeLocCode pcode = LocCode::parent(ncode);

            const TypeCoord node_sz = AMM_pow2(mTreeDepth-LocCode::level(pcode));
            norigin = lcode_to_ncenter(pcode) - TypeVertex(node_sz>>1);
            nsize = TypeVertex(node_sz+1);

            AMTree::get_bounds_child(cid, norigin, nsize);
        }
    }


    //! --------------------------------------------------------------------------------
    //! find corners for a set of node bounds
    //! --------------------------------------------------------------------------------
    inline
    void
    get_node_corners(const TypeVertex &norigin, const TypeVertex &nsize,
                     std::vector<TypeIndex> &corners) const {
        OctUtils::get_node_corners(p2idx(norigin), nsize, mTreeSize, corners);
    }


    inline
    void
    get_node_corners(const TypeVertex &norigin, const TypeVertex &nsize,
                     std::vector<TypeIndex> &corners,
                     std::vector<TypeVertex> &cornerPts) const {

        get_node_corners(norigin, nsize, corners);
        for(TypeCornerId i = 0; i < snCorners; i++) {
            cornerPts[i] = idx2p(corners[i]);
        }
    }


    inline
    void
    get_node_corners(const TypeVertex &norigin, const TypeVertex &nsize,
                     std::vector<TypeIndex> &corners,
                     std::vector<TypeValue> &cornerVals) const {

        get_node_corners(norigin, nsize, corners);
        for(TypeCornerId i = 0; i < snCorners; i++) {
            cornerVals[i] = get_vertex_value(corners[i]);
        }
    }



    //! --------------------------------------------------------------------------------
    //! get the child id that contains a given point (wrt node's center)
    //! --------------------------------------------------------------------------------
    inline
    TypeChildId
    get_child_containing(const TypeLocCode ncode, const TypeVertex &ncenter,
                         const TypeVertex &v) {

        const TypeChildFlag cflags = get_node_cflags(ncode).value();
        const TypeChildId child_id = OctUtils::get_child_containing(v, ncenter);

        // if this child exists, we're good!
        if (AMM_is_set_bit32(cflags, child_id))
            return child_id;

        // if this child does not exist, we need to find the "larger" child that contains this point!
        const TypeChildFlag conflicts = NodeConfig::get_conflicts_for_child(child_id).value();

        static std::vector<TypeChildId> common_children;
        NodeConfig::create_childlist(conflicts&cflags, common_children);

        if (common_children.size() == 1) {
            return common_children.front();
        }

        AMM_error_logic(true, "AdaptiveTree.get_child_containing(%s) failed!\n", v.c_str());
    }


    //! --------------------------------------------------------------------------------
    //! get value and precision of a vertex
    //! --------------------------------------------------------------------------------
    inline
    TypeValue
    get_vertex_value(const TypeIndex _, const bool include_vacuum = false) const {

        bool is_found = false;
        TypeValue val = mVertices.get(_, is_found);
        if (is_found)
            return val;

        if (m_enableImproperNodes && include_vacuum)
            val = mVertices_Improper.get(_, is_found);

        return is_found ? val :
                          AMM_missing_vertex(TypeValue);
    }


    inline
    TypePrecision
    get_vertex_precision(const TypeIndex _, const bool include_vacuum = false) const {

        TypePrecision prec = mVertices.precision(_);
        if (prec > 0)
            return prec;

        if (m_enableImproperNodes && include_vacuum)
            prec = mVertices_Improper.precision(_);

        return prec;
    }


    inline
    TypeChildMask
    get_node_cflags(const TypeLocCode &_) const {
        auto iter = mNodes.find(_);
        return iter == mNodes.end() ? TypeChildMask (0)
                                    : NodeConfig::decode(iter->second);
    }


    //! --------------------------------------------------------------------------------
    //! get value at an arbitrary point on the domain
    //! --------------------------------------------------------------------------------
    inline
    TypeValue
    get_value(const TypeVertex &_) const {
        return get_value(_, find_leaf(_));
    }


    inline
    TypeValue
    get_value(const TypeVertex &_, const TypeLocCode &l) const {

        // use a cached interpolator
        static TypeLocCode prev_l = 0;
        static TypeInterpolator* minterp = nullptr;
        if (prev_l != l) {
            delete minterp;
            minterp = get_node_interpolator(l);
            prev_l = l;
        }

        // compute the value
        return minterp->compute(_, true);
    }



    //! create a multilinear interpolator for a given node
    inline
    TypeInterpolator*
    get_node_interpolator(const TypeLocCode &_) const {

        static TypeVertex nbox0, nbox1, nsize;
        static std::vector<TypeIndex> ncorners(snCorners);
        static std::vector<TypeValue> ncorner_values(snCorners);

        get_bounds_node(_, nbox0, nsize);
        get_node_corners(nbox0, nsize, ncorners, ncorner_values);

        OctUtils::os2bb(nbox0, nsize, nbox1);
        return new TypeInterpolator (nbox0, nbox1, ncorner_values);
    }


    //! create the nodes and vertices at the boundary of the domain
    void
    finalize_boundary(const TypeVertex &domainBox1) {

        amm::timer t;
        AMM_log_info << "\tCreating boundary...";
        fflush(stdout);

        //! clear existing boundary vertices
        mVertices_at_bdry.clear();
        mLeaves_at_bdry.clear();

        //! create the boundary recursively by finding nodes that contain the boundary
        create_boundary(domainBox1, 1, mOrigin, mTreeSize);

        t.stop();
        AMM_logc_info << " done!" << t
                      << " created " << mVertices_at_bdry.size() << " boundary vertices for "
                                     << mLeaves_at_bdry.size() << " boundary nodes!\n";
    }


    //! create the vacuum leaves and vacuum vertices by finalizing the vacuum nodes
    void
    finalize_improper_nodes() {

        if (!m_enableImproperNodes)
            return;

        AMM_log_info << "\tFinalizing improper nodes...";

        clear_vcache();
        mVertices_Improper.clear();

        size_t nnodes_vac = 0;
        for (auto iter = mNodes.begin(); iter != mNodes.end(); iter++) {

            const TypeChildFlag cflags = iter->second;
            if (!NodeConfig::has_vacuum(cflags)) {
                continue;
            }

            const TypeChildFlag vflags = NodeConfig::fill_vacuum_complete(cflags).value();
            update_node(iter->first, vflags, true);
            nnodes_vac += 1;
        }
        AMM_logc_info << " done! created " << mVertices_Improper.size() <<" vacuum vertices from " << nnodes_vac << " nodes!\n";
    }


    //! aggregate nodes and vertex staging
    void
    finalize_nodes_and_vertices() {

        // ------------------------------------------------------------------------------
        // unstage nodes (creates nodes, add split points)
        // ------------------------------------------------------------------------------
        amm::timer tn;

#ifdef AMM_STAGE_NODES
        AMM_log_info << "\tUnstaging ["<<count_staged_nodes()<<" nodes]...";
        fflush(stdout);

        m_is_nodeStaging_phase = false;

        auto snodes = _nstaged.as_sorted_vector();
        for(auto iter = snodes.begin(); iter != snodes.end(); iter++ ) {
            create_node(iter->first);
            if (iter->second > 0) {
                split_node(iter->first, iter->second);
            }
        }

        m_is_nodeStaging_phase = true;
        tn.stop();
        AMM_logc_info << " done!" << tn
                      << " created " << mNodes.size() << " nodes and " << mVertices.size() << " vertices!\n";

#else
        // TODO: populate snodes!
#endif
        // ------------------------------------------------------------------------------
        // finalizing vertex staging
        // ------------------------------------------------------------------------------
        amm::timer tv;
        AMM_log_info << "\tFinalizing ["<<count_staged_vertices()<<" vertices from "<<count_staged_nodes()<<" nodes]...";
        fflush(stdout);

        //m_vcache.clear();
        clear_vcache();
        clear_vcache_atlvl();
        clear_aggregates();

        // aggregate using a recursive depth-first traversal
        TypeVertex nbox0, nsize;
        for(auto iter = snodes.begin(); iter != snodes.end(); iter++) {

            TypeLocCode ncode = iter->first;

            // if this is not staged anymore!
            if (!is_staged_node(ncode))
                continue;

            if (ncode > 1) {
                ncode = LocCode::parent(ncode);
            }
            get_bounds_node(ncode, nbox0, nsize);
            aggregate_vertex_updates_pushdown(ncode, nbox0, nsize);
        }

        AMM_error_logic(count_staged_nodes(), "Cannot have staged nodes after finalizing");

        // ------------------------------------------------------------------------------
        // move from aggegated_v to final_v
        const size_t n = mVertices_agg.size();
        mVertices_agg.copy_to(mVertices);

        // clear all the by-level vertices!
        clear_staged_vertices();
        clear_aggregates();

        tv.stop();
        AMM_logc_info << " done!" << tv
                     << " aggregated " << n << " vertices."
                     << " total vertices = " << mVertices.size() << "!\n";
    }


    void
    create_leaves(NodeSet &leaves, NodeSet &leaves_improper) const {

        amm::timer t;
        AMM_log_info << "Populating leaves...";
        fflush(stdout);

        leaves.clear();
        leaves_improper.clear();

        size_t nnodes_vac = 0;
        const size_t nnodes = mNodes.size();
        for (auto iter = mNodes.begin(); iter != mNodes.end(); iter++) {

            const TypeChildFlag cflags = iter->second;
            create_leaves(iter->first, cflags, leaves);

            if (!m_enableImproperNodes) {
                continue;
            }
            if (!NodeConfig::has_vacuum(cflags)) {
                continue;
            }

            const TypeChildFlag vflags = NodeConfig::fill_vacuum_complete(cflags).value() & ~cflags;
            create_leaves(iter->first, vflags, leaves_improper);
            nnodes_vac += 1;
        }

        t.stop();
        AMM_logc_info << " done!" << t << " Created " << leaves.size() << "+" << leaves_improper.size() << " leaves from " << nnodes << "+" << nnodes_vac << " nodes!\n";
    }


    bool
    reconstruct(const TypeVertex &dbox0, const TypeVertex &dsize,
                ListFinalVerts &vertices, ListFinalCells &cells) const {

        AMM_error_invalid_arg(!(dbox0 == TypeVertex(TypeCoord(0))), "ATree::reconstruct(): requires dbox0 = 0");

        NodeSet leaves, leaves_vacuum;
        create_leaves(leaves, leaves_vacuum);

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);

        amm::timer t;
        AMM_log_info << "Constructing unstructured grid ("<<dbox0 << ", " << dbox1 << ")...";
        fflush(stdout);

        vertices.clear();
        cells.reserve(leaves.size()+leaves_vacuum.size());

        for(auto iter = leaves.begin(); iter != leaves.end(); ++iter) {
            reconstruct(*iter, false, dbox0, dsize, vertices, cells);
        }
        for(auto iter = leaves_vacuum.begin(); iter != leaves_vacuum.end(); ++iter) {
            reconstruct(*iter, true, dbox0, dsize, vertices, cells);
        }

        t.stop();
        AMM_logc_info << " done!" << t << " Created " << vertices.size() << " vertices and " << cells.size() << " cells!\n";


        // count number of verts at different precision
        if (true) {
            std::map<TypePrecision, size_t> pcounts;
            for(auto viter = vertices.begin(); viter != vertices.end(); viter++) {
                pcounts[viter->second.second] ++;
            }

            std::cout << "\t prec: nverts = ";
            for(auto piter = pcounts.begin(); piter != pcounts.end(); piter++) {
                std::cout << " ("<< size_t(piter->first) << " : " << piter->second << ") ";
            }
            std::cout << "\n";
        }
        return true;
    }


    bool
    reconstruct(const TypeVertex &dbox0, const TypeVertex &dsize, std::vector<TypeValue> &func) const {

        AMM_error_invalid_arg(!(dbox0 == TypeVertex(TypeCoord(0))), "ATree::reconstruct(): requires dbox0 = 0");

        NodeSet leaves, leaves_vacuum;
        create_leaves(leaves, leaves_vacuum);

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);
        const size_t dnvalues = OctUtils::product(dsize);

        amm::timer t;
        AMM_log_info << "Constructing structured grid ("<<dbox0 << ", " << dbox1 << ")...";
        fflush(stdout);

        /*std::cout << "\n";
        print_nodes(mNodes);
        std::cout << "\n";
        print_nodes(leaves);*/
        static constexpr TypeValue vinvalid = AMM_AMTREE_INVALID_VAL(TypeValue);

        //! create an empty function (initialize with invalids)
        if (func.size() != dnvalues) {  func.resize(dnvalues, vinvalid);                }
        else {                          std::fill(func.begin(), func.end(), vinvalid);  }

        //! we will also check for c0 continuous function!
        size_t nmismatches_on_corners = 0;
        size_t nmismatches_on_boundary = 0;
        size_t nmismatches_inside = 0;

        std::set<TypeLocCode> errorneous_nodes;
        std::set<TypeIndex> errorneous_verts;

        // go over all leaf nodes and populate the function (no need to sort)
        auto sleaves = leaves.as_sorted_vector();
        for(auto iter = sleaves.rbegin(); iter != sleaves.rend(); ++iter) {
            reconstruct(*iter, false, dbox0, dsize, func,
                         nmismatches_on_corners, nmismatches_on_boundary, nmismatches_inside,
                         errorneous_nodes, errorneous_verts);
        }

        if (m_enableImproperNodes) {
            auto sleaves_vacuum = leaves_vacuum.as_sorted_vector();
            for(auto iter = sleaves_vacuum.rbegin(); iter != sleaves_vacuum.rend(); ++iter) {
                reconstruct(*iter, true, dbox0, dsize, func,
                             nmismatches_on_corners, nmismatches_on_boundary, nmismatches_inside,
                             errorneous_nodes, errorneous_verts);
            }
        }

        // check mismatches
        if (nmismatches_on_corners > 0) {
            AMM_log_error << "AMTree.reconstruct() failed matching value test for " << nmismatches_on_corners << " corners!\n";
        }
        if (nmismatches_on_boundary > 0) {
            AMM_log_error << "AMTree.reconstruct() failed matching value test for " << nmismatches_on_boundary << " boundary points!\n";
        }
        if (nmismatches_inside > 0) {
            AMM_log_error << "AMTree.reconstruct() failed matching value test for " << nmismatches_inside << " internal points!\n";
        }

        // check coverage of space
        size_t nninvalids = std::count_if(func.begin(), func.end(),
                                          [](const TypeValue v) { return AMM_AMTREE_IS_INVALID(v); });

        if (nninvalids > 0) {
            AMM_log_error << "AMTree.reconstruct() failed coverage test for " << nninvalids << " vertices!\n";

          /*for(int i = 0; i < func.size(); i++) {
              if (AMM_AMTREE_IS_INVALID(func[i])) {
                std::cerr << " : " << i << " : " << idx2p(i) << " = " << func[i] << "\n";
              }
          }*/
        }

        t.stop();
        AMM_logc_info << " done!" << t << std::endl;
        return nmismatches_on_corners==0 &&
               nmismatches_on_boundary==0 &&
               nmismatches_inside==0 &&
               nninvalids == 0;
    }


private:
    //! --------------------------------------------------------------------------------
    //! initialize the tree
    //! --------------------------------------------------------------------------------
    void
    init() {

        AMM_log_info << "Creating AMMTree<Dim="<<int(Dim)<<", L="<<int(MaxLevels)<<">:"
                     << " size = " << mTreeSize << ", origin = " << mOrigin << ":"
                     << " data_depth = " << int(mTreeDepth) << "\n";

        // size should be 2^L + 1 of equal size in all dimensions
        AMM_error_invalid_arg(!OctUtils::is_pow2plus1(mTreeSize), "Octree expects size 2^L+1! but found %d\n", mTreeSize);
        AMM_error_invalid_arg(!OctUtils::is_square(mTreeSize), " Octree expects square domain of size 2^L+1 but found %s\n", mTreeSize.c_str());

        // initialize the vertex managers
        mVertices.init(mTreeSize);
        mVertices_agg.init(mTreeSize);
        mVertices_at_bdry.init(mTreeSize);
        mVertices_Improper.init(mTreeSize);

        // and caches
        const bool use_dense_cache = false;
        const size_t msz = OctUtils::product(mTreeSize);
        _vcache.init(msz, use_dense_cache);
        for(TypeScale l = 0; l <= MaxLevels; l++) {
            _vstaged[l].init(mTreeSize);
            _vcache_bylvl[l].init(msz, (use_dense_cache && l > MaxLevels/2));
        }

#ifdef AMM_STAGE_NODES
        m_is_nodeStaging_phase = true;
#else
        m_is_nodeStaging_phase = false;
#endif
        // update the time!
        set_utime();

#ifdef AMM_DEBUG_TRACK_CONFIGS
        mconfigs_create.resize(snChildren);
        mconfigs_split.resize(snChildren);
#endif
    }


    //! --------------------------------------------------------------------------------
    //! staging and cache management
    //! --------------------------------------------------------------------------------
    inline
    void
    clear_staged_vertices() {
        for(size_t i = 0; i <= mTreeDepth; i++) {
            _vstaged[i].clear();
        }
    }


    inline
    TypeValue
    get_staged_vertex(const TypeIndex &p, const TypeScale lvl, bool &exists) const {
        return _vstaged[lvl].get(p, exists);
    }


    inline
    bool
    is_vertex_cached(const TypeIndex &p) const {
        return _vcache.contains(p);
    }


    inline
    void
    cache_vertex(const TypeIndex &p) {
        _vcache.set(p);
    }


    inline
    void
    clear_vcache() {
        _vcache.clear();
    }


    inline
    bool
    is_vertex_cached_at_lvl(const TypeIndex &p, const TypeScale l) const {
        return _vcache_bylvl[l].contains(p);
    }


    inline
    void
    cache_vertex_at_lvl(const TypeIndex &p, const TypeScale l) {
        _vcache_bylvl[l].set(p);
    }


    inline void
    clear_vcache_atlvl() {
        for(size_t i = 0; i <= mTreeDepth; i++) {
            _vcache_bylvl[i].clear();
        }
    }


    inline
    size_t
    count_staged_nodes() const {
        return _nstaged.size();
    }


    inline
    bool
    is_staged_node(const TypeLocCode &lcode) const {
        return _nstaged.contains(lcode);
    }


    inline
    void
    stage_node(const TypeLocCode &lcode, EnumAxes split_axes=EnumAxes::None) {
        auto iter = _nstaged.find(lcode);
        if (iter == _nstaged.end()) {
            _nstaged[lcode] = split_axes;
        }
        else {
            iter->second = as_enum<EnumAxes,uint8_t>(as_utype(iter->second) | as_utype(split_axes));
        }
    }


    inline
    void
    unstage_node(const TypeLocCode &lcode) {
        _nstaged.erase(lcode);
    }


    inline
    size_t
    count_staged_vertices() const {
        size_t sz = 0;
        for(size_t i = 0; i <= mTreeDepth; i++) {
            sz += _vstaged[i].size();
        }
        return sz;
    }

    inline
    void
    aggregate_vertex(const TypeIndex cid, TypeValue val) {
        mVertices_agg.add(cid, val);
    }

    inline
    void
    clear_aggregates() {
        mVertices_agg.clear();
    }


    //! --------------------------------------------------------------------------------
    //! add split points
    //! --------------------------------------------------------------------------------
    inline void
    add_split_vertices(const std::vector<TypeVertex> &ncorners,
                       const std::vector<TypeValue> &svalues,
                       VManager &vertices,
                       const TypeSplitFlag &splits_to_add=0) {

        static TypeVertex spoint;

        // all splits to be added!
        if (splits_to_add.none()) {

            for(TypeCornerId i = 0; i < snSplits; i++) {

                const TypeValue &sval = svalues[i];
                if (AMM_MLSPLIT_is_invalid(sval)) {
                    continue;
                }

                TypeSplitter::get_split_point(i, ncorners, spoint);
                auto spidx = this->p2idx(spoint);
                if (!mVertices.contains(spidx) && !is_vertex_cached(spidx)) {
                    cache_vertex(spidx);
                    vertices.add(spidx, sval);
                }
            }
        }

        // only specified splits to be added!
        else {

            for(TypeCornerId i = 0; i < snSplits; i++) {

                const TypeValue &sval = svalues[i];
                if (!splits_to_add[i] || AMM_MLSPLIT_is_invalid(sval)) {
                    continue;
                }

                TypeSplitter::get_split_point(i, ncorners, spoint);
                auto spidx = this->p2idx(spoint);
                if (!mVertices.contains(spidx) && !is_vertex_cached(spidx)) {
                    cache_vertex(spidx);
                    vertices.add(spidx, sval);
                }
            }
        }
    }


    //! --------------------------------------------------------------------------------
    //! create and split ndoes
    //! --------------------------------------------------------------------------------
    inline void
    create_node(const TypeLocCode &ncode) {

        //std::cout << " : create_node("<<ncode<<") : staging = " << m_is_nodeStaging_phase << "\n";
        if (m_is_nodeStaging_phase) {
            stage_node(ncode);
            return;
        }

        // @TODO: duplicate traversal (and hash queries)
        // first we go up to find the closest ancestor and then come down again
        // could we make this into fewer queries?
        // does it make sense to a binary lookup?

        TypeLocCode pcode = find_closest_ancestor(ncode);
        const TypeScale plvl = LocCode::level(pcode);
        const TypeScale nlvl = LocCode::level(ncode);


        if (!m_enableRectangularNodes) {

            for(TypeScale l = plvl; l < nlvl; l++) {
                pcode = LocCode::parentn(ncode, nlvl-l);
                split_node_basic(pcode);
            }
        }
        else {

            static TypeChildId childId = 0;
            for(TypeScale l = plvl; l < nlvl; l++) {
                pcode = LocCode::parentn(ncode, nlvl-l);
                childId = LocCode::childId(LocCode::parentn(ncode, nlvl-l-1));
                split_node_adaptive_for_child(pcode, childId);
            }
        }
    }


    void
    split_node_basic(const TypeLocCode ncode) {

        // static variables used in this function
        static TypeVertex norigin, nsize;
        static std::vector<TypeIndex> ncorners (snCorners);
        static std::vector<TypeVertex> ncornersPts (snCorners);
        static std::vector<TypeValue> svalues (snSplits);

        static const TypeConfig fconfig = NodeConfig::encode(NodeConfig::full());

        // if this is not a leaf, nothing to do!
        if (!is_leaf(ncode)) {
            return;
        }

        // insert the node with default full config
        mNodes.set(ncode, fconfig);

        // get the corners of this node
        get_bounds_node(ncode, norigin, nsize);
        get_node_corners(norigin, nsize, ncorners, ncornersPts);

        // get all valid split values
        TypeSplitter::split_node(nsize, ncornersPts, mVertices, svalues);
        add_split_vertices(ncornersPts, svalues, mVertices);
    }


    void
    split_node_adaptive_for_axes(const TypeLocCode ncode, const EnumAxes split_axes) {

        // static variables used in this function
        static TypeChildFlag cflags_cur = 0;
        static TypeChildFlag cflags_new = 0;

        // if this node is already full refined, nothing to do!
        cflags_cur = get_node_cflags(ncode).value();
        if (NodeConfig::is_full(cflags_cur))
            return;

        // given the existing config and requested splitting, find out what config should be created
        cflags_new = NodeConfig::split_along_axes(cflags_cur, split_axes).value();

        // already exists, nothing to do!
        if (cflags_cur == cflags_new)
            return;

        if (!m_enableImproperNodes) {
            cflags_new = NodeConfig::fill_vacuum_complete(cflags_new).value();
        }

        // now actually update the node (and split the node correspondingly)
        update_node(ncode, cflags_new);

#ifdef AMM_DEBUG_TRACK_CONFIGS
        NodeConfig::insert_config(mconfigs_create, nxt_child, cflags_cur, cflags_new);
#endif
    }


    void
    split_node_adaptive_for_child(const TypeLocCode ncode, const TypeChildId child) {

        // static variables used in this function
        static TypeChildFlag cflags_cur = 0;
        static TypeChildFlag cflags_new = 0;

        // if this node is already full refined, nothing to do!
        cflags_cur = get_node_cflags(ncode).value();
        if (NodeConfig::is_full(cflags_cur))
            return;

        // figure out the child flags that create the requested child
        cflags_new = NodeConfig::create_standard_child(cflags_cur, child).value();

        // already exists, nothing to do!
        if (cflags_cur == cflags_new)
            return;

        if (!m_enableImproperNodes) {
            cflags_new = NodeConfig::fill_vacuum_complete(cflags_new).value();
        }

        // now actually update the node (and split the node correspondingly)
        update_node(ncode, cflags_new);

#ifdef AMM_DEBUG_TRACK_CONFIGS
        NodeConfig::insert_config(mconfigs_create, nxt_child, cflags_cur, cflags_new);
#endif
    }


    //! update a node to create a new configuration
    bool
    update_node(const TypeLocCode &_, const TypeChildFlag &cflags_req, bool fill_vacuum = false) {

        // check if the requested configuration is correct
        AMM_error_logic(!m_enableImproperNodes && NodeConfig::has_vacuum(cflags_req),
                        "Requested config is invalid as it contains vacuum!\n");

        const TypeChildFlag cflags_cur = get_node_cflags(_).value();

#ifdef AMM_DEBUG_TRACK_CONFIGS
        NodeConfig::insert_config(mconfigs, cflags_cur);
        NodeConfig::insert_config(mconfigs, cflags_req);
#endif

        // nothing to do
        if (NodeConfig::is_full(cflags_cur))    return false;
        if (cflags_cur == cflags_req)           return false;


        // get the changes in child flags!
        const TypeChildFlag removed = ( cflags_cur & ~cflags_req);
        const TypeChildFlag created = (~cflags_cur &  cflags_req);
        const TypeChildFlag vac = NodeConfig::fill_vacuum_complete(cflags_cur).value() & ~cflags_cur;

        AMM_error_logic(!m_enableImproperNodes && cflags_cur > 0 && vac > 0,
                        "AdaptiveTree.update_node() cannot have vacuum for nonleaf!\n");

        if ((NodeConfig::full() & removed) > 0) {
            AMM_log_error << " >> current "; NodeConfig::print(cflags_cur);
            AMM_log_error << " >> request "; NodeConfig::print(cflags_req);
            AMM_error_logic(true, "AdaptiveTree.update_node() is trying to erase a regular child!\n");
        }

        // use static variables to avoid allocating new memory for each invocation
        static TypeVertex norigin, nsize;
        static TypeVertex rorigin, rsize;
        static std::vector<TypeValue> svalues (snSplits);
        static std::vector<TypeIndex> ncorners (snCorners);
        static std::vector<TypeIndex> rcorners (snCorners);
        static std::vector<TypeVertex> ncornersPts (snCorners);
        static std::vector<TypeVertex> rcornersPts (snCorners);

        // get the bounds of this node
        get_bounds_node(_, norigin, nsize);
        get_node_corners(norigin, nsize, ncorners, ncornersPts);

        // which vertex manager will we be updating?
        auto &_vertices = (fill_vacuum) ? mVertices_Improper : mVertices;

        // --------------------------------------------------------------------------------
        // case 1: splitting a leaf (empty) node,
        // use the split points of the parent node to add corners of the child nodes
        if (cflags_cur == 0) {

            // if we're not filling vacuum, we will add the split points
            if (!fill_vacuum) {

                // figure out which splits are needed
                TypeSplitFlag splits_to_add = get_split_flags(created);

                // add the split points of this node
                TypeSplitter::split_node(nsize, ncornersPts, mVertices, svalues);
                add_split_vertices(ncornersPts, svalues, _vertices, splits_to_add);
            }
        }

        // --------------------------------------------------------------------------------
        // case 2: creating a node out of vacuum
        // use the corners of the parent node to create new splits
        else if (removed == 0)  {

            // figure out which splits are needed
            TypeSplitFlag splits_to_add = get_split_flags(created);

            // add the split points of this node
            TypeSplitter::split_node(nsize, ncornersPts, mVertices, svalues);
            add_split_vertices(ncornersPts, svalues, _vertices, splits_to_add);
        }

        // --------------------------------------------------------------------------------
        // case 3: remove some nonstandard children
        // use the split points of the removed children to add corners of their components
        else {

            TypeChildFlag final_to_create = 0;

            // for each removed rectangular child, create its components sibling nodes
            for(TypeChildId removed_child = snChildren_nonstandard-1; removed_child >= snChildren; removed_child--) {

                if(!AMM_is_set_bit32(removed, removed_child))
                    continue;

                const TypeChildFlag created_comps = created & NodeConfig::get_node_components(removed_child).value();
                const TypeLocCode rcode = LocCode::child(_, removed_child);

                if (!fill_vacuum) {

                    // get the corners of the removed child!
                    rorigin = norigin;
                    rsize = nsize;
                    AMTree::get_bounds_child(removed_child, rorigin, rsize);
                    get_node_corners(rorigin, rsize, rcorners, rcornersPts);

                    // compute the split points with respect to the removed node!
                    TypeSplitter::split_node(rsize, rcornersPts, mVertices, svalues);

                    // transform them to with respect to the parent node!
                    TypeSplitter::offset_longnode(removed_child, svalues);

                    // figure out which splits are needed
                    TypeSplitFlag splits_to_add = AMTree::get_split_flags(created);

                    // add the splits
                    add_split_vertices(ncornersPts, svalues, _vertices, splits_to_add);
                }

                // now add the nodes
                final_to_create |= created_comps;
            }
        }

        // --------------------------------------------------------------------------------
        // update the actual nodes (unless we are filling vacuum)
        if (!fill_vacuum) {
            const TypeConfig config_req = NodeConfig::encode(cflags_req);
            mNodes.set(_, config_req);
        }
        return true;
    }


    void
    aggregate_vertex_updates_pushdown(const TypeLocCode &_, const TypeVertex &nbox0, const TypeVertex &nsize,
                                       TypeInterpolator *parent_minterp = nullptr) {

        static std::vector<TypeIndex> nodeCorners(snCorners);
        static std::vector<TypeValue> nodeCornerValues(snCorners);
        static TypeValue val_delta = 0;
        static TypeValue val_parent = 0;
        static bool exists_at_lvl = false;

        // level of this node
        const TypeScale nlevel = LocCode::level(_);

        // phase 1: update the corners of this node
        get_node_corners(nbox0, nsize, nodeCorners);

        // this is a root if there is no interpolator given
        const bool parent_available = (parent_minterp != nullptr);

        if (!parent_available) {

            for(TypeCornerId i = 0; i < snCorners; i++) {

                const TypeIndex &cid = nodeCorners[i];
                val_delta = get_staged_vertex(cid, nlevel, exists_at_lvl);

                // if this vertex update exists at this level but hasnt been added yet!
                if (exists_at_lvl) {
                    if(!is_vertex_cached_at_lvl(cid, nlevel)) {
                        cache_vertex_at_lvl(cid, nlevel);
                        aggregate_vertex(cid, val_delta);
                    }
                }
                nodeCornerValues[i] = val_delta;
            }
            unstage_node(_);
        }

        // else, we want to use the parent node's interpolator
        else {

            for(TypeCornerId i = 0; i < snCorners; i++) {

                const TypeIndex &cid = nodeCorners[i];
                val_delta = get_staged_vertex(cid, nlevel, exists_at_lvl);

                // val_parent contains the vertex update up until the parent!
                val_parent = parent_minterp->compute(idx2p(cid));

                // aggregate the values from the parent's update
                // add parent's update (tracked by cache)
                if (!is_vertex_cached(cid)) {
                    cache_vertex(cid);
                    aggregate_vertex(cid, val_parent);
                }

                // aggregate current level's update
                // if this vertex was updated at this level but hasn't been added yet!
                if (exists_at_lvl && !is_vertex_cached_at_lvl(cid, nlevel)) {
                    cache_vertex_at_lvl(cid, nlevel);
                    aggregate_vertex(cid, val_delta);

                }

                nodeCornerValues[i] = AMM_add_potential_missing(val_parent, val_delta);
            }
            unstage_node(_);
        }

        // nothing left to do for leaf!
        auto niter = mNodes.find(_);
        if (niter == mNodes.end()) {
            return;
        }

        // phase 2: create a multilinear interpolator and recurse down
        TypeInterpolator miter = TypeInterpolator (idx2p(nodeCorners.front()),
                                                   idx2p(nodeCorners.back()),
                                                   nodeCornerValues);

        // need to go over all the children that exist
        const TypeChildFlag cflag = NodeConfig::decode(niter->second).value();

        // (cannot do static variables for recursive function)
        TypeVertex cbox0, csize;
        std::vector<TypeIndex> childCorners (snCorners);

        for(TypeCornerId child_id = 0; child_id < NodeConfig::snChildren_nonstandard; child_id++) {

            if (!AMM_is_set_bit32(cflag, child_id))
                continue;

            cbox0 = nbox0;
            csize = nsize;
            AMTree::get_bounds_child(child_id, cbox0, csize);
            get_node_corners(cbox0, csize, childCorners);

            aggregate_vertex_updates_pushdown(LocCode::child(_, child_id), cbox0, csize, &miter);
        }
    }


    //! create leaves
    void
    create_leaves(const TypeLocCode &node, const TypeChildFlag &cflags, NodeSet &leaves) const {

        static const TypeChildId nchildren =
                m_enableRectangularNodes ? snChildren_nonstandard : snChildren;

        for(TypeChildId child_id = 0; child_id < nchildren; child_id++) {

            if (!AMM_is_set_bit32(cflags, child_id))
                continue;

            const TypeLocCode ccode = LocCode::child(node, child_id);
            if (!is_node(ccode)) {
                leaves.insert(ccode);
            }
        }
    }


    //! create the boundary recursively by finding nodes that contain the boundary
    void
    create_boundary(const TypeVertex &db1, const TypeLocCode &ncode, const TypeVertex &nbox0, const TypeVertex &nsize) {

            static const TypeVertex &db0 = mOrigin;

            // recursive function: do not do static!
            TypeVertex p;
            TypeVertex nbox1;
            TypeVertex tbox0, tbox1, tsize;
            std::vector<TypeIndex> ncorners(snCorners);
            std::vector<TypeValue> ncorner_values(snCorners);

            // compute the bbox1
            OctUtils::os2bb(nbox0, nsize, nbox1);

            // if this is a leaf, then this must be trimmed
            if (is_leaf(ncode)) {

                tbox0 = nbox0;
                tbox1 = nbox1;
                uint8_t ntrims = OctUtils::trim_bounds(tbox0, tbox1, db0, db1);
                if (ntrims == 0 || ntrims > Dim) {
                    return;
                }

                mLeaves_at_bdry.insert(ncode);

                // create a multilinear interpolator using the actual corners
                get_node_corners(nbox0, nsize, ncorners, ncorner_values);

                TypeInterpolator minterp = TypeInterpolator (nbox0, nbox1);

                // now, created trimmed corners
                OctUtils::bb2s(tbox0, tbox1, tsize);
                get_node_corners(tbox0, tsize, ncorners);

                // the first corner is always the origin and cannot exist on boundary!
                for(TypeCornerId i = 1; i < snCorners; i++) {

                    const TypeIndex &pidx = ncorners[i];
                    p = idx2p(pidx);

                    if (OctUtils::is_on_boundary(p, db1)) {
                        mVertices_at_bdry.set(pidx, minterp.compute(p, ncorner_values));
                    }
                }
                return;
            }

            // else, check which child contains the boundary
            const TypeChildFlag cflags = get_node_cflags(ncode).value();
            for(TypeChildId child_id = 0; child_id < NodeConfig::snChildren_nonstandard; child_id++) {

                if (!AMM_is_set_bit32(cflags, child_id))
                    continue;


                tbox0 = nbox0;
                tsize = nsize;
                get_bounds_child(child_id, tbox0, tsize);
                OctUtils::os2bb(tbox0, tsize, tbox1);

                if (OctUtils::is_across_boundary(tbox0, tbox1, db1)) {
                    create_boundary(db1, LocCode::child(ncode, child_id), tbox0, tsize);
                }
            }
        }


    //! reconstruction
    void
    reconstruct(const TypeLocCode &ncode, const bool &is_vacuum,
                 const TypeVertex &dbox0, const TypeVertex &dsize,
                 ListFinalVerts &vertices, ListFinalCells &cells) const {

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);

        // static variables for node bounds and corners
        static TypeVertex nbox0, nbox1, nsize;
        static TypeVertex tbox0, tbox1, tsize;
        static std::vector<TypeIndex> ncorners(snCorners);
        static std::vector<TypeValue> ncorner_values(snCorners);
        static std::vector<TypePrecision> ncorner_precs(snCorners);

        // get the bounds of this node!
        get_bounds_node(ncode, nbox0, nsize);
        OctUtils::os2bb(nbox0, nsize, nbox1);

        // trim the node to match the requested domain
        tbox0 = nbox0;
        tbox1 = nbox1;
        uint8_t ntrims = OctUtils::trim_bounds(tbox0, tbox1, dbox0, dbox1);

        // node is completely out of domain
        if (ntrims > Dim)   return;


        // use node's corners to create the corner values
        get_node_corners(nbox0, nsize, ncorners);
        for(TypeCornerId i = 0; i < snCorners; i++) {
            ncorner_values[i] = get_vertex_value(ncorners[i], is_vacuum);
            ncorner_precs[i] = get_vertex_precision(ncorners[i], is_vacuum);
        }

        //! overwrite the corners and values with trimmed corners and values
        if (ntrims > 0) {

            //! and an interpolator for the node
            TypeInterpolator minterp (nbox0, nbox1, ncorner_values);

            OctUtils::bb2s(tbox0, tbox1, tsize);
            get_node_corners(tbox0, tsize, ncorners);

            for(TypeCornerId i = 0; i < snCorners; i++) {
                ncorner_values[i] = minterp.compute(idx2p(ncorners[i]));
            }
        }

        //! collect the data
        for(TypeCornerId i = 0; i < snCorners; i++) {
            vertices.insert(std::make_pair(ncorners[i],
                                            TypeFinalVert(ncorner_values[i], ncorner_precs[i])));
        }


        uint8_t ntype = NodeConfig::node_type(LocCode::childId(ncode));
        TypeChildId ctype = childId(nbox0, nsize);

        if (is_vacuum) {
            ntype += Dim;
        }
        cells.push_back(TypeFinalCell(ncorners, LocCode::level(ncode), ntype, ctype));
    }


    void
    reconstruct(const TypeLocCode &ncode, const bool &consider_vacuum,
                 const TypeVertex &dbox0, const TypeVertex &dsize,
                 std::vector<TypeValue> &func,
                 size_t &nmismatches_on_corners,
                 size_t &nmismatches_on_boundary,
                 size_t &nmismatches_inside,
                 std::set<TypeLocCode> &erroneous_nodes, std::set<TypeIndex> &erroneous_verts) const {


        bool verbose = false;
        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);

        //! static variables for node bounds and corners
        static TypeVertex nbox0, nbox1, nsize;
        static TypeVertex tbox0, tbox1;

        static std::vector<TypeIndex> ncorners(snCorners);
        static std::vector<TypeValue> ncorner_values(snCorners);

        static TypeVertex p;
        static TypeIndex pidx = 0;
        static TypeIndex z0 = 0, z1 = 1;

        // get the bounds of this node!
        get_bounds_node(ncode, nbox0, nsize);
        OctUtils::os2bb(nbox0, nsize, nbox1);

       // check if the node is outside the requested domain
        tbox0 = nbox0;
        tbox1 = nbox1;
        uint8_t ntrims = OctUtils::trim_bounds(tbox0, tbox1, dbox0, dbox1);

        //! node totally out of domain
        if (ntrims > Dim)   return;


        //! get the corners and values
        get_node_corners(nbox0, nsize, ncorners);
        for(TypeCornerId i = 0; i < snCorners; i++) {
            ncorner_values[i] = get_vertex_value(ncorners[i], consider_vacuum);
        }

        TypeInterpolator minterp = TypeInterpolator (nbox0, nbox1);

        //! now, the actual stuff
        if (Dim == 3) { z0 = tbox0[2];  z1 = tbox1[2]; }

        for(TypeIndex z = z0;       z <= z1;       z++) {
        for(TypeIndex y = tbox0[1]; y <= tbox1[1]; y++) {
        for(TypeIndex x = tbox0[0]; x <= tbox1[0]; x++) {

            p[0] = x;       p[1] = y;
            if (Dim == 3){  p[2] = z;   }

            pidx = OctUtils::p2idx(p-dbox0, dsize);

            TypeValue &val = func[pidx];
            TypeValue  val_new = minterp.compute(p, ncorner_values, true);

            // if the value does not exist, simply create the value
            if (AMM_AMTREE_IS_INVALID(val)) {  // initialized nan in reconstruct()
                val = val_new;
                continue;
            }

            // if the value exists, it must match!
            if (amm::utils::approximatelyEqual(val, val_new)) {
                continue;
            }

            // otherwise, we have hit an error!
            erroneous_verts.insert(pidx);
            erroneous_nodes.insert(LocCode::parent(ncode));

            // this is a vertex
            if (is_vertex(pidx)) {
                nmismatches_on_corners ++;
                if (verbose) {
                    AMM_log_error << "AMTree._reconstruct() -- mismatch(node corner) " << pidx << " : " << p-dbox0 << " : "
                                  << std::setprecision(12) << val << " != " << val_new << "; err = " << val-val_new << "\n";
                    print_node(ncode);
                }
            }

            // this is something on the boundary
            else if (x == tbox0[0] || y == tbox0[1] || z == tbox0[2] ||
                     x == tbox1[0] || y == tbox1[1] || z == tbox1[2]) {
                nmismatches_on_boundary ++;
                if (verbose) {
                    AMM_log_error << "AMTree._reconstruct() -- mismatch(node boundary) " << pidx << " : " << p-dbox0 << " : "
                                  << std::setprecision(12) << val << " != " << val_new << "; err = " << val-val_new << "\n";
                    print_node(ncode);
                }
            }

            // something else?
            else {
                nmismatches_inside ++;
                if (verbose) {
                    AMM_log_error << "AMTree._reconstruct() -- mismatch(node inside) " << pidx << " : " << p-dbox0 << " : "
                                  << std::setprecision(12) << val << " != " << val_new << "; err = " << val-val_new << "\n";
                    print_node(ncode);
                }
            }
        }}}
    }


    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! static methods
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------

    //! based on child flags, decide which splits should be created
    static inline
    TypeSplitFlag
    get_split_flags(const TypeChildFlag &cflags) {

        TypeSplitFlag splits_flags;
        for(TypeChildId child_id = 0; child_id < snChildren_nonstandard; child_id++) {
            if (!AMM_is_set_bit32(cflags, child_id))
                continue;
            TypeSplitter::set_splits_flags(child_id, splits_flags);
        }
        return splits_flags;
    }


    //! given the bounds of a node, find what child it is
    static inline
    TypeChildId
    childId(const TypeVertex &bb0, const TypeVertex &sz) {

        TypeCoord dsz = 0;
        TypeChildId f = 0;

        // Compute scale and shift bits for each dimension
        for (TypeDim d = 0; d < Dim; d++) {

            dsz = sz[d];
            if (sz[d] == sz.min()) {
                AMM_set_bit32(f, d);
                dsz  = AMM_lsize_dbl(dsz);
            }

            if (bb0[d] % (dsz-1)) {
                AMM_set_bit32(f, Dim+d);
            }
        }

        return NodeConfig::get_child_id_for_bounds_conversion_flag(f);
    }


    //! convert a given node's bounds to its child's/parent's bounds
    static inline
    void
    get_bounds_child(const TypeChildId _, TypeVertex &bb0, TypeVertex &sz) {

        // standard children
        if (_ < snChildren){
            return OctUtils::get_bounds_child(_, bb0, sz);
        }

        // based on the child id, figure out which dimension to expand and/or shift
        TypeDim flags = NodeConfig::get_bounds_conversion_flag_for_child_id(_);

        // first shrink and then shift
        for(TypeDim d = 0; d < Dim; d++) {
            if (AMM_is_set_bit32(flags, d))     sz[d] = AMM_lsize_half(sz[d]);
            if (AMM_is_set_bit32(flags, d+Dim)) bb0[d] = AMM_lsize_dwn(bb0[d], sz[d]);
        }
    }


    static inline
    void
    get_bounds_parent(const TypeChildId _, TypeVertex &bb0, TypeVertex &sz) {

        // standard children
        if (_ < snChildren){
            return OctUtils::get_bounds_parent(_, bb0, sz);
        }

        // based on the child id, figure out which dimension to expand and/or shift
        TypeDim flags = NodeConfig::get_bounds_conversion_flag_for_child_id(_);

        // first shift and then expand
        for(uint8_t d = 0; d < Dim; d++) {
            if (AMM_is_set_bit32(flags, d+Dim)) bb0[d] = AMM_lsize_up(bb0[d], sz[d]);
            if (AMM_is_set_bit32(flags, d))     sz[d] = AMM_lsize_dbl(sz[d]);
        }
    }


public:
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! printing functions
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    inline
    void
    print_memory() const {

        float m = 0.;

        AMM_log_info << "---- AMTree memory ----\n";
        AMM_log_info << "\tmNodes               = " << mNodes.size()     << "    [ " << mNodes.memory_in_kb() << " KB]\n";
        AMM_log_info << "\tmVertices            = " << mVertices.size()  << "    [ " << mVertices.memory_in_kb() << " KB]\n";

        AMM_log_info << "\tmVertices_aggregated = " << mVertices_agg.size() << "    [ " << mVertices_agg.memory_in_kb() << " KB]\n";
        AMM_log_info << "\tmNodes_staged        = " << _nstaged.size()      << "    [ " << _nstaged.memory_in_kb() << " KB]\n";
        AMM_log_info << "\tmVerts_staged        =";
        for (TypeScale i = 0; i <= mTreeDepth; ++i) {
            AMM_log_info << " (" << size_t(i) << ": " << _vstaged[i].size() << ")";
            m += _vstaged[i].memory_in_kb();
        }
        AMM_log_info << "    [ " << m << " KB]\n";

        AMM_log_info << "\tmLeaves_at_bdry      = " << mLeaves_at_bdry.size()      << "    [ " << mLeaves_at_bdry.memory_in_kb() << " KB]\n";
        AMM_log_info << "\tmVertices_at_bdry    = " << mVertices_at_bdry.size()    << "    [ " << mVertices_at_bdry.memory_in_kb() << " KB]\n";
        AMM_log_info << "\tm_vcache             = " << _vcache.size()              << "    [ " << _vcache.memory_in_kb() << " KB]\n";

        m = 0.;
        AMM_log_info << "\tm_vcache_bylvl       =";
        for (TypeScale i = 0; i <= mTreeDepth; ++i) {
            AMM_log_info << " (" << size_t(i) << ": " << _vcache_bylvl[i].size() << ")";
            m += _vcache_bylvl[i].memory_in_kb();
        }
        AMM_log_info << "    [ " << m << " KB]\n";
        AMM_log_info << "---- end AMTree memory ----\n";
    }


    inline
    void
    print_node(const TypeLocCode &_, const TypeChildMask cflags = 0) const {

        static TypeVertex nbox0, nsize;
        get_bounds_node(_, nbox0, nsize);
        OctUtils::print_a_node(_, LocCode::level(_), nbox0, nsize, false);
        std::cout << " :: "; NodeConfig::print(cflags);
    }


    inline
    void
    print_nodes(const NodeSet &nodes, const std::string &label = "leaves", const bool sorted=true) const {

        std::cout << " > printing " << label << " [= " << nodes.size() << "]!\n";
        if (!sorted) {
            for(auto iter = nodes.begin(); iter != nodes.end(); ++iter){
                print_node(*iter);
            }
        }
        else {
            auto snodes = nodes.as_sorted_vector();
            for(auto iter = snodes.begin(); iter != snodes.end(); ++iter){
                print_node(*iter);
            }
        }
    }


    inline void
    print_nodes(const NodeMap &nodes, const std::string &label = "nodes", const bool sorted=true) const {

        std::cout << " > printing " << label << " [= " << nodes.size() << "]!\n";
        if (!sorted) {
            for(auto iter = nodes.begin(); iter != nodes.end(); ++iter){
                print_node(iter->first, iter->second);
            }
        }
        else {
            auto snodes = nodes.as_sorted_vector();
            for(auto iter = snodes.begin(); iter != snodes.end(); ++iter){
                print_node(iter->first, iter->second);
            }
        }
    }


    inline
    void
    print_nodes(const NodeMap_Stg &nodes, const std::string &label = "nodes", const bool sorted=true) const {

        std::cout << " > printing " << label << " [= " << nodes.size() << "]!\n";
        if (!sorted) {
            for(auto iter = nodes.begin(); iter != nodes.end(); ++iter){
                print_node(iter->first, iter->second);
            }
        }
        else {
            auto snodes = nodes.as_sorted_vector();
            for(auto iter = snodes.begin(); iter != snodes.end(); ++iter){
                print_node(iter->first, iter->second);
            }
        }
    }


    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    //! some testing functions on vertices
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    // check if we created any zero vertices
    inline
    bool
    test_vertices_no_zero() const {

        size_t nzero = std::count_if(mVertices.begin(), mVertices.end(),
                                     [](auto v) { return AMM_is_zero(v.second); });

        if (nzero > 0) {
            AMM_log_error << "Error: Found " << nzero << " out of " << mVertices.size()
                          << " vertices to have zero value!\n";
        }
        return (nzero == 0);
    }


    // compare vertices to a given function
    inline
    bool
    test_vertices_against_function(const std::vector<TypeValue> &func, bool verbose=true) const {

        TypeValue max_err = 0;
        size_t nmismatch = 0;

        amm::timer t;
        AMM_log_info << "Validating mesh vertices (= "<<mVertices.size()<<") against given function...";
        fflush(stdout);

        auto biter = mVertices.begin();
        auto eiter = mVertices.end();
        for(auto iter = biter; iter != eiter; ++iter) {

            //const TypeIndex &i = idx2p(mVertices.unhash(iter->first));
            const TypeIndex &i = iter->first;
            const TypeValue &v = iter->second;
            const TypeValue &fv = func[i];

            if (amm::utils::approximatelyEqual(v,fv)) {
                continue;
            }

            if (verbose) {
                if (nmismatch==0)
                    AMM_logc_error << "\n";
                AMM_log_error << " mismatch in vertex " << i << " : " << idx2p(i) << " : ";
                AMM_log_error << std::fixed << std::setprecision (std::numeric_limits<TypeValue>::digits10 + 1)
                              << "mesh = "<< v << " != true = " << fv << " : " << (v-fv) << std::endl;
            }

            max_err = std::max(max_err, TypeValue(fabs(v-fv)));
            nmismatch ++;
        }

        t.stop();
        if (nmismatch == 0) {
            AMM_logc_info << " found success!" << t << "\n";
            return true;
        }

        AMM_logc_info << " found failure! " << t << "\n";
        AMM_log_error << "Error: Found " << nmismatch << " out of " << mVertices.size()
                      << " vertices to have mismatching values!"
                      << std::fixed << std::setprecision (std::numeric_limits<TypeValue>::digits10 + 1)
                      << " max error = " << max_err << std::endl;

        return false;
    }


    // compare function against function
    inline
    bool
    test_function_against_function(const std::vector<TypeValue> &func, const TypeVertex &domainSize, bool verbose=false) const {

        static constexpr TypeValue tzero (0);
        static std::vector<TypeValue> m_ofunc;
        static std::vector<TypeValue> m_oerror;

        const size_t m_nTreePoints = OctUtils::product(mTreeSize);

        if (m_ofunc.size() != m_nTreePoints) {  m_ofunc.resize(m_nTreePoints, tzero);   }
        if (m_oerror.size() != m_nTreePoints) { m_oerror.resize(m_nTreePoints, tzero);  }

        // reconstruct the mesh functions
        bool success = reconstruct(TypeVertex(TypeCoord(0)), mTreeSize, m_ofunc);

        amm::timer t;
        AMM_log_info << "Validating mesh function against given function...";
        fflush(stdout);

        // compute absolute errors!
        size_t _dims[3] = {domainSize[0], domainSize[1], 1};
        if (3 == Dim) {   _dims[2] = domainSize[2];  }

        for(size_t z = 0; z < _dims[2]; z++) {
        for(size_t y = 0; y < _dims[1]; y++) {
        for(size_t x = 0; x < _dims[0]; x++) {
            const size_t idx = AMM_xyz2grid(x,y,z,_dims);

            m_oerror[idx] = func[idx] - m_ofunc[idx];
            success = success && amm::utils::approximatelyEqual(func[idx], m_ofunc[idx]);
        }}}

        t.stop();

        if(success) {
            AMM_log_info << " found success! " << t << "\n";
            return true;
        }

        AMM_log_info << " found failure! " << t << "\n";

        // else, print the error info and write the errors
        auto erng = amm::utils::frange(m_oerror);
        auto mit = std::distance(m_oerror.begin(), std::max_element(m_oerror.begin(), m_oerror.end()));

        AMM_log_error << " found error in mesh function: ("
                      << std::fixed << std::setprecision (std::numeric_limits<TypeValue>::digits10 + 1)
                      << erng.first <<"," << erng.second<<")" << std::endl
                      << " max error at " << mit << " : " << idx2p(mit) << " = " << m_oerror[mit]
                      << " : mesh = " << m_ofunc[mit] << ", lfnc = " << func[mit] << std::endl;

        if (verbose) {
          AMM_log_error << "Saving error to (amm_func_error.raw)\n";
          amm::utils::write_binary(m_oerror, "amm_func_error.raw");
        }
        return success;
    }


    // compare two sets of vertices
    bool test_vertices_against_vertices(const VManager &a, const std::string &aname,
                                        const VManager &b, const std::string &bname) const {

        bool failed = false;
        if (a.size() != b.size()) {
            std::cout << "Error: mismatch in size: " << aname << " = " << a.size() << ", " << bname << " = " << b.size() << "\n";
            return false;
        }

        // compare b against a
        auto biter = a.begin();
        auto eiter = a.end();
        for(auto iter = biter; iter != eiter; iter++) {

            const TypeIndex &idx = iter->first;
            const TypeValue &val = iter->second;

            if (!b.contains(idx)) {
                AMM_log_error << " idx " << idx << " = " << idx2p(idx) << " not found in " << bname << "! value in " << aname << " = " << val << "\n";
                continue;
            }

            const TypeValue &val_b = b.get(idx);
            if (!AMM_is_tolerable(val-val_b)) {
                AMM_log_error << " value mismatch ("<<aname<<" = "<<val<<" != "<<bname<<" = "<<val_b<<") [need to add "<<(val-val_b)<<"] in "<<bname<<" for " << idx2p(idx) << "!\n";
                failed = true;
            }
        }
        return !failed;
    }


//! ------------------------------------------------------------------------------------
//! ------------------------------------------------------------------------------------
//!  THE END!
//! ------------------------------------------------------------------------------------
//! ------------------------------------------------------------------------------------













    //! ------------------------------------------------------------------
    //! create a cell in the given bounds
    //!
    //! the cell has to be created as a set of tree nodes (align with the
    //! hierarchy). The nodes corresponding to this cell must exist fully
    //! inside the cell unless the cell sits along the domain boundary.
    //! In this case, we allow nodes span across the cell boundary.
    //!
    //! @param cbb0:            cell min bounds
    //! @param cbb1:            cell max bounds
    //! @param corner_values:   values at cell corners
    //!
    //! ------------------------------------------------------------------
public:
    inline
    void create_cell(const TypeVertex &cbb0, TypeVertex cbb1,
                     const std::vector<TypeValue> &corner_values) {

        // @TODO: need to test this function
        AMM_error_invalid_arg(corner_values.size() != snCorners, "Incorrect number of values");

        static TypeVertex csz;
        static std::vector<TypeIndex> nodeCornerIdxs(snCorners);
        static std::vector<TypeVertex> nodeCornerPts(snCorners);

        // Get cell size
        OctUtils::bb2s(cbb0, cbb1, csz);

        /*std::cout << "amtree::create_cell: [" << cbb0 << ", " << cbb1 << "],"
                  << " sz = " << csz << ", "
                  << OctUtils::is_valid_node_sz(csz, m_enableRectangularNodes) << " --- "
                  << m_enableRectangularNodes ;
        std::cout<<"\n";*/

        // --------------------------------------------------------------------
        // for a valid node, we will simply create the node and add it's corners
        if (OctUtils::is_valid_node_sz(csz, m_enableRectangularNodes)) {

            create_node(cbb0, csz);
            get_node_corners(cbb0, csz, nodeCornerIdxs);

            TypeCornerId i = 0;
            for(auto citer = nodeCornerIdxs.begin(); citer != nodeCornerIdxs.end(); citer++, i++) {
                mVertices.set(*citer, corner_values[i]);
            }
        }

        // --------------------------------------------------------------------
        else {
            TypeInterpolator miter (cbb0, cbb1, corner_values);

            // now expand the cell
            OctUtils::expand_node(csz, m_enableRectangularNodes);
            OctUtils::os2bb(cbb0, csz, cbb1);

            create_node(cbb0, csz);
            OctUtils::get_node_corners(cbb0, cbb1, nodeCornerPts);

            // TODO: we need to compute only the vertices that have changed
            for (TypeCornerId i = 0; i < snCorners; i++) {
                mVertices.set(p2idx(nodeCornerPts[i]), miter.compute(nodeCornerPts[i]));
            }
        }

        // --------------------------------------------------------------------
        return;


#if 0
        // this is older code attempted for arbitrary (non-amm) cells
        //! split the requested bounds into tree nodes
        std::vector<std::pair<TypeIndex, TypeIndex>> nodes;
        std::cout << "split_cell_along_tree\n";
        split_cell_along_tree(cbb0, csz, nodes);

        //! create an interpolator for this cell
        TypeInterpolator miter (cbb0, cbb1, corner_values);
        static std::vector<TypeIndex> nodeCorners(snCorners);

        //! use a cache to prevent duplicate vertex setting
        VertexSet cache;

        //! create all these component nodes
        for(auto niter = nodes.begin(); niter != nodes.end(); niter++) {

            std::cout << " doung node: "<< niter->first << "\n";
            const TypeVertex nbox0 = idx2p(niter->first);
            const TypeVertex nsize = OctUtils::bb2s(nbox0, idx2p(niter->second));

            create_node(nbox0, nsize);
            get_node_corners(nbox0, nsize, nodeCorners);

            for(auto citer = nodeCorners.begin(); citer != nodeCorners.end(); citer++) {

                const TypeIndex &cidx = *citer;

                if (cache.contains(cidx))
                    continue;

                set_vertex(cidx, miter.compute(idx2p(cidx)));
                cache.insert(cidx);
            }
        }
#endif
    }


    //! create node with given bounds
    inline
    TypeLocCode
    create_node(const TypeVertex &nbox0, const TypeVertex &nsize) {

        AMM_error_invalid_arg(!OctUtils::is_valid_node_sz(nsize, true),
                              "Wrong size of node %s", nsize.c_str());

        // @TODO: need to test this function

        // if this is a square node, we can create it directly.
        // if the node has any dimension smaller than 2,
        // it can not be created directly (undefined center).
        if (OctUtils::is_square(nsize) && (nsize[0] > 2)) {

            static TypeVertex cell_center;
            OctUtils::os2c(nbox0, nsize, cell_center);
            return create_node_at(cell_center);
        }

        // else, create its parent node and split it
        else {

            // compute child id
            const TypeChildId cid = childId(nbox0, nsize);

            // compute parent node
            static TypeVertex parent_bb0, parent_sz, parent_center;
            parent_bb0 = nbox0;
            parent_sz = nsize;

            get_bounds_parent(cid, parent_bb0, parent_sz);
            OctUtils::os2c(parent_bb0, parent_sz, parent_center);

            // construct parent
            const TypeLocCode lcode = create_node_at(parent_center);

            // split along child's axis
            const EnumAxes split_axes = m_enableRectangularNodes ?
                        NodeConfig::get_split_axes(cid) :
                        EnumAxes(AMM_pow2(Dim)-1);

            split_node(lcode, split_axes);
            return LocCode::child(lcode, cid);
        }
    }


















#if 0


    //! ----------------------------------------------------------------------------
    //! write configuration flags to file
    //! ----------------------------------------------------------------------------
    void dump_configs() {
#ifdef AMM_DEBUG_TRACK_CONFIGS
        for (auto iter = mInternalNodes.begin(); iter != mInternalNodes.end(); iter++) {
            NodeConfig::insert_config(mconfigs, NodeConfig::decode(iter->second).value());
        }
        for (auto iter = mNodes_Improper.begin(); iter != mNodes_Improper.end(); iter++) {
            NodeConfig::insert_config(mconfigs, NodeConfig::decode(mInternalNodes.find(*iter)->second).value());
        }


        NodeConfig::update_configs(mconfigs, m_enableImproperNodes);
        //NodeConfig::update_encodings(mconfigs_create, "create");
        //NodeConfig::update_encodings(mconfigs_split, "split");
#endif

    }



















/*
    inline void
    print_summary() const {

        const size_t nnodes = mNodes.size();
        const size_t nverts = mVertices.size();
        const size_t snode = sizeof(TypeLocCode) + sizeof(TypeConfig);
        const size_t svert = sizeof(TypeIndex) + sizeof(TypeValue);

        std::cout << " Octree has:\n"
                  << "\t" << nnodes << " nodes! ("<< float(nnodes*snode)/1024. << " KB)\n"
                  << "\t" << nverts << " vertices! ("<< float(nverts*svert)/1024. << " KB)\n";

        return;
        print_nodes(mNodes, "nodes" , true);
        mVertices.print("verts", true);
    }*/

    //inline size_t nnodes() const {             return mNodes.size();          }
    //inline size_t nvertices() const {          return mVertices.size();          }
    //inline size_t nvertices_bdry() const {     return mVertices_at_bdry.size();  }


    /*inline
    bool
    is_parent_to(const TypeLocCode &_, const TypeChildId &cid) const {
        if (is_leaf(_))       return false;
        return is_leaf(LocCode::child(_, cid));
    }*/


    //inline size_t nnodes_internal() const {     return mNodes.size();       }
    //inline size_t nnodes_vacuum() const {       return mNodes_Improper.size();      }
    //inline size_t nleaves_vacuum() const {      return mLeaves_Improper.size();     }
    //inline size_t nvertices_vacuum() const {    return mVertices_Improper.size();   }

    //inline size_t nnodes() const {  return nleaves() + nnodes_internal();  }
/*
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    inline size_t nnodes_mesh(size_t &type0, size_t &type1, size_t &type2) const {

        printf("amtree.nnodes_mesh is deprecated due to leaves!\n");
        exit(1);
#if 0
        type0 = 0;
        type1 = 0;
        type2 = 0;

        //! count normal leaves
        for (auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); iter++) {
            const TypeLocCode &ncode = *iter;
            if (!is_node_intersects(ncode, mOrigin, dBox1))
                continue;

            uint8_t ntype = NodeConfig::node_type(LocCode::childId(ncode));
            switch(ntype) {
            case 0: type0++;    break;
            case 1: type1++;    break;
            case 2: type2++;    break;
            default: std::cerr << "incorrect node type("<<int(ntype)<<") while reading leaves!\n"; exit(1);
            }
        }

        //! count vacuum leaves
        for (auto iter = mLeaves_Improper.begin(); iter != mLeaves_Improper.end(); iter++) {
            const TypeLocCode &ncode = *iter;
            if (!is_node_intersects(ncode, mOrigin, dBox1))
                continue;

            uint8_t ntype = NodeConfig::node_type(LocCode::childId(ncode));
            switch(ntype) {
            case 0: type0++;    break;
            case 1: type1++;    break;
            case 2: type2++;    break;
            default: std::cerr << "incorrect node type("<<int(ntype)<<") while reading vacuum leaves!\n"; exit(1);
            }
        }

        return type0+type1+type2;
#endif
        return 0;
    }
    inline size_t nvertices_mesh() const {

        size_t cnt = 0;
        for (auto iter = mVertices.begin(); iter != mVertices.end(); iter++) {
            const TypeVertex v = OctUtils::idx2p((*iter).first, mTreeSize);
            if (OctUtils::contains_bb(v, mOrigin, mDomainBox1)){
                cnt++;
            }
        }
        for (auto iter = mVertices_Improper.begin(); iter != mVertices_Improper.end(); iter++) {
            const TypeVertex v = OctUtils::idx2p((*iter).first, mTreeSize);
            if (OctUtils::contains_bb(v, mOrigin, mDomainBox1)){
                cnt++;
            }
        }
        for (auto iter = mVertices_at_bdry.begin(); iter != mVertices_at_bdry.end(); iter++) {
            const TypeVertex v = OctUtils::idx2p((*iter).first, mTreeSize);
            if (OctUtils::contains_bb(v, mOrigin, mDomainBox1)) {
                cnt++;
            }
        }
        return cnt;
    }
*/
    //! --------------------------------------------------------------------------------
    //! --------------------------------------------------------------------------------
    /*inline bool is_node_internal(const TypeLocCode _) const {
        return mNodes.contains(_);
    }
    inline bool is_node(const TypeLocCode _) const {
        return is_leaf(_) || is_node_internal(_);
    }

    inline bool is_parent_to(const TypeLocCode _, const TypeChildId &child_id) const {
        return NodeConfig::child_exists(get_node_cflags(_), child_id);
    }*/



    /*
    void api_print_nodes() const {

        std::map<TypeLocCode, TypeConfig> sortedInternal;
        sortedInternal.insert(mInternalNodes.begin(), mInternalNodes.end());

        std::cout << " > internal nodes " << mInternalNodes.size() << "\n";
        for(auto iter = sortedInternal.begin(); iter != sortedInternal.end(); ++iter){
            print_node(iter->first, NodeConfig::decode(iter->second));
        }

        std::cout << " > leaf nodes " << mLeafNodes.size() << "\n";
        for(auto iter = sbegin_leaf(); iter != send_leaf(); ++iter){
            print_node(*iter, 0);
        }

        if (m_enableImproperNodes) {
        std::set<TypeLocCode> sortedVacuum;
        sortedVacuum.insert(mNodes_Improper.begin(), mNodes_Improper.end());

        std::cout << " > vacuum nodes " << mNodes_Improper.size() << "\n";
        for(auto iter = sortedVacuum.begin(); iter != sortedVacuum.end(); ++iter){
            print_node(*iter, get_node_cflags(*iter));
        }

        sortedVacuum.clear();
        sortedVacuum.insert(mLeaves_Improper.begin(), mLeaves_Improper.end());

        std::cout << " > vacuum leaves " << mLeaves_Improper.size() << "\n";
        for(auto iter = sortedVacuum.begin(); iter != sortedVacuum.end(); ++iter){
            print_node(*iter, 0);
        }
        }
    }

    //! --------------------------------------------------------------------------------
    //! print the complete tree
    //!
    //! verbose = 0: just print the stats
    //! verbose = 1: print counts by level
    //! verbose = 2: print nodes
    //!
    //! --------------------------------------------------------------------------------
    void api_print(int verbose = 0) const {

        printf(LOG::INFO, "\n AdaptiveTree[%d,%d].print(%d)\n", Dim, MaxLevels, verbose);

        if (verbose == 0)
            return;

        if (verbose == 1) {

            const TypeScale nsz = 1+mDataDepth;

#ifdef VERTS_POSTHOC
            printf(LOG::INFO, "  > verts [");
            for(TypeScale i = 0; i < nsz; i++){
                if (mVertices_deltas[i].size() > 0)
                    printf(LOG::INFO, "(%d:%d) ", i, mVertices_deltas[i].size());
            }
            printf(LOG::INFO, "]\n");
#endif
            //! leaf nodes
            std::map<size_t, size_t> lcounts;
            for(auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); iter++ ) {
                size_t nlvl = LocCode::level(*iter);
                if(lcounts.find(nlvl) == lcounts.end())     lcounts[nlvl] = 1;
                else                                        lcounts[nlvl]++;
            }

            printf(LOG::INFO, "  > leaves [");
            for(auto iter = lcounts.begin(); iter!= lcounts.end(); iter++)
                printf(LOG::INFO, "(%d:%d) ", iter->first, iter->second);
            printf(LOG::INFO, "]\n");

            return;
        }

        if (verbose == 2) {
            return;
        }
    }*/

    void* api_serialize() const {}
    void api_deserialize(const void* _) {}

    void api_save(const std::string &_) const {}
    void api_load(const std::string &_) {}

    //! --------------------------------------------------------------------------------
    //! profiling functions
    //! --------------------------------------------------------------------------------
    void profile_prefinalize(amm::amm_profile &_) const {
        _.tree_vert_bylvl = count_staged_vertices();
        _.tree_vertcache = nvcache();
        _.tree_vertcache_bylvl = nvcache_bylvl();
        _.tree_nodecache = nncache();
    }
    void profile_postfinalize(amm::amm_profile &_) const {
        _.mesh_vert = nvertices_mesh();
        _.mesh_cell = nnodes_mesh(_.mesh_cell_by_type[0], _.mesh_cell_by_type[1], _.mesh_cell_by_type[2]);

        _.tree_vert = nvertices();
        _.tree_vert_bdry = nvertices_bdry();
        _.tree_vert_vac = nvertices_vacuum();

        _.tree_node_leaf = nleaves();
        _.tree_node_vacleaf = nleaves_vacuum();
        _.tree_node_int = nnodes_internal();
        _.tree_node_vac = nnodes_vacuum();
        _.tree_node_bdry = nleaves_bdry();
        _.tree_nodes = _.tree_node_leaf+_.tree_node_vacleaf+_.tree_node_int;

#ifdef AMM_ENABLE_PRECISION
        _.tree_blocks = mVertices.nblocks(_.tree_blocks_at_prec, _.tree_verts_at_prec, _.tree_blocks_with_verts);
#else
        _.tree_blocks = 0;
        _.tree_blocks_at_prec.fill(0);
        _.tree_verts_at_prec.fill(0);
        _.tree_blocks_with_verts.fill(0);
#endif
    }
    void profile_quality(amm::amm_profile &_, const size_t &wcount, const size_t &total_bytes, double &psnr) const {

        _.psnr = psnr;
        _.num_coefficients = wcount;
        _.num_bytes = total_bytes;
    }











    //! ----------------------------------------------------------------------------
    //! ----------------------------------------------------------------------------
    bool test_nodes() const {

        printf("amtree.test_nodes is deprecated due to leaves!\n");
        exit(1);
        /*
        //! test 1: leaf nodes should not be internal
        //! test 2: a leaf's parents should be internal
        //! test 3: a leaf may not have any children
        size_t failed1 = 0;
        size_t failed2 = 0;
        size_t failed3 = 0;

        for(auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); iter++) {

            const TypeLocCode &ncode = *iter;

            if (is_node_internal(ncode))
                failed1++;

            if (!is_node_internal(LocCode::parent(ncode)))
                failed2++;

            for (TypeChildId i = 0; i < NodeConfig::snChildren_nonstandard; i++) {
                if (is_node(LocCode::child(ncode, i)))
                    failed3++;
            }
        }

        if (failed1 > 0) {
            std::cout << " incorrect nodes: " << failed1 << " out of " << mLeafNodes.size() << " leaf nodes are also internal nodes!\n";
        }
        if (failed3 > 0) {
            std::cout << " incorrect nodes: " << failed3 << " out of " << mLeafNodes.size() << " leaf nodes have children nodes!\n";
        }

        //! test 4: internal nodes should not be leaf
        //! test 5: internal nodes' config should be valid
        //! test 6: internal nodes' configs should lead to existing children
        size_t failed4 = 0;
        size_t failed5 = 0;
        size_t failed6 = 0;
        for(auto iter = mInternalNodes.begin(); iter != mInternalNodes.end(); iter++) {

            const TypeLocCode &ncode = iter->first;

            if (is_leaf(ncode))
                failed4++;

            const TypeChildFlag c = NodeConfig::decode(iter->second);
            if (!NodeConfig::is_valid_node(c))
                failed5++;

            for (TypeChildId i = 0; i < NodeConfig::snChildren_nonstandard; i++) {
                if ( AMM_is_set_bit32(c, i) != is_node(LocCode::child(ncode, i))) {
                    std::cout << " failed : " << (long long)ncode << " : " << int(i) << std::endl;
                    print_node(ncode);
                    failed6++;
                }
            }
        }

        if (failed4 > 0) {
            std::cout << " incorrect nodes: " << failed4 << " out of " << mInternalNodes.size() << " internal nodes are also leaf nodes!\n";
        }

        if (failed5 > 0) {
            std::cout << " incorrect nodes: " << failed5 << " out of " << mInternalNodes.size() << " internal nodes have invalid configs!\n";
        }

        if (failed6 > 0) {
            std::cout << " incorrect nodes: " << failed6 << " out of " << mInternalNodes.size() << " internal nodes have conflicting children information!\n";
        }

        return (failed1 == 0 && failed2 == 0 && failed3 == 0 && failed4 == 0 && failed5 == 0 && failed6 == 0);
        */
        return true;
    }



    //! ----------------------------------------------------------------------------
    //! compute histogram of function values with n bins and print output
    //! ----------------------------------------------------------------------------
    void histogram(const size_t &nbins) const {

        /*
        printf(LOG::INFO, " compute histogram with %d bins!\n", nbins);

        const VMap_FinalV &verts = mVertices.mData;
        const NodeSet &cells = mLeafNodes;

        //! use the function range to create bins
        auto frng = std::minmax_element(verts.begin(), verts.end(), vvalue_lessthan);
        const TypeValue hmin = frng.first->second;
        const TypeValue hmax = frng.second->second;
        const TypeValue bin_width = (hmax-hmin) / TypeValue(nbins);
        const TypeValue oobw = 1.0 / bin_width;

        //! bin edges
        std::vector<TypeValue> bin_edges (nbins+1, 0.0);
        bin_edges[0] = hmin;
        for(size_t b = 0; b < nbins; b++) {
            bin_edges[b+1] = bin_edges[b]+bin_width;
        }

        //! the histogram
        std::vector<size_t> histo (nbins, 0);

        printf(LOG::INFO, "  overall range = %lf : %lf : %lf\n", hmin, hmax, bin_width);

        //! now go over each cell to populate the histogram
        TypeVertex nbox0, nsize;
        IndexVec ncorners(snCorners);
        ValueVec ncorner_values(snCorners);

        //! to define the multilinear interpolator in this cell
        Vec<Dim, TypeValue> offset, slope;

        for(auto iter = cells.begin(); iter != cells.end(); ++iter) {

            const TypeLocCode &ncode = *iter;

            //! get the bounds of this node!
            foo(ncode, nbox0, nsize);
            get_corners_node_o_osiv(nbox0, nsize, ncorners, ncorner_values);

            //! get the range of this cell
            auto crng = std::minmax_element(ncorner_values.begin(), ncorner_values.end());
            const TypeValue &cmin = *crng.first;
            const TypeValue &cmax = *crng.second;

            if (Dim == 2) {
                slope[0] = (ncorner_values[1] - ncorner_values[0]) ;//! TypeValue(nsize[0]);
                slope[1] = (ncorner_values[3] - ncorner_values[2]) ;//! TypeValue(nsize[1]);

                for(size_t i = 0; i < 4; i++)
                    printf(LOG::INFO, "%s = %lf\n", idx2p(ncorners[i]).c_str(), ncorner_values[i]);
            }

            printf(LOG::INFO, " slope = %s\n", slope.c_str());
            const size_t ccount = nsize.product();
            const size_t max_i = ccount-1;

            //! TODO: deal with proper multilinear interpolation
            const TypeValue &dc = (cmax-cmin) / TypeValue(ccount-1);
            const TypeValue &oodc = 1.0 / dc;

            //! find the smallest and largest bin that this cell intersects with
            const size_t b0 = int(floor((cmin-hmin) * oobw));
            const size_t b1 = int(floor((cmax-hmin) * oobw));

            size_t i0 = 0;
            for(size_t b = b0; b <= b1; b++) {

                const TypeValue &bmin = bin_edges[b];
                const TypeValue &bmax = bin_edges[b+1];

                //! no overlap with the bin
                if (bmin > cmax || bmax < cmin) {
                    printf(LOG::ERROR, " should not be outside range!\n");
                    exit(1);
                }

                //! find the largest index that is smaller than bin max
                size_t i1 = floor((bmax-cmin)*oodc);
                i1 = std::min(i1, max_i);

                if (i1 < i0) {
                    printf(LOG::ERROR, " cannot happen!\n");
                    exit(1);
                }

                histo[b] += (i1-i0);
                i0 = i1;
            }
            exit(1);
        }

        for(size_t b = 0; b < nbins; b++)
            printf(LOG::INFO, " histo: %d in (%lf, %lf)\n", histo[b], bin_edges[b], bin_edges[b+1]);
        */
    }

    //! ----------------------------------------------------------------------------
    //! make continuous
    //! ----------------------------------------------------------------------------
    void make_continuous(const TypeLocCode &_) {

         //TODO: this function should be redesigned using iterators
        AMM_log_debug << "\n make_continuous ";
        print_node(_);
        /*
        TypeVertex nbox0, nsize;
        std::set<TypeIndex> spoints;
        VertexMapSimple spoints_with_values;

        TypeVertex cbox0, csize;
        std::set<TypeIndex> cpoints;
        VertexSet ccorners(snCorners);

        foo(_, nbox0, nsize);
        get_node_splits(nbox0, nsize, spoints);

        const TypeChildFlag cflags = get_node_cflags(_);
        for(TypeChildId i = 0; i < NodeConfig::snChildren_nonstandard; i++) {

            if (!IS_SET_BIT(cflags, i))
                continue;

            const TypeLocCode ccode = LocCode::child(_, i);
            if (!is_leaf(ccode)) {
                printf(LOG::ERROR, " child %d is not a leaf!\n", int(i));
                continue;
            }

            printf(LOG::DEBUG, " do leaf %d\n", int(i));
            print_node(ccode);

            cbox0 = nbox0;
            csize = nsize;
            cpoints.clear();
            ccorners.clear();

            AMTree::bounds_child(i, cbox0, csize);
            get_node_splits(cbox0, csize, cpoints);
            BasicOctree::get_node_corners_s3(p2idx(cbox0), csize, mSize, ccorners);

            cpoints.insert(ccorners.begin(), ccorners.end());

            MLInterpolator miter;
            get_node_interpolator(ccode, miter);


            for(auto iter = cpoints.begin(); iter != cpoints.end(); iter++) {

                const TypeIndex &sp = *iter;
                const TypeValue sval = miter.compute(idx2p(sp), true);

                std::cout << "\t" << idx2p(sp) << " : ";
                auto siter = spoints_with_values.find(sp);
                if (siter == spoints_with_values.end()) {
                    printf(LOG::DEBUG, " appending value %lf\n", sval);
                    spoints_with_values[sp] = sval;
                    continue;
                }

                //! if the point already existed, it should have the same value!
                const TypeValue &sval2 = siter->second;
                if (!utils::is_zero(sval-sval2)) {
                    printf(LOG::DEBUG, " value mismatch %lf != %lf\n", sval, sval2);

                    //! need to split this node!
                    split_node_along_axes(_, Axes_y);
                    make_continuous(_);
                }
                else {
                    printf(LOG::DEBUG, " matching value %lf\n", sval);
                }
            }
        }*/
    }

    void make_continuous() {

        return;
        /*
        vmanager.mVertices[p2idx(TypeVertex(8,8))] = 1.0;
        print(vertices(), "vertices");

        NodeSet penultimateNodes;
        for(auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); iter++) {
            penultimateNodes.insert(LocCode::parent(*iter));
        }

        std::cout << "  I have " << penultimateNodes.size() << " penultimate nodes for " << mLeafNodes.size() << " leaves!\n";
        for(auto iter = penultimateNodes.begin(); iter != penultimateNodes.end(); iter++) {
            make_continuous(*iter);
        }

        vmanager.mVertices[p2idx(TypeVertex(8,8))] = 0.0;
        print(vertices(), "vertices");*/
    }




    //! -------------------------------------------------------------------------
        //! creation of a cell
        //!
        //! @param a:      TODO: description
        //! @param b:      TODO: description
        //! @param max_l:  TODO: description
        //! @param _:      TODO: description
        //!
        //! -------------------------------------------------------------------------
    public:
        void split_range_along_tree(const TypeCoord &a, const TypeCoord &b, const TypeScale &max_l, std::set<TypeCoord> &_) {

            //! for each dimension
            //! partition the range along powers of 2 (as large as possible)
            bool debug = 0;

            const TypeCoord max_sz = AMM_pow2(max_l)+1;
            const TypeCoord s = b-a+1;

            if (debug)
                std::cout << "\npartition(" << int(a) << ", " << int(b) << ", " << int(s) << "). max = [" << int(max_l) << ", " << int(max_sz) << "]" << std::endl;

            if (s > 2*max_sz) {
                if (debug)
                    std::cout << " larger than the max size\n";
            }

            //! find the largest size that fits within the given range
            TypeScale sl = 0;
            for(sl = max_l; sl > 0; sl--) {
                if (AMM_pow2(sl) < s) {
                    break;
                }
            }
            if (debug)
                std::cout << "\t" << int(sl) << " -- " << AMM_pow2(sl) << std::endl;

            //! now, we find the partition that splits the range into two pieces
            TypeCoord l2 = AMM_pow2(sl);
            TypeCoord m = 0;
            while(m <= a) {
                m += l2;
            }

            if (debug)
                std::cout << "\t-- [" << int(a) << " : " << int(m) << " : "<< int(b) << "]" << std::endl;

            if (m == a || m == b) {
                if (debug)
                    std::cout << "  no split needed!\n";
                return;
            }

            if (debug)
                std::cout << " inserting [2] " << int(m) << std::endl;

            _.insert(m);

            if (sl == 1) {
                if (debug)
                    std::cout <<" return l\n";
                return;
            }

            if ((!AMM_is_pow2(a) && (m > a)) || (m-a) > max_sz) {   split_range_along_tree(a, m, sl, _);    }
            if ((!AMM_is_pow2(b) && (b > m)) || (m-b) > max_sz) {   split_range_along_tree(m, b, sl, _);    }
        }

        //! ------------------------------------------------------------------
        //! TODO: description
        //!
        //! @param bb0:        //TODO: description
        //! @param sz:         //TODO: description
        //! @param components: //TODO: description
        //!
        //! ------------------------------------------------------------------
        void split_cell_along_tree(TypeVertex bb0, TypeVertex sz, std::vector<std::pair<TypeIndex,TypeIndex> > &components) {

            const TypeVertex &dbounds = dbox1();
            static TypeVertex bb1;

            //! the requested cell max
            OctUtils::os2bb(bb0, sz, bb1);

            //! the minimum size in the requested cell and the corresponding scale
            const TypeCoord msize = sz.min();
            const TypeScale mscale = std::log2(msize-1) + 1;

            AMM_log_debug << "\nsplitting_cell_along_tree( ("<<bb0<< ":"<<bb1<<" -- "<<sz<<"): " << int(msize) << " at " << int(mscale) <<"\n";

            //! Step 1: handle any potential cell expansions
            //!  if a cell touches the max edge of the mesh, but is not a valid node
            //!  we should expand it
            components.clear();

            bool expanded = false;
            for (TypeDim d = 0; d < Dim; d++) {

                if (bb1[d] == dbounds[d] && (!AMM_is_pow2_plus1(sz[d]) || sz[d] == 2)) {
                    sz[d] = AMM_next_pow2_plus1(sz[d]);
                    expanded = true;
                }
            }

            //! the expanded cell max
            OctUtils::os2bb(bb0, sz, bb1);

            if (expanded){
                //AMM_log_debug << " expanded to ("<<bb0<< ":"<<bb1<<" -- "<<sz<<")\n";
            }

            //! Step 2: split each dimension indepdenently along the tree's hierachy
            std::set<TypeCoord> partitioned_dims[Dim];
            for (TypeDim d = 0; d < Dim; d++) {
                partitioned_dims[d].insert(bb0[d]);
                partitioned_dims[d].insert(bb1[d]);
                split_range_along_tree(bb0[d], bb1[d], mscale, partitioned_dims[d]);

                AMM_log_debug << "p["<<int(d)<<"] = ((";
                for(auto piter = partitioned_dims[d].begin(); piter != partitioned_dims[d].end(); piter++)
                    AMM_log_debug << *piter;
                AMM_log_debug << "))\n";
            }

            AMM_log_info << "("<<bb0 << "," <<bb1<<") has been split into ";
            for(TypeDim d = 0; d < Dim-1; d++)
                AMM_log_info << partitioned_dims[d].size() << " x ";
            AMM_log_info << partitioned_dims[Dim-1].size() << "\n";

            //! if there wasnt any split, simply create the cell and return
            if (partitioned_dims[0].size() == 2 && partitioned_dims[1].size() == 2) {
                components.push_back(std::make_pair(p2idx(bb0), p2idx(bb1)));
                return;
            }

            //! Step 3: compute the tensor product of these partitions
            std::vector<std::pair<TypeIndex, TypeIndex>> partitioned_cells;

            std::vector<TypeCoord> Px (partitioned_dims[0].begin(), partitioned_dims[0].end());
            std::vector<TypeCoord> Py (partitioned_dims[1].begin(), partitioned_dims[1].end());

            if (2 == Dim) {

                for (size_t iy = 0; iy < Py.size()-1; iy++) {
                for (size_t ix = 0; ix < Px.size()-1; ix++) {

                    const TypeVertex b0 (Px[ix],   Py[iy]);
                    const TypeVertex b1 (Px[ix+1], Py[iy+1]);
                    const TypeVertex bs = OctUtils::bb2s(b0, b1);

                    const TypeCoord ar = OctUtils::aspect_ratio(bs);

                    //! valid aspect ratio
                    if (ar == 1 || ar == 2) {
                        components.push_back(std::make_pair(p2idx(b0), p2idx(b1)));
                        continue;
                    }

                    //! otherwise, we will be splitting the long axes to make sure
                    //! that the cells have aspect ratio 1 or 2

                    //! figure out the size of these smaller cells
                    const TypeDim mindim = (TypeDim)bs.argmin();
                    const TypeCoord minsize = bs[mindim];

                    TypeVertex offsz (minsize);
                    for(TypeDim d = 0; d < Dim; d++) {
                        offsz[d] = std::min(TypeCoord(2*minsize-1), bs[d]);
                    }

                    for(TypeCoord y = b0[1]; y < b1[1]; y+=(offsz[1]-1)) {
                    for(TypeCoord x = b0[0]; x < b1[0]; x+=(offsz[0]-1)) {

                        TypeVertex sb0 (x,y);
                        TypeVertex sb1 = OctUtils::os2bb(sb0, offsz);
                        components.push_back(std::make_pair(p2idx(sb0), p2idx(sb1)));
                    }}
                }}
            }

            else {

                std::vector<TypeCoord> Pz (partitioned_dims[2].begin(), partitioned_dims[2].end());

                for (size_t iz = 0; iz < Pz.size()-1; iz++) {
                for (size_t iy = 0; iy < Py.size()-1; iy++) {
                for (size_t ix = 0; ix < Px.size()-1; ix++) {

                    const TypeVertex b0 (Px[ix],   Py[iy],   Pz[iz]);
                    const TypeVertex b1 (Px[ix+1], Py[iy+1], Pz[iz+1]);
                    const TypeVertex bs = OctUtils::bb2s(b0, b1);

                    const TypeCoord ar = OctUtils::aspect_ratio(bs);

                    //! valid aspect ratio
                    if (ar == 1 || ar == 2) {
                        components.push_back(std::make_pair(p2idx(b0), p2idx(b1)));
                        continue;
                    }

                    //! otherwise, we will be splitting the long axes to make sure
                    //! that the cells have aspect ratio 1 or 2

                    //! figure out the size of these smaller cells
                    const TypeDim mindim = (TypeDim)bs.argmin();
                    const TypeCoord minsize = bs[mindim];

                    TypeVertex offsz (minsize);
                    for(TypeDim d = 0; d < Dim; d++) {
                        offsz[d] = std::min(TypeCoord(2*minsize-1), bs[d]);
                    }

                    for(TypeCoord z = b0[2]; z < b1[2]; z+=(offsz[2]-1)) {
                    for(TypeCoord y = b0[1]; y < b1[1]; y+=(offsz[1]-1)) {
                    for(TypeCoord x = b0[0]; x < b1[0]; x+=(offsz[0]-1)) {

                        TypeVertex sb0 (x,y,z);
                        TypeVertex sb1 = OctUtils::os2bb(sb0, offsz);
                        components.push_back(std::make_pair(p2idx(sb0), p2idx(sb1)));
                    }}}
                }}}
            }
        }


        //! --------------------------------------------------------------------------------
        //! TODO: description
        //!
        //! @param bb0:         TODO: description
        //! @param sz:          TODO: description
        //! @param components:  TODO: description
        //! @param do_expand:   TODO: description
        //!
        //! --------------------------------------------------------------------------------
        void split_cell_along_tree_v1(TypeVertex bb0, TypeVertex sz,
                                   std::vector<std::pair<TypeVertex,TypeVertex> > &components,
                                   const bool do_expand = true) {

            bool debug = true;

            const TypeVertex &dbounds = dbox1();
            static TypeVertex bb1;

            //! the requested cell max
            OctUtils::os2bb(bb0, sz, bb1);

            //! the minimum size in the requested cell and the corresponding scale
            const TypeCoord msize = sz.min();
            const TypeScale mscale = std::log2(msize-1) + 1;

            if (debug)
                std::cout << "\n >> splitting_cell_along_tree( "<<bb0<<" : "<<bb1<<" -- "<<sz<<" ) : " << msize << " at " << mscale << " : " << do_expand << "\n";

            //! Step 1: handle any potential cell expansions
            //!  if a cell touches the max edge of the mesh, but is not a valid node
            //!  we should expand it
            if (do_expand) {

                components.clear();

                bool expanded = false;
                for (TypeDim d = 0; d < Dim; d++) {

                    if (bb1[d] == dbounds[d] && (!AMM_is_pow2_plus1(sz[d]) || sz[d] == 2)) {
                        sz[d] = AMM_next_pow2_plus1(sz[d]);
                        expanded = true;
                    }
                }

                //! the expanded cell max
                OctUtils::os2bb(bb0, sz, bb1);

                if (debug) {
                    if (expanded) {
                        std::cout << " expanded to ( "<<bb0<<" : "<<bb1<<" -- "<<sz<<" )\n";
                    }
                }
            }

    #ifndef ARBIT_SPLIT
            components.push_back(std::make_pair(bb0, sz));
    #else

            //! Step 2: split each dimension indepdenently along the tree's hierachy
            std::set<TypeCoord> p[Dim];
            for (TypeDim d = 0; d < Dim; d++) {
                p[d].insert(bb0[d]);
                p[d].insert(bb1[d]);
                partition_a_dimension(bb0[d], bb1[d], mscale, p[d]);

                if (debug) {
                    std::cout << "p["<<int(d)<<"] = (( ";
                    for(auto piter = p[d].begin(); piter != p[d].end(); piter++)
                        std::cout << *piter << " ";
                    std::cout << "))\n";
                }
            }

            if (debug) {
                std::cout << bb0 << ", " << bb1 << " has been split into ";
                for(TypeDim d = 0; d < Dim-1; d++) {
                    std::cout << p[d].size() << " x ";
                }
                std::cout << p[Dim-1].size() << std::endl;
            }

            //! if there wasnt any split, simply create the cell and return
            if (p[0].size() == 2 && p[1].size() == 2) {
                components.push_back(std::make_pair(bb0, sz));
                return;
            }

            if (0) {
                for (TypeDim d = 0; d < Dim; d++) {
                std::cout << " when reading the input data, should not find anything more!\n";
                std::cout << bb0 << ", " << bb1 << " : " << sz << std::endl;
                for(auto piter = p[d].begin(); piter != p[d].end(); piter++)
                    std::cout << *piter << std::endl;
                exit(1);
                }
            }

            //! Step 3: compute the tensor product of these partitions
            //!   along the way, capture any nodes that need further split
            std::vector<std::pair<TypeIndex, TypeIndex>> further_splits;

            if (2 == Dim) {

                std::vector<TypeCoord> xp (p[0].begin(), p[0].end());
                std::vector<TypeCoord> yp (p[1].begin(), p[1].end());

                for (size_t y = 0; y < yp.size()-1; y++) {
                for (size_t x = 0; x < xp.size()-1; x++) {

                    TypeVertex b0 (xp[x],   yp[y]);
                    TypeVertex b1 (xp[x+1], yp[y+1]);

                    further_splits.push_back(std::make_pair(p2idx(b0), p2idx(b1)));
                }
                }
            }
            else {
                //! TODO
            }

            //! Step 4: split large aspect ratio nodes into tree nodes
            for(auto iter = further_splits.begin(); iter != further_splits.end(); iter++) {

                TypeVertex b0 = idx2p(iter->first);
                TypeVertex b1 = idx2p(iter->second);
                TypeVertex bs = TypeNode::bb2s(b0, b1);
                std::cout << " --> node " << components.size() << " ::  " << b0 << ", " << b1 << " : " << bs << " :: " << TypeNode::os2c(b0,bs) << std::endl;

                const TypeCoord ar = BasicOctree::aspect_ratio(bs);

                //! valid aspect ratio
                if (ar == 1 || ar == 2) {
                    components.push_back(std::make_pair(b0, bs));
                }

                else {

                    TypeVertex b12 = b1;

                    //! the smallest dimension in the size
                    const TypeCoord smallsize = bs.min();

                    //! let offsz be a square node of small size
                    TypeVertex offsz (smallsize);

                    for(TypeDim d = 0; d < Dim; d++) {
                        offsz[d] = std::min(TypeCoord(2*smallsize-1), bs[d]);
                    }

                    TypeVertex offst;
                    std::cout << " offset = " << offst << " :" << offsz << std::endl;

                    //! if x is small
                    if (bs[1] > bs[0]) {
                        offst[1] = offsz[1]-1;
                    }
                    else {
                        offst[0] = offsz[0]-1;
                    }

                    if (debug)
                        std::cout << " offset = " << offst << " :" << offsz << std::endl;

                    while (true) {

                        components.push_back(std::make_pair(b0, offsz));
                        TypeNode::os2bb(b0, offsz, b1);

                        if (debug)
                            std::cout << " node " << components.size() << " ::  " << b0 << ", " << b1 << " : " << offsz << " :: " << TypeNode::os2c(b0,bs) << std::endl;

                        b0 += offst;

                        if (b0[0] >= b12[0] || b0[1] >= b12[1])
                            break;
                    }
                }
            }
    #endif
        }



#if 0
    //! ---------------------------------------------------------------------------------
    //! TODO: description
    //!
    //! @param ncode:                   TODO: description
    //! @param consider_vacuum:         TODO: description
    //! @param dbox0:                   TODO: description
    //! @param dsize:                   TODO: description
    //! @param func:                    TODO: description
    //! @param nmismatches:             TODO: description
    //! @param nmismatches_on_boundary: TODO: description
    //! @param erroneous_nodes:         TODO: description
    //! @param erroneous_verts:         TODO: description
    //!
    //! ---------------------------------------------------------------------------------
    void reconstruct_leaf_depth(const TypeLocCode &ncode, const bool &consider_vacuum,
                          const TypeVertex &dbox0, const TypeVertex &dsize,
                          uint8_t *func,
                          size_t &nmismatches, size_t &nmismatches_on_boundary,
                          std::set<TypeLocCode> &errorneous_nodes, std::set<TypeIndex> &errorneous_verts) const {

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);

        //! static variables for node bounds and corners
        static TypeVertex nbox0, nbox1, nsize;
        static TypeVertex tbox0, tbox1;

        static VecIndex ncorners(snCorners);
        static VecValue ncorner_values(snCorners);

        static TypeVertex p;
        static TypeIndex pidx = 0;
        static TypeCoord z0 = 0, z1 = 1;


        //! get the bounds of this node!
        foo(ncode, nbox0, nsize);
        OctUtils::os2bb(nbox0, nsize, nbox1);

        //! check if the node is outside the requested domain
        tbox0 = nbox0;
        tbox1 = nbox1;
        uint8_t ntrims = OctUtils::trim_bounds(tbox0, tbox1, dbox0, dbox1);

        //! node totally out of domain
        if (ntrims > Dim)   return;


        //! get the corners and values
        get_node_corners_o_osi(nbox0, nsize, ncorners);
        for(TypeCornerId i = 0; i < snCorners; i++) {
            ncorner_values[i] = get_vertex_value(ncorners[i], consider_vacuum);
        }

        TypeInterpolator minterp = TypeInterpolator (nbox0, nbox1, ncorner_values);

        //! now, the actual stuff
        if (Dim == 3) { z0 = tbox0[2];  z1 = tbox1[2]; }

        for(TypeCoord z = z0;       z <= z1;       z++) {
        for(TypeCoord y = tbox0[1]; y <= tbox1[1]; y++) {
        for(TypeCoord x = tbox0[0]; x <= tbox1[0]; x++) {

            p[0] = x;       p[1] = y;
            if (Dim == 3){  p[2] = z;   }

            pidx = BasicOctree::p2idx(p-dbox0, dsize);
            func[pidx] = LocCode::level(ncode);
        }
        }
        }
    }
#endif


#if 0
    //! ---------------------------------------------------------------------------------
    //! TODO: description
    //!
    //! @param dbox0:  origin of domain to reconstruct over
    //! @param dsize:  size of domain to reconstruct over
    //! @param func:   function to perform reconstruction with
    //! @returns TODO: always returns true, do we need a bool return here?
    //!
    //! ---------------------------------------------------------------------------------
    bool reconstruct_depth(const TypeVertex &dbox0, const TypeVertex &dsize, std::vector<uint8_t> &func) const {

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);
        const size_t dnvalues = OctUtils::product(dsize);

        if(2 == Dim)    printf(LOG::INFO, "\n --> Constructing depth of structured grid ([%d,%d], [%d,%d])... ", dbox0[0],dbox0[1], dbox1[0],dbox1[1]);
        else            printf(LOG::INFO, "\n --> Constructing depth of structured grid ([%d,%d,%d], [%d,%d,%d])... ", dbox0[0],dbox0[1],dbox0[2],dbox1[0],dbox1[1],dbox1[2]);
        fflush(stdout);

        static constexpr TypeValue vinvalid = AMM_AMTREE_INVALID_VAL(TypeValue);

        //! create an empty function (initialize with invalid values)
        if (func.size() != dnvalues) {
            func.resize(dnvalues, vinvalid);
        }
        else {
            std::fill(func.begin(), func.end(), vinvalid);
        }

        //! we will also check for c0 continuous function!
        size_t nmismatches_on_boundary = 0;
        size_t nmismatches = 0;

        std::set<TypeLocCode> errorneous_nodes;
        std::set<TypeIndex> errorneous_verts;

        //! go over all leaf nodes and populate the function (no need to sort)
        for(auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); ++iter) {
            reconstruct_leaf_depth(*iter, false, dbox0, dsize, func.data(),
                                   nmismatches, nmismatches_on_boundary, errorneous_nodes, errorneous_verts);
        }

        //! check mismatches
        AMM_error_logic(nmismatches_on_boundary > 0, "Octree.reconstruct() failed continuity test for %d vertices!\n", nmismatches_on_boundary);
        AMM_error_logic(nmismatches > 0, "Octree.reconstruct() failed matching value test for %d vertices!\n", nmismatches_on_boundary);

        for(auto iter = mLeaves_Improper.begin(); iter != mLeaves_Improper.end(); ++iter) {
            reconstruct_leaf_depth(*iter, true, dbox0, dsize, func.data(),
                                   nmismatches, nmismatches_on_boundary, errorneous_nodes, errorneous_verts);
        }

        if (errorneous_verts.size() > 0) {
            /*size_t ic = 0;
            for(auto iter = errorneous_verts.begin(); iter != errorneous_verts.end(); iter++) {
                auto v1 = vmanager.mVertices.find(*iter);
            }*/
        }

        //! check coverage of space
        size_t ninvalids = std::count_if(func.begin(), func.end(),
                                         [](const TypeValue v) { return AMM_AMTREE_IS_INVALID(v); });
        AMM_error_logic(ninvalids > 0, "Octree.reconstruct() failed coverage test for %d vertices!\n", ninvalids);

        printf(LOG::INFO, "Done!\n");
        return true;
    }
#endif
#if 0
    //! ---------------------------------------------------------------------------------
    //! TODO: description
    //!
    //! @param dbox0:  origin of domain to reconstruct over
    //! @param dsize:  size of domain to reconstruct over
    //! @param func:   function to perform reconstruction with
    //! @returns TODO: always returns true, do we need a bool return here?
    //!
    //! ---------------------------------------------------------------------------------
    bool reconstruct_depth(const TypeVertex &dbox0, const TypeVertex &dsize, std::vector<uint8_t> &func) const {

        const TypeVertex dbox1 = OctUtils::os2bb(dbox0, dsize);
        const size_t dnvalues = dsize.product();

        if(2 == Dim)    printf(LOG::INFO, "\n --> Constructing depth of structured grid ([%d,%d], [%d,%d])... ", dbox0[0],dbox0[1], dbox1[0],dbox1[1]);
        else            printf(LOG::INFO, "\n --> Constructing depth of structured grid ([%d,%d,%d], [%d,%d,%d])... ", dbox0[0],dbox0[1],dbox0[2],dbox1[0],dbox1[1],dbox1[2]);
        fflush(stdout);

        static constexpr TypeValue vinvalid = AMM_AMTREE_INVALID_VAL(TypeValue);

        //! create an empty function (initialize with invalid values)
        if (func.size() != dnvalues) {
            func.resize(dnvalues, vinvalid);
        }
        else {
            std::fill(func.begin(), func.end(), vinvalid);
        }

        //! we will also check for c0 continuous function!
        size_t nmismatches_on_boundary = 0;
        size_t nmismatches = 0;

        std::set<TypeLocCode> errorneous_nodes;
        std::set<TypeIndex> errorneous_verts;

        //! go over all leaf nodes and populate the function (no need to sort)
        for(auto iter = mLeafNodes.begin(); iter != mLeafNodes.end(); ++iter) {
            reconstruct_leaf_depth(*iter, false, dbox0, dsize, func.data(),
                                   nmismatches, nmismatches_on_boundary, errorneous_nodes, errorneous_verts);
        }

        //! check mismatches
        AMM_error_logic(nmismatches_on_boundary > 0, "Octree.reconstruct() failed continuity test for %d vertices!\n", nmismatches_on_boundary);
        AMM_error_logic(nmismatches > 0, "Octree.reconstruct() failed matching value test for %d vertices!\n", nmismatches_on_boundary);

        for(auto iter = mLeaves_Improper.begin(); iter != mLeaves_Improper.end(); ++iter) {
            reconstruct_leaf_depth(*iter, true, dbox0, dsize, func.data(),
                                   nmismatches, nmismatches_on_boundary, errorneous_nodes, errorneous_verts);
        }

        if (errorneous_verts.size() > 0) {
            /*size_t ic = 0;
            for(auto iter = errorneous_verts.begin(); iter != errorneous_verts.end(); iter++) {
                auto v1 = vmanager.mVertices.find(*iter);
            }*/
        }

        //! check coverage of space
        size_t ninvalids = std::count_if(func.begin(), func.end(),
                                         [](const TypeValue v) { return AMM_AMTREE_IS_INVALID(v); });
        AMM_error_logic(ninvalids > 0, "Octree.reconstruct() failed coverage test for %d vertices!\n", ninvalids);

        printf(LOG::INFO, "Done!\n");
        return true;
    }
#endif


#if 0
    /*
        //! --------------------------------------------------------------------------------
        //! get value at an arbitrary point on the domain
        //! --------------------------------------------------------------------------------
        virtual inline
        TypeValue get_value(const TypeVertex &_, const TypeLocCode &l) const {

            // use a cached interpolator
            static TypeLocCode prev_l = 0;
            static TypeInterpolator* minterp = nullptr;
            if (prev_l != l) {
                delete minterp;
                minterp = get_node_interpolator(l);
                prev_l = l;
            }

            // compute the value
            return minterp->compute(_, true);
        }
        virtual inline
        TypeValue get_value(const TypeVertex &_) const {
            return get_value(_, find_leaf(_));
        }
        virtual inline
        TypeValue get_value(const TypeIndex _, const TypeLocCode l) const {
            return get_value(idx2p(_), l);
        }
        virtual inline
        TypeValue get_value(const TypeIndex _) const {
            return get_value(idx2p(_));
        }*/

        /*
        //! --------------------------------------------------------------------------------
        //! whether a node intersects with a give bounding box
        //! --------------------------------------------------------------------------------
        virtual inline
        bool is_node_intersects(const TypeLocCode &_, const TypeVertex &box0, const TypeVertex &box1) const {

            static TypeVertex nbox0, nbox1, nsize;
            get_bounds_node(_, nbox0, nsize);
            OctUtils::os2bb(nbox0, nsize, nbox1);
            return !OctUtils::is_outside(nbox0, nbox1, box0, box1);
        }

        //! --------------------------------------------------------------------------------
        //! create a multilinear interpolator for a given node
        //! --------------------------------------------------------------------------------
        virtual inline
        TypeInterpolator* get_node_interpolator(const TypeLocCode &_) const {

            static TypeVertex nbox0, nbox1, nsize;
            static std::vector<TypeIndex> ncorners(snCorners);
            static std::vector<TypeValue> ncorner_values(snCorners);

            get_bounds_node(_, nbox0, nsize);
            get_node_corners_o_osi(nbox0, nsize, ncorners);
            for(TypeCornerId i = 0; i < snCorners; i++) {
                ncorner_values[i] = get_vertex_value(ncorners[i]);
            }

            OctUtils::os2bb(nbox0, nsize, nbox1);
            return new TypeInterpolator (nbox0, nbox1, ncorner_values);
        }
    */
        /*
        //! --------------------------------------------------------------------------------
        //! traverse the tree to find a leaf that contains a given vertex
        //! --------------------------------------------------------------------------------
        virtual inline
        TypeLocCode find_leaf(const TypeVertex &_) const {

    #ifdef AMM_DEBUG_LOGIC
            AMM_error_invalid_arg(!is_valid_vertex(_), "AdaptiveTree.find_leaf(%s) got invalid vertex!\n", _.c_str());
    #endif

            // starting from root, figure out how to traverse to this location
            TypeLocCode ncode = 1;
            TypeVertex nbox0 = mOrigin;
            TypeVertex nsize = mSize;
            TypeVertex ncenter = mCenter;

            for(TypeScale l = 0; l < mDataDepth; l++) {

    #ifdef AMM_DEBUG_LOGIC
                //! make sure this vertex lies in this node
                AMM_error_logic(!BasicOctree::contains_os(_, nbox0, nsize), "AdaptiveTree.find_leaf() going in wrong direction!\n");
    #endif
                // if i have found the leaf!
                if (is_leaf(ncode)) {
                    break;
                }

                // which child next?
                TypeChildId child_id = AMTree::get_child_containing(_, ncenter, get_node_cflags(ncode).value());

                AMTree::get_bounds_child(child_id, nbox0, nsize);
                OctUtils::os2c(nbox0, nsize, ncenter);
                ncode = LocCode::child(ncode, child_id);
            }

    #ifdef AMM_DEBUG_LOGIC
            AMM_error_logic(!is_leaf(ncode), "AdaptiveTree.find_leaf(%s) failed!\n", _.c_str());
    #endif
            return ncode;
        }*/


    inline TypeLocCode
    create_node_at(const TypeVertex &_) {

        //std::cout << " amtree::create_node_at("<<_<<")\n";
#if 1
        const TypeLocCode ncode = ncenter_to_lcode(_);
        create_node(ncode);

#else
        //std::cout << "\n creating node at " << _ << " :: " <<  BasicOctree::ncenter_to_lcode(_) << "\n";
#ifdef AMM_STAGE_NODES
        TypeLocCode ncode0 = BasicOctree::ncenter_to_lcode(_);

        if (staging_phase) {
            stage_node(ncode0);
            //std::cout << " skip creaing during staging phase!\n";
            return ncode0;
        }
#endif

        if (!m_enableRectangularNodes) {
            TypeLocCode ncode = BasicOctree::create_node_at(_);
#ifdef AMM_STAGE_NODES
            if (ncode != ncode0) {
                std::cout << " mismatch 1: " << ncode <<", "<< ncode0 << "\n";
                exit(1);
            }
#endif
            //std::cout << " created node " << ncode << " (no vacuum)\n";
            //print_nodes(mNodes);
            return ncode;
        }

        // starting from root, figure out how to traverse to this location
        TypeLocCode ncode = 1;
        TypeVertex ncenter = mCenter;
        TypeVertex nbox0 = mOrigin;
        TypeVertex nsize = mSize;
        TypeChildId nxt_child = 0;

        // figure out the expected level of this vertex
        const TypeScale nlvl = OctUtils::find_lvl_of_nodeCenter(_, mDataDepth);
        for(TypeScale l = 0; l < nlvl; l++) {

#ifdef AMM_DEBUG_LOGIC
            // make sure this vertex lies in this node
            AMM_error_logic(!Octree::contains_os(_, nbox0, nsize), "AMTree.create_node_at() going in wrong direction!\n");
#endif

            // the id of the next child
            nxt_child = OctUtils::get_child_containing(_, ncenter);

            // figure out the updated configuration
            const TypeChildFlag cflags_cur = get_node_cflags(ncode).value();
            TypeChildFlag cflags_new = NodeConfig::create_standard_child(cflags_cur, nxt_child).value();

            if (!m_enableImproperNodes) {
                cflags_new = NodeConfig::fill_vacuum_complete(cflags_new).value();
            }

            // now actually update the node (and split the node correspondingly)
            update_node(ncode, cflags_new);

            // the id and bounds of the next child
            OctUtils::get_bounds_child(nxt_child, nbox0, nsize);
            OctUtils::os2c(nbox0, nsize, ncenter);
            ncode = LocCode::child(ncode, nxt_child);

#ifdef AMM_DEBUG_TRACK_CONFIGS
            NodeConfig::insert_config(mconfigs_create, nxt_child, cflags_cur, cflags_new);
#endif
        }

#ifdef AMM_DEBUG_LOGIC
        AMM_error_logic(!(_ == ncenter), "AMTree.create_node_at(%s) did not reach the requested vertex!\n", _.c_str());
#endif
#ifdef AMM_STAGE_NODES
        if (ncode != ncode0) {
            std::cout << " mismatch 2: " << ncode <<", "<< ncode0 << "\n";
            exit(1);
        }
#endif
#endif
        //std::cout << " created node " << ncode << " (yes vacuum)\n";
        return ncode;
    }
#endif
#endif
    //! ----------------------------------------------------------------------------
    //! ----------------------------------------------------------------------------
};

}}   // end of namespace
#endif
