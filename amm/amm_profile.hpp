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

#ifndef AMM_PROFILE_H
#define AMM_PROFILE_H
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>

#include "macros.hpp"
#include "tree/block_abstract.hpp"

namespace amm {

struct amm_profile {

    size_t num_coefficients;
    size_t num_bytes;
    double psnr;

    // mesh components
    size_t mesh_vert;
    size_t mesh_cell;
    std::array<size_t, 3> mesh_cell_by_type;

    // tree components
    size_t tree_vert, tree_vert_bdry, tree_vert_vac, tree_vert_bylvl, tree_vertcache, tree_vertcache_bylvl;
    size_t tree_nodes, tree_node_bdry, tree_node_leaf, tree_node_int, tree_node_vac, tree_node_vacleaf, tree_nodecache;

    // vertices by block
    size_t tree_blocks;
    std::array<size_t, AMM_MAX_PRECISION> tree_blocks_at_prec;
    std::array<size_t, AMM_MAX_PRECISION> tree_verts_at_prec;
    std::array<size_t, AMM_BLOCK_NVERTS> tree_blocks_with_verts;

    inline void write_csv_header(std::ostream &out) const {

        out << "num_coefficients,num_bytes,psnr,";
        out << "mesh_vert,mesh_cell,mesh_cell_by_type[0],mesh_cell_by_type[1],mesh_cell_by_type[2],";
        out << "tree_vert,tree_vert_bdry,tree_vert_vac,tree_vert_bylvl,tree_vertcache,tree_vertcache_bylvl,";
        out << "tree_nodes,tree_node_bdry,tree_node_leaf,tree_node_int,tree_node_vac,tree_node_vacleaf,tree_nodecache,";
        out << "tree_blocks,";
        for(size_t i = 0; i < tree_blocks_at_prec.size(); i++)          out << "tree_blocks_at_prec["<<i<<"],";
        for(size_t i = 0; i < tree_verts_at_prec.size(); i++)           out << "tree_verts_at_prec["<<i<<"],";
        for(size_t i = 0; i < tree_blocks_with_verts.size(); i++)       out << "tree_blocks_with_verts["<<i<<"],";
        out << std::endl;
    }

    inline void write_csv(std::ostream &out) const {

        out << num_coefficients <<"," << num_bytes << "," << psnr << ",";
        out << mesh_vert << "," << mesh_cell << "," << mesh_cell_by_type[0] << "," << mesh_cell_by_type[1] << "," << mesh_cell_by_type[2] << ",";
        out << tree_vert << "," << tree_vert_bdry << "," << tree_vert_vac << "," << tree_vert_bylvl << "," << tree_vertcache << "," << tree_vertcache_bylvl << ",";
        out << tree_nodes << "," << tree_node_bdry << "," << tree_node_leaf << "," << tree_node_int << "," << tree_node_vac << "," << tree_node_vacleaf << "," << tree_nodecache << ",";
        out << tree_blocks << ",";
        for(auto i = tree_blocks_at_prec.begin(); i != tree_blocks_at_prec.end(); ++i)       out << *i << ",";
        for(auto i = tree_verts_at_prec.begin(); i != tree_verts_at_prec.end(); ++i)         out << *i << ",";
        for(auto i = tree_blocks_with_verts.begin(); i != tree_blocks_with_verts.end(); ++i) out << *i << ",";
        out << std::endl;
    }

    inline void write_csv(const std::string &filename, bool include_header=false) const {

        std::ofstream out;
        out.open(filename);

        if (include_header)
            this->write_csv_header(out);

        this->write_csv(out);
        out.close();
    }
};
}   // end of namespace
#endif
