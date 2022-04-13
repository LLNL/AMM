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
#ifdef AMM_USE_VTK
#ifndef AMM_VTK_AMM_H
#define AMM_VTK_AMM_H

//! ----------------------------------------------------------------------------
#include <cstdio>

#include <vtkType.h>
#include <vtkCell.h>
#include <vtkCellType.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
//#include <vtkUnstructuredGridWriter.h>
//#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "types/dtypes.hpp"
#include "utils/exceptions.hpp"
#include "tree/octree_utils.hpp"
#include "tree/amtree.hpp"
#include "amm.hpp"

//! ----------------------------------------------------------------------------
//! an extension of vtkUnstructured grid to store amm
//! ----------------------------------------------------------------------------
class vtkAMM : public vtkUnstructuredGrid {

    //! ------------------------------------------------------------------------
    //! convert a c++ data type to vtk data type
    template <typename T>
    static inline int vtkDataType(const T v=0) {
        static std::unordered_map<std::size_t, long> _;
        if(_.empty()) {
            _[typeid(void).hash_code()]               = VTK_VOID;
            _[typeid(char).hash_code()]               = VTK_CHAR;
            _[typeid(signed char).hash_code()]        = VTK_SIGNED_CHAR;
            _[typeid(unsigned char).hash_code()]      = VTK_UNSIGNED_CHAR;
            _[typeid(short).hash_code()]              = VTK_SHORT;
            _[typeid(unsigned short).hash_code()]     = VTK_UNSIGNED_SHORT;
            _[typeid(int).hash_code()]                = VTK_INT;
            _[typeid(unsigned int).hash_code()]       = VTK_UNSIGNED_INT;
            _[typeid(long).hash_code()]               = VTK_LONG;
            _[typeid(unsigned long).hash_code()]      = VTK_UNSIGNED_LONG;
            _[typeid(float).hash_code()]              = VTK_FLOAT;
            _[typeid(double).hash_code()]             = VTK_DOUBLE;
            _[typeid(std::string).hash_code()]        = VTK_STRING;
            _[typeid(long long).hash_code()]          = VTK_LONG_LONG;
            _[typeid(unsigned long long).hash_code()] = VTK_UNSIGNED_LONG_LONG;
            _[typeid(int64_t).hash_code()]            = VTK_LONG_LONG;
            _[typeid(uint64_t).hash_code()]           = VTK_UNSIGNED_LONG_LONG;
        }
        return _[typeid(T).hash_code()];
    }

    //! create a new vtk data array
    template<typename T>
    static inline vtkDataArray* data_array(const std::string &name, const size_t sz, const T v=0) {

        vtkDataArray* _ = vtkDataArray::CreateDataArray(vtkDataType<T>());
        _->SetNumberOfComponents(1);
        _->SetNumberOfTuples(static_cast<vtkIdType>(sz));
        _->SetName(name.c_str());
        return _;
    }

    //! ------------------------------------------------------------------------
    //! convert the vertex ordering in octree cells to vtk cells
    static inline uint8_t vtkCellOrdering(const uint8_t i) {

        // create ordering of points with respect to vtk cell
        switch(i) {
            case 2:     return 3;
            case 3:     return 2;

            // will be valid only for 3D
            case 6:     return 7;
            case 7:     return 6;
        }
        return i;
    }

    //! ------------------------------------------------------------------------
    //! convert x,y to a vertex
    template <typename T>
    static inline auto vtk2vertex(const T x, const T y) {
        return Vec<2, TypeCoord> (static_cast<TypeCoord>(x),
                                  static_cast<TypeCoord>(y));
    }

    //! convert x,y,z to a vertex
    template <typename T>
    static inline auto vtk2vertex(const T x, const T y, const T z) {
        return Vec<3, TypeCoord> (static_cast<TypeCoord>(x),
                                  static_cast<TypeCoord>(y),
                                  static_cast<TypeCoord>(z));
    }

    //! convert a point to a vertex
    template <TypeDim Dim, typename T>
    static inline auto vtk2vertex(const T pnt[]) {
        Vec<Dim, TypeCoord> _;
        for(TypeDim d = 0; d < Dim; d++)
            _[d] = static_cast<TypeCoord>(pnt[d]);
        return _;
    }

    //! convert bounds to a vertex
    static inline void vtk2bounds(const double bounds[6], Vertex2 &bb0, Vertex2 &bb1) {
        bb0 = vtk2vertex(bounds[0], bounds[2]);
        bb1 = vtk2vertex(bounds[1], bounds[3]);
    }
    static inline void vtk2bounds(const double bounds[6], Vertex3 &bb0, Vertex3 &bb1) {
        bb0 = vtk2vertex(bounds[0], bounds[2], bounds[4]);
        bb1 = vtk2vertex(bounds[1], bounds[3], bounds[5]);
    }

    //! ------------------------------------------------------------------------
    inline static auto reads(const std::string &filename, vtkUnstructuredGrid* data) {

        amm::timer t;
        AMM_log_info << "Loading vtkUnstructuredGrid from [" << filename << "]...";
        fflush(stdout);

        vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
        reader->SetFileName(filename.c_str());
        reader->Update();
        data->DeepCopy(reader->GetOutput());

        t.stop();
        AMM_logc_info << t << std::endl;
        return data;
    }

    inline static void writes(const std::string &filename, const vtkUnstructuredGrid* data) {

        amm::timer t;
        AMM_log_info << "Writing vtkUnstructuredGrid to [" << filename << "]...";
        fflush(stdout);

        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkXMLUnstructuredGridWriter::New();
        writer->SetInputData(const_cast<vtkUnstructuredGrid*>(data));
        writer->SetFileName(filename.c_str());
        writer->Update();

        t.stop();
        AMM_logc_info << " done!" << t << std::endl;
    }

public:
    //! ------------------------------------------------------------------------
    void PrintHeader (ostream &os, vtkIndent indent) {
        //vtkUnstructuredGrid::PrintHeader(os, indent);
    }
    void PrintTrailer (ostream &os, vtkIndent indent) {
        //vtkUnstructuredGrid::PrintTrailer(os, indent);
    }
    void PrintSelf (ostream &os, vtkIndent indent) {
        //this->Superclass::PrintSelf(os, indent);
    }

    //! ------------------------------------------------------------------------
    inline auto read(const std::string &filename) {
        return vtkAMM::reads(filename, this);
    }
    inline void write(const std::string &filename) const {
        return vtkAMM::writes(filename, this);
    }

    //! ------------------------------------------------------------------------
    inline TypeDim dim(size_t dsize[3]) const {

        double dbounds[6];
        const_cast<vtkAMM*>(this)->GetBounds(dbounds);

        bool is_valid = true;
        for (TypeDim d = 0; d < 3; d++) {
            is_valid &= AMM_is_zero(dbounds[2*d]);
            dsize[d] = dbounds[2*d+1] - dbounds[2*d] + 1;
        }

        if (!is_valid) {
            throw std::invalid_argument("vtkAMM::validate_dims(): Invalid mesh origin!\n");
        }

        return (dsize[2] == 1) ? 2 : 3;
    }

    //! ------------------------------------------------------------------------
    //! convert a vtk mesh to amm
    //! ------------------------------------------------------------------------
    template <typename TypeValue, TypeDim Dim, TypeScale Depth>
    auto vtk2amm(bool allow_rectangular, bool allow_vacuum) {

        static_assert((Dim == 2 || Dim == 3), "vtk2amm works for 2D and 3D only!");

        using TypeAMM          = amm::AMM<TypeValue, Dim, Depth>;
        using TypeATree        = typename TypeAMM::AMTree;
        using TypeVertex     = typename TypeAMM::TypeVertex;

        //! --------------------------------------------------------------------
        // vtk-related data types
        static constexpr int vCellType = (2 == Dim) ? VTK_QUAD : VTK_HEXAHEDRON;

        //! --------------------------------------------------------------------
        // get the bounds and test the dimensionality
        size_t dsize[3] = {0, 0, 0};
        const TypeDim dim = this->dim(dsize);

        //! --------------------------------------------------------------------
        const vtkIdType ncells = vtkUnstructuredGrid::GetNumberOfCells();
        const vtkIdType nvertices = vtkUnstructuredGrid::GetNumberOfPoints();

        // vtk is not const-safe
        vtkDataArray *values = vtkUnstructuredGrid::GetPointData()->GetArray("value");
        vtkDataArray *clevels = vtkUnstructuredGrid::GetCellData()->GetArray("cell_level");
        vtkDataArray *ctypes = vtkUnstructuredGrid::GetCellData()->GetArray("cell_type");

        AMM_log_info << "Creating AMM of size (" << dsize[0] << ", " << dsize[1] << ", " << dsize[2] << ")"
                     << " with " << ncells << " cells and " << nvertices << " vertices\n";

        AMM_error_invalid_arg(dim != Dim,
                              "Incorrect template instance for given data of dim %d", dim);
        AMM_error_invalid_arg(nvertices != values->GetNumberOfValues(),
                              "Incorrect data. nvertices = %d; nvalues %d",
                              nvertices != values->GetNumberOfValues());
        AMM_error_invalid_arg(ncells != clevels->GetNumberOfValues() || ncells != ctypes->GetNumberOfValues(),
                              "Incorrect data. ncells = %d; nlevels = %d, ntypes = %d",
                              ncells, clevels->GetNumberOfValues(), ctypes->GetNumberOfValues());

        //! --------------------------------------------------------------------
        TypeAMM *mesh_amm = new TypeAMM(dsize, allow_rectangular, allow_vacuum);

        // Declare the variables we will use for each cell in the loop
        static double cell_bounds_vtk[6];
        static TypeVertex cell_bounds_amm[2];
        static std::vector<TypeValue> corner_vals(TypeATree::snCorners, 0.0);

        // Convert vtk mesh cells to adaptive mesh nodes
        for (vtkIdType cell_idx = 0; cell_idx < ncells; cell_idx++) {

            // All cells should be the same type
            AMM_error_invalid_arg(vtkUnstructuredGrid::GetCellType(cell_idx) != vCellType,
                                  "Incorrect cell encountered");

            vtkCell* cell = vtkUnstructuredGrid::GetCell(cell_idx);
            AMM_error_invalid_arg(cell->GetNumberOfPoints() != TypeATree::snCorners,
                                  "Incorrect cell (%d) encountered! expected %d corners but found %d",
                                  cell_idx, TypeATree::snCorners, cell->GetNumberOfPoints());

            // Get cell bounds
            cell->GetBounds(cell_bounds_vtk);
            vtk2bounds(cell_bounds_vtk, cell_bounds_amm[0], cell_bounds_amm[1]);

            // don't create vacuum nodes (type>=Dim)!
            if (allow_vacuum && ctypes->GetTuple1(cell_idx) >= Dim) {
                continue;
            }

            // Get the values at cells's corners
            for (TypeCornerId corner_idx = 0; corner_idx < TypeATree::snCorners; corner_idx++) {
                const TypeIndex pidx = static_cast<TypeIndex>(cell->GetPointId(corner_idx));
                corner_vals[corner_idx] = static_cast<TypeValue>(values->GetTuple1(pidx));
            }

            // convert from ccw order (vtk) to row-major (needed for mlinear interpolator)
            std::swap(corner_vals[2], corner_vals[3]);
            if (Dim == 3) std::swap(corner_vals[6], corner_vals[7]);

            // TODO: once this functionality is enabled
            // we also need to add the additional vertices
            mesh_amm->create_cell(cell_bounds_amm[0], cell_bounds_amm[1], corner_vals);
        }

        mesh_amm->finalize();
        //mesh_amm->finalize_improper_nodes();
        //mesh_amm->print_summary();
        return mesh_amm;
    }

    //! ------------------------------------------------------------------------
    //! convert amm to a vtk mesh
    //! ------------------------------------------------------------------------
    template<typename TypeValue, TypeDim Dim, TypeScale Depth>
    auto amm2vtk(const amm::AMM<TypeValue, Dim, Depth> &mesh,
                 const amm::AMM_field_data &field_data) {

        static_assert((Dim == 2 || Dim == 3), "amm2vtk works for 2D and 3D only!");

        // AMM object
        using TypeAMM = amm::AMM<TypeValue, Dim, Depth>;
        using TypeVertList = typename TypeAMM::ListFinalVerts;
        using TypeCellList = typename TypeAMM::ListFinalCells;

        // vtk-related data types
        static constexpr int vCellType = (2 == Dim) ? VTK_QUAD : VTK_HEXAHEDRON;
        static constexpr TypeCornerId  ncorners = (2 == Dim) ? 4 : 8;

        // mesh spacings
        static constexpr TypeCoord dx = 1;
        static constexpr TypeCoord dy = 1;
        static constexpr TypeCoord dz = (2 == Dim) ? 0 : 1;

        //! --------------------------------------------------------------------
        // reconstructed information
        TypeVertList vertices;
        TypeCellList cells;
        mesh.reconstruct(vertices, cells);

        const size_t nverts = vertices.size();
        const size_t ncells = cells.size();

        //! --------------------------------------------------------------------
        amm::timer t;

        // now, create the vtk representation
        AMM_log_info << "Constructing vtk representation...";
        fflush(stdout);

        // ---------------------------------------------------------------------
        // add vertices and vertex data
        vtkPoints* vpoints = vtkPoints::New();
        vpoints->SetDataTypeToUnsignedInt();
        vpoints->SetNumberOfPoints (nverts);

        vtkDataArray* vd0 = data_array<TypeValue>(amm::AMM_vertex_data_names[0].c_str(), nverts);
        vtkDataArray* vd1 = data_array<TypePrecision>(amm::AMM_vertex_data_names[1], nverts);

        // convert to vtkpoints
        std::unordered_map<size_t, size_t> point2index;
        size_t pcnt = 0;
        for (auto idx = vertices.begin(); idx != vertices.end(); idx++) {

            auto pidx = idx->first;
            auto pt = mesh.idx2p(idx->first);
            const float z = (2 == Dim) ? 0 : float(pt[2])*dz;

            // capture the vertex
            vpoints->SetPoint(pcnt, float(pt[0])*dx, float(pt[1])*dy, z);

            // capture vertex data
            vd0->InsertComponent(pcnt, 0, std::get<0>(idx->second));
            vd1->InsertComponent(pcnt, 0, std::get<1>(idx->second));

            point2index[pidx] = pcnt++;
        }
        vtkUnstructuredGrid::SetPoints(vpoints);

        // ---------------------------------------------------------------------
        // add cells and cell data
        vtkDataArray* cd1 = data_array<TypeScale>(amm::AMM_cell_data_names[1], ncells);
        vtkDataArray* cd2 = data_array<TypeDim>(amm::AMM_cell_data_names[2], ncells);
        vtkDataArray* cd3 = data_array<TypeChildId>(amm::AMM_cell_data_names[3], ncells);

        vtkIdType cpidxs[ncorners];
        size_t ccnt = 0;
        for(auto iter = cells.begin(); iter != cells.end(); iter++) {

            auto cell = *iter;
            const auto &cpoints = std::get<0>(cell);

            for(size_t j = 0; j < cpoints.size(); j++) {
                uint8_t vj = vtkCellOrdering(uint8_t(j));
                cpidxs[vj] = point2index[cpoints[j]];
            }

            // capture the cell
            vtkUnstructuredGrid::InsertNextCell(vCellType, ncorners, cpidxs);

            // capture cell data
            cd1->InsertComponent(ccnt, 0, std::get<1>(cell));
            cd2->InsertComponent(ccnt, 0, std::get<2>(cell));
            cd3->InsertComponent(ccnt, 0, std::get<3>(cell));
            ccnt++;
        }

        // ---------------------------------------------------------------------
        // add field data
        vtkDataArray* fd0 = data_array<double>(amm::AMM_field_data_names[0], 1);
        vtkDataArray* fd1 = data_array<size_t>(amm::AMM_field_data_names[1], 1);

        fd0->InsertComponent(0, 0, std::get<0>(field_data));
        fd1->InsertComponent(0, 0, std::get<1>(field_data));

        // ---------------------------------------------------------------------
        vtkUnstructuredGrid::GetPointData()->AddArray(vd0);
        vtkUnstructuredGrid::GetPointData()->AddArray(vd1);
        vtkUnstructuredGrid::GetCellData()->AddArray(cd1);
        vtkUnstructuredGrid::GetCellData()->AddArray(cd2);
        vtkUnstructuredGrid::GetCellData()->AddArray(cd3);
        vtkUnstructuredGrid::GetFieldData()->AddArray(fd0);
        vtkUnstructuredGrid::GetFieldData()->AddArray(fd1);

        vtkUnstructuredGrid::GetPointData()->SetActiveScalars(amm::AMM_vertex_data_names[0].c_str());

        // ---------------------------------------------------------------------
        vpoints->Delete();
        vd0->Delete();
        vd1->Delete();
        cd1->Delete();
        cd2->Delete();
        cd3->Delete();
        fd0->Delete();
        fd1->Delete();

        // ---------------------------------------------------------------------
        t.stop();
        AMM_logc_info << " done!" << t << " Created "
                      << vtkUnstructuredGrid::GetNumberOfPoints() << " vertices and "
                      << vtkUnstructuredGrid::GetNumberOfCells() << " cells!\n";
        return this;
    }

    //! ------------------------------------------------------------------------
    //! ------------------------------------------------------------------------
};
#endif  // AMM_VTK_AMM_H
#endif  // USE_VTK
