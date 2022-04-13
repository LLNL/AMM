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

//! -------------------------------------------------------------------------------------
//! command line tool to compute the amm representation
//! -------------------------------------------------------------------------------------

#include <cstring>

#ifdef AMM_USE_VTK
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGridReader.h>
#include "amm/vtk/vtkAMM.hpp"
#endif

#include "types/enums.hpp"
#include "streams/enums.hpp"
#include "utils/data_utils.hpp"
#include "utils/logger.hpp"
#include "amm/amm_config.hpp"
#include "amm/macros.hpp"
#include "amm_handler.hpp"


//! -------------------------------------------------------------------------------------
//! restrictions for version 0.1
//! -------------------------------------------------------------------------------------

const std::string _version = "0.1";
void check_version_restrictions(amm::amm_config &c) {

    std::cout << "\n --------- AMM v"<<_version<<"--------\n";

#ifndef AMM_STAGE_VERTS
    std::cerr << "Vertex staging is required for version " <<_version<<"!\n";
    exit(1);
#endif
#ifndef AMM_STAGE_NODES
    std::cerr << "Node staging is required for version " <<_version<<"!\n";
    exit(1);
#endif


#ifdef AWR_FILTER_EXTERNAL_COEFFICIENTS
    if (c.amm_enableImproper) {
        std::cerr << "Improper nodes do not properly work with filtering!\n";
        exit(1);
    }
#endif

#ifdef AWR_VALIDATE_PER_COEFF
    std::cerr << "Per-coefficient validation is currently not active!\n";
    exit(1);
#endif

    auto stream_type = as_enum<EnumStream>(c.amm_streamType);

#ifndef AMM_ENABLE_PRECISION
    if (streams::is_precision_stream(stream_type)) {
        std::cerr << "Warning: Processing precision stream without AMM_ENABLE_PRECISION still uses full-precision values!\n"
                  << "         Please recompile with AMM_ENABLE_PRECISION to use precision streams effectively!\n";
    }
#endif

    AMM_error_invalid_arg((streams::is_precision_stream(stream_type) &&
                           c.amm_streamEndCriterion.compare("kb") != 0),
                           "Precision streams support \"kb\" end criterion only");


    if (c.amm_enableImproper && !c.amm_enableRectangular) {
        std::cerr << "Warning: Disabling the request for Improper nodes as they can be used only with Cuboidal nodes!\n";
        c.amm_enableImproper = false;
    }
    if (c.amm_writeLowres && c.amm_outPath.empty()) {
        std::cerr << "Warning: Disabling the request to write low-res function because no output path was provided!\n";
        c.amm_writeLowres = false;
    }
}



//! -------------------------------------------------------------------------------------
//! build amm from function or wavelet coefficients
//! -------------------------------------------------------------------------------------
template <typename TypeFunc>
void build_amm(const amm::amm_config& c) {

    AMMRepresentationHandler<TypeFunc> awr (c);
    if (c.amm_inputType.compare("func") == 0) {

        // function could be a different datatype than awr!
        if (c.amm_inputPrecision.compare("u8") == 0) {
            awr.template init_with_function<uint8_t>(c.amm_dataDims[0], c.amm_dataDims[1], c.amm_dataDims[2]);
        }
        else if (c.amm_inputPrecision.compare("f32") == 0) {
            awr.template init_with_function<float>(c.amm_dataDims[0], c.amm_dataDims[1], c.amm_dataDims[2]);
        }
        else if (c.amm_inputPrecision.compare("f64") == 0) {
            awr.template init_with_function<double>(c.amm_dataDims[0], c.amm_dataDims[1], c.amm_dataDims[2]);
        }
    }
    else if (c.amm_inputType == "wcoeffs") {
        // wavelet will be a floating type that is same as awr type!
        awr.init_with_wcoeffs(c.amm_dataDims[0], c.amm_dataDims[1], c.amm_dataDims[2]);
    }

    awr.compute_amm();
    if (!c.amm_outPath.empty()) {
        awr.write_vtkamm();
        // write after validation automatically!
        /*if (c.amm_writeLowres) {
            awr.write_inverse();
        }*/
    }
}


//! -------------------------------------------------------------------------------------
//! -------------------------------------------------------------------------------------
//! entry point to the amm cli
//! -------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    //! ---------------------------------------------------------------------------------
    amm::set_loglevel(amm::LogLevel::Debug);
    if (0) {
        AMM_log_debug << "Debug logging enabled!\n";
        AMM_log_info << "Info logging enabled!\n";
        AMM_log_warn << "Warning logging enabeld!\n";
        AMM_log_error << "Error logging enabled!\n";
    }

    //! ---------------------------------------------------------------------------------
    amm::amm_config c;
    c.parse(argc, argv);
    if (argc <= 1)
        c.usage();

    check_version_restrictions(c);

    //! ---------------------------------------------------------------------------------
    //! create the awr
    //! ---------------------------------------------------------------------------------
    if (c.amm_inputType.compare("amm") == 0){

#ifndef AMM_USE_VTK
        std::cerr << "Please recompile with \"USE_VTK\" to read AMM meshes!\n";
        exit(1);
#else
        // need this here only to fetch the data type!
        vtkAMM *vtk_mesh = new vtkAMM();
        vtk_mesh->read(c.amm_inputFilename);

        const std::string dtype = vtk_mesh->GetPointData()->GetArray(0)->GetDataTypeAsString();
        if (dtype == "float") {
            AMMRepresentationHandler<float> awr (c);
            awr.init_with_amm(*vtk_mesh);
            awr.write_vtkamm("test-reproduced");
        }
        else if (dtype == "double") {
            AMMRepresentationHandler<double> awr (c);
            awr.init_with_amm(*vtk_mesh);
            awr.write_vtkamm("test-reproduced");
            //awr.test_iterator();
        }
#endif
    }

    // TODO: currently, we only do floats and doubles
    else if (c.amm_inputPrecision.compare("u8") == 0){
        build_amm<float>(c);
    }
    else if (c.amm_inputPrecision.compare("f32") == 0){
        build_amm<float>(c);
    }
    else if (c.amm_inputPrecision.compare("f64") == 0){
        build_amm<double>(c);
    }
    else {
        std::cerr << "Invalid precision! should be \"u8\", \"f32\", or \f64\"\n";
        exit(1);
    }

    //! ---------------------------------------------------------------------------------
    //! all done!
    //! ---------------------------------------------------------------------------------
    return EXIT_SUCCESS;
}

//! -------------------------------------------------------------------------------------
