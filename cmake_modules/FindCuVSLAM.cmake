# - Find cuVSLAM (https://github.com/NVIDIA-ISAAC-ROS/cuVSLAM)
#
# CUVSLAM_ROOT_DIR or one of CUVSLAM_ROOT, CUVSLAM_SRC_DIR,
# CUVSLAM_DST_DIR and CUVSLAM_BUILD_DIR may point to a release SDK,
# source tree or build tree.
#
# It sets the following variables:
#  CUVSLAM_FOUND         - Set to false, or undefined, if cuVSLAM isn't found.
#  CUVSLAM_VERSION       - The version of cuVSLAM found when a source tree
#                          containing VERSION is used (e.g., "17.0.0").
#  CUVSLAM_INCLUDE_DIRS  - The cuVSLAM include directory.
#  CUVSLAM_LIBRARIES     - The cuVSLAM library to link against.

set(_CUVSLAM_ROOT_HINTS
    ${CUVSLAM_ROOT_DIR}
    $ENV{CUVSLAM_ROOT}
    $ENV{CUVSLAM_ROOT_DIR}
    $ENV{CUVSLAM_SRC_DIR}
    $ENV{CUVSLAM_DST_DIR}
    $ENV{CUVSLAM_BUILD_DIR}
)

find_path(CUVSLAM_INCLUDE_DIR
    NAMES cuvslam/cuvslam2.h
    HINTS ${_CUVSLAM_ROOT_HINTS}
    PATHS /usr /usr/local /opt/cuvslam
    PATH_SUFFIXES include libs
)

find_library(CUVSLAM_LIBRARY
    NAMES cuvslam
    HINTS ${_CUVSLAM_ROOT_HINTS}
    PATHS /usr /usr/local /opt/cuvslam
    PATH_SUFFIXES lib lib64 bin build/bin
)

if(CUVSLAM_INCLUDE_DIR)
    get_filename_component(_CUVSLAM_ROOT_FROM_INCLUDE "${CUVSLAM_INCLUDE_DIR}/.." ABSOLUTE)
    if(EXISTS "${_CUVSLAM_ROOT_FROM_INCLUDE}/VERSION")
        file(STRINGS "${_CUVSLAM_ROOT_FROM_INCLUDE}/VERSION" CUVSLAM_VERSION LIMIT_COUNT 1)
        string(STRIP "${CUVSLAM_VERSION}" CUVSLAM_VERSION)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CuVSLAM
    FOUND_VAR CUVSLAM_FOUND
    REQUIRED_VARS CUVSLAM_LIBRARY CUVSLAM_INCLUDE_DIR
    VERSION_VAR CUVSLAM_VERSION
)

if(CUVSLAM_FOUND)
    set(CUVSLAM_INCLUDE_DIRS "${CUVSLAM_INCLUDE_DIR}")
    set(CUVSLAM_LIBRARIES "${CUVSLAM_LIBRARY}")

    if(NOT TARGET cuvslam::cuvslam)
        add_library(cuvslam::cuvslam UNKNOWN IMPORTED)
        set_target_properties(cuvslam::cuvslam PROPERTIES
            IMPORTED_LOCATION "${CUVSLAM_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUVSLAM_INCLUDE_DIR}"
            INTERFACE_COMPILE_FEATURES cxx_std_17
        )
    endif()
endif()

mark_as_advanced(CUVSLAM_INCLUDE_DIR CUVSLAM_LIBRARY)
unset(_CUVSLAM_ROOT_HINTS)
unset(_CUVSLAM_ROOT_FROM_INCLUDE)
