#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MPPI::cartpole_mppi" for configuration "Release"
set_property(TARGET MPPI::cartpole_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPPI::cartpole_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libcartpole_mppi.so"
  IMPORTED_SONAME_RELEASE "libcartpole_mppi.so"
  )

list(APPEND _cmake_import_check_targets MPPI::cartpole_mppi )
list(APPEND _cmake_import_check_files_for_MPPI::cartpole_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libcartpole_mppi.so" )

# Import target "MPPI::autorally_mppi" for configuration "Release"
set_property(TARGET MPPI::autorally_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPPI::autorally_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libautorally_mppi.so"
  IMPORTED_SONAME_RELEASE "libautorally_mppi.so"
  )

list(APPEND _cmake_import_check_targets MPPI::autorally_mppi )
list(APPEND _cmake_import_check_files_for_MPPI::autorally_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libautorally_mppi.so" )

# Import target "MPPI::double_integrator_mppi" for configuration "Release"
set_property(TARGET MPPI::double_integrator_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPPI::double_integrator_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libdouble_integrator_mppi.so"
  IMPORTED_SONAME_RELEASE "libdouble_integrator_mppi.so"
  )

list(APPEND _cmake_import_check_targets MPPI::double_integrator_mppi )
list(APPEND _cmake_import_check_files_for_MPPI::double_integrator_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libdouble_integrator_mppi.so" )

# Import target "MPPI::quadrotor_mppi" for configuration "Release"
set_property(TARGET MPPI::quadrotor_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MPPI::quadrotor_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libquadrotor_mppi.so"
  IMPORTED_SONAME_RELEASE "libquadrotor_mppi.so"
  )

list(APPEND _cmake_import_check_targets MPPI::quadrotor_mppi )
list(APPEND _cmake_import_check_files_for_MPPI::quadrotor_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libquadrotor_mppi.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
