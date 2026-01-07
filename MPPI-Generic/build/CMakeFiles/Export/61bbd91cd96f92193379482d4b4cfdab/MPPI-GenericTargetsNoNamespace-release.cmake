#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cartpole_mppi" for configuration "Release"
set_property(TARGET cartpole_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cartpole_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libcartpole_mppi.so"
  IMPORTED_SONAME_RELEASE "libcartpole_mppi.so"
  )

list(APPEND _cmake_import_check_targets cartpole_mppi )
list(APPEND _cmake_import_check_files_for_cartpole_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libcartpole_mppi.so" )

# Import target "autorally_mppi" for configuration "Release"
set_property(TARGET autorally_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(autorally_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libautorally_mppi.so"
  IMPORTED_SONAME_RELEASE "libautorally_mppi.so"
  )

list(APPEND _cmake_import_check_targets autorally_mppi )
list(APPEND _cmake_import_check_files_for_autorally_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libautorally_mppi.so" )

# Import target "double_integrator_mppi" for configuration "Release"
set_property(TARGET double_integrator_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(double_integrator_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libdouble_integrator_mppi.so"
  IMPORTED_SONAME_RELEASE "libdouble_integrator_mppi.so"
  )

list(APPEND _cmake_import_check_targets double_integrator_mppi )
list(APPEND _cmake_import_check_files_for_double_integrator_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libdouble_integrator_mppi.so" )

# Import target "quadrotor_mppi" for configuration "Release"
set_property(TARGET quadrotor_mppi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(quadrotor_mppi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/MPPI-Generic/libquadrotor_mppi.so"
  IMPORTED_SONAME_RELEASE "libquadrotor_mppi.so"
  )

list(APPEND _cmake_import_check_targets quadrotor_mppi )
list(APPEND _cmake_import_check_files_for_quadrotor_mppi "${_IMPORT_PREFIX}/lib/MPPI-Generic/libquadrotor_mppi.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
