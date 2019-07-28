# toplevel CMakeLists.txt for a catkin workspace
# catkin/cmake/toplevel.cmake

cmake_minimum_required(VERSION 2.8.3)

Project(kitcar-gazebo-simulation)

set(CATKIN_TOPLEVEL TRUE)

include($ENV{KITCAR_REPO_PATH}/kitcar-ros/KITcar_brain/cmake/format.cmake)
include($ENV{KITCAR_REPO_PATH}/kitcar-ros/KITcar_brain/cmake/common_compiler_options.cmake)

if(NOT DEFINED ENV{CAR_NAME})
  message(FATAL_ERROR "CAR_NAME is not set!!!")
endif()

# setting up BULID_TYPEs

# if nothing else is defines, Debug is default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

message("BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# dont generate messages for lisp
set(ENV{ROS_LANG_DISABLE} "genlisp:geneus:gennodejs")

# search for catkin within the workspace
set(_cmd "catkin_find_pkg" "catkin" "${CMAKE_SOURCE_DIR}")
execute_process(COMMAND ${_cmd}
  RESULT_VARIABLE _res
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _err
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT _res EQUAL 0 AND NOT _res EQUAL 2)
  # searching fot catkin resulted in an error
  string(REPLACE ";" " " _cmd_str "${_cmd}")
  message(FATAL_ERROR "Search for 'catkin' in workspace failed (${_cmd_str}): ${_err}")
endif()

# include catkin from workspace or via find_package()
if(_res EQUAL 0)
  set(catkin_EXTRAS_DIR "${CMAKE_SOURCE_DIR}/${_out}/cmake")
  # include all.cmake without add_subdirectory to let it operate in same scope
  include(${catkin_EXTRAS_DIR}/all.cmake NO_POLICY_SCOPE)
  add_subdirectory("${_out}")

else()
  # use either CMAKE_PREFIX_PATH explicitly passed to CMake as a command line argument
  # or CMAKE_PREFIX_PATH from the environment
  if(NOT DEFINED CMAKE_PREFIX_PATH)
    if(NOT "$ENV{CMAKE_PREFIX_PATH}" STREQUAL "")
      string(REPLACE ":" ";" CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
    endif()
  endif()

  # list of catkin workspaces
  set(catkin_search_path "")
  foreach(path ${CMAKE_PREFIX_PATH})
    if(EXISTS "${path}/.catkin")
      list(FIND catkin_search_path ${path} _index)
      if(_index EQUAL -1)
        list(APPEND catkin_search_path ${path})
      endif()
    endif()
  endforeach()

  # search for catkin in all workspaces
  set(CATKIN_TOPLEVEL_FIND_PACKAGE TRUE)
  find_package(catkin QUIET
    NO_POLICY_SCOPE
    PATHS ${catkin_search_path}
  #  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
  )
  unset(CATKIN_TOPLEVEL_FIND_PACKAGE)

  if(NOT catkin_FOUND)
    message(FATAL_ERROR "find_package(catkin) failed. catkin was neither found in the workspace nor in the CMAKE_PREFIX_PATH. One reason may be that no ROS setup.sh was sourced before.")
  endif()
endif()

catkin_workspace()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-everything")
  set_target_properties(gtest_main PROPERTIES COMPILE_FLAGS "-Wno-everything")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-suggest-override -Wno-error -Wno-zero-as-null-pointer-constant -Wno-useless-cast")
  set_target_properties(gtest_main PROPERTIES COMPILE_FLAGS "-Wno-suggest-override -Wno-error -Wno-zero-as-null-pointer-constant -Wno-useless-cast")
endif()

