# toplevel CMakeLists.txt for a catkin workspace
# catkin/cmake/toplevel.cmake

cmake_minimum_required(VERSION 2.8.3)

Project(kitcar-gazebo-simulation)

set(CATKIN_TOPLEVEL TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BASE_LINE_FLAGS "-Wall -fstack-protector -Wextra -Wnon-virtual-dtor -Werror=old-style-cast -Werror=return-type -Werror=reorder -Wmissing-braces -Wmissing-field-initializers -fPIC")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(BASE_LINE_FLAGS "${BASE_LINE_FLAGS} -Wsuggest-override -Wuseless-cast -Wcast-qual -Wlogical-op -Wredundant-decls")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(BASE_LINE_FLAGS "${BASE_LINE_FLAGS} -Wshadow=local -Wduplicated-cond")
  endif()
  if(NOT "$ENV{HOSTNAME}" STREQUAL "kitcarnuc" AND NOT "$ENV{HOSTNAME}" STREQUAL "build_extern_for_nuc")
    # This makes all warnings into errors (with two exceptions). This is NOT done on the car, since it would be too annoying.
    # unused-variable and unused-but-set-variable are excluded because these warnings come up frequently during debugging and
    # would be distrubing.
    set(BASE_LINE_FLAGS "${BASE_LINE_FLAGS} -Werror -Wno-error=unused-variable -Wno-error=unused-but-set-variable")
  endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(BASE_LINE_FLAGS "${BASE_LINE_FLAGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c99-extensions -Wno-shadow -Wno-date-time -Wno-missing-prototypes -Wno-exit-time-destructors -Wno-global-constructors -Wno-conversion -Wno-float-conversion -Wno-double-promotion -Wno-sign-conversion -Wno-float-equal -Wno-padded -Wno-weak-vtables -Wno-abstract-vbase-init -Wno-potentially-evaluated-expression -Wno-zero-as-null-pointer-constant -Wno-undefined-func-template -Wno-covered-switch-default -Wno-return-std-move-in-c++11 -Werror=inconsistent-missing-override -Werror=inconsistent-missing-destructor-override -Werror=header-hygiene -Werror=shadow-field")

# -Wcovered-switch-default is disabled because clang-tidy can detect this too and can be configure more fine-grained.
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${BASE_LINE_FLAGS} -g -Og -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${BASE_LINE_FLAGS} -O3 -fdevirtualize-at-ltrans -fipa-pta -fuse-linker-plugin -fno-fat-lto-objects -flto -fuse-ld=gold -Wl,-flto -DNDEBUG -DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g -DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_INFO")
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} Warn Stfu Unoptimized)
set(CMAKE_CXX_FLAGS_WARN "${BASE_LINE_FLAGS}  -Wconversion")
set(CMAKE_CXX_FLAGS_STFU "${CMAKE_CXX_FLAGS_RELEASE} -DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_NONE")
set(CMAKE_CXX_FLAGS_COVERAGE "${BASE_LINE_FLAGS} -fprofile-arcs -ftest-coverage -fPIC -Og -fsanitize=address -g")
set(CMAKE_CXX_FLAGS_UNOPTIMIZED "${BASE_LINE_FLAGS}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(CMAKE_AR "gcc-ar")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH true)
endif()

include_directories(SYSTEM ${CATKIN_DEVEL_PREFIX}/include)
include_directories(SYSTEM /usr/src/googletest/googletest/include/)

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

