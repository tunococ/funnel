cmake_minimum_required(VERSION 3.12)

if(NOT DEFINED PROJECT_NAME)
  set(NOT_SUBPROJECT ON)
else()
  set(NOT_SUBPROJECT OFF)
endif()

project(funnel)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

if(NOT_SUBPROJECT)

  include(CTest)

  set(catch2_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/Catch2")
  add_subdirectory("${catch2_SOURCE_DIR}")
  include("${catch2_SOURCE_DIR}/extras/Catch.cmake")

  add_subdirectory(tests)

endif()

