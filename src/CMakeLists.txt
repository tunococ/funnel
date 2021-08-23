add_library(funnel INTERFACE)

set_property(TARGET funnel PROPERTY CXX_STANDARD_REQUIRED OFF)
set_property(TARGET funnel PROPERTY CXX_STANDARD 20)

target_include_directories(funnel INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_sources(funnel INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/funnel_base.hpp")

