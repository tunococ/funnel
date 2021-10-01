add_library(funnel OBJECT)

target_include_directories(funnel PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_sources(funnel PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include/funnel/funnel_base.hpp"
)

