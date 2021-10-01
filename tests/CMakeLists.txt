add_library(funnel_test_lib OBJECT)
target_sources(funnel_test_lib PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/funnel_test_lib/funnel_test_lib.hpp"
)
target_include_directories(funnel_test_lib PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/funnel_test_lib"
)
target_link_libraries(funnel_test_lib PUBLIC
  funnel
)

function(add_funnel_test module_name)
  add_executable("${module_name}" ${ARGN})
  set(ASAN_OPTIONS
    -fsanitize=address
    -fno-omit-frame-pointer
  )
  target_link_libraries("${module_name}" PRIVATE
    funnel_test_lib
    Catch2::Catch2WithMain
    ${ASAN_OPTIONS}
    gcov
  )
  target_compile_options("${module_name}" PRIVATE
    ${ASAN_OPTIONS}
    -g -O0 -Wall -fprofile-arcs -ftest-coverage
  )
  catch_discover_tests("${module_name}")
endfunction()

add_funnel_test(funnel_base_test
  "${CMAKE_CURRENT_SOURCE_DIR}/funnel_base_test.cpp"
)

