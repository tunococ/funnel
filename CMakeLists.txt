cmake_minimum_required(VERSION 3.10)

project(funnel)

add_subdirectory(src)
add_subdirectory(tests)

enable_testing()
add_test(NAME test-all COMMAND funnel-test)

