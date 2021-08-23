add_executable(funnel-test test_main.cpp)

set_property(TARGET funnel-test PROPERTY CXX_STANDARD_REQUIRED OFF)
set_property(TARGET funnel-test PROPERTY CXX_STANDARD 20)

target_link_libraries(funnel-test PUBLIC funnel)

