include_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(post_function post_function.cc)
target_link_libraries(post_function mvn)

add_executable(three_classes three_classes.cc)
target_link_libraries(three_classes mvn)