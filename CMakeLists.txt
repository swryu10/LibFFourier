cmake_minimum_required(VERSION 3.31)

project(FFourier)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
add_library(FFourier STATIC)
target_sources(FFourier PUBLIC ${PROJECT_SOURCE_DIR}/src/CNumber.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp)
target_include_directories(FFourier PUBLIC ${PROJECT_SOURCE_DIR}/include)
