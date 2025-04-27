cmake_minimum_required(VERSION 3.31)

project(FFourier)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
add_library(FFourier STATIC)
target_sources(FFourier PUBLIC ${PROJECT_SOURCE_DIR}/src/CNumber.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer2D.cpp)
target_include_directories(FFourier PUBLIC ${PROJECT_SOURCE_DIR}/include)

project(test_FTransformer1D)
if(OPENMP_FOUND)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable(test_FTransformer1D ${PROJECT_SOURCE_DIR}/src/test_FTransformer1D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/CNumber.cpp
                                   ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp)
set_target_properties(test_FTransformer1D PROPERTIES OUTPUT_NAME "test_FTransformer1D.exec")
target_include_directories(test_FTransformer1D PUBLIC "${PROJECT_SOURCE_DIR}/include")
