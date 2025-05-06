cmake_minimum_required(VERSION 3.31)

project(FFourier)
add_library(FFourier STATIC)
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(MPI)
if(MPI_FOUND)
    add_compile_definitions(_MPI)
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
    set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
endif()
target_sources(FFourier PUBLIC ${PROJECT_SOURCE_DIR}/src/Parallel.cpp
                               ${PROJECT_SOURCE_DIR}/src/CNumber.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer2D.cpp
                               ${PROJECT_SOURCE_DIR}/src/FTransformer3D.cpp)
target_include_directories(FFourier PUBLIC "${PROJECT_SOURCE_DIR}/include")
if(MPI_FOUND)
    target_include_directories(FFourier PUBLIC "${MPI_INCLUDE_PATH}")
    target_link_libraries(FFourier PUBLIC ${MPI_LIBRARIES})
endif()

project(test_FTransformer1D)
if(MPI_FOUND)
    add_compile_definitions(_MPI)
endif()
add_executable(test_FTransformer1D ${PROJECT_SOURCE_DIR}/src/test_FTransformer1D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/Parallel.cpp
                                   ${PROJECT_SOURCE_DIR}/src/CNumber.cpp)
set_target_properties(test_FTransformer1D PROPERTIES OUTPUT_NAME "test_FTransformer1D.exec")
target_include_directories(test_FTransformer1D PUBLIC "${PROJECT_SOURCE_DIR}/include")
if(MPI_FOUND)
    target_include_directories(test_FTransformer1D PUBLIC "${MPI_INCLUDE_PATH}")
    target_link_libraries(test_FTransformer1D PUBLIC ${MPI_LIBRARIES})
endif()

project(test_FTransformer2D)
if(MPI_FOUND)
    add_compile_definitions(_MPI)
endif()
add_executable(test_FTransformer2D ${PROJECT_SOURCE_DIR}/src/test_FTransformer2D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/FTransformer2D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/FTransformer1D.cpp
                                   ${PROJECT_SOURCE_DIR}/src/Parallel.cpp
                                   ${PROJECT_SOURCE_DIR}/src/CNumber.cpp)
set_target_properties(test_FTransformer2D PROPERTIES OUTPUT_NAME "test_FTransformer2D.exec")
target_include_directories(test_FTransformer2D PUBLIC "${PROJECT_SOURCE_DIR}/include")
if(MPI_FOUND)
    target_include_directories(test_FTransformer2D PUBLIC "${MPI_INCLUDE_PATH}")
    target_link_libraries(test_FTransformer2D PUBLIC ${MPI_LIBRARIES})
endif()

project(test_FTransformer3D)
if(MPI_FOUND)
    add_compile_definitions(_MPI)
endif()
add_executable(test_FTransformer3D ${PROJECT_SOURCE_DIR}/src/test_FTransformer3D.cpp)
set_target_properties(test_FTransformer3D PROPERTIES OUTPUT_NAME "test_FTransformer3D.exec")
target_include_directories(test_FTransformer3D PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(test_FTransformer3D PUBLIC FFourier)
if(MPI_FOUND)
    target_include_directories(test_FTransformer3D PUBLIC "${MPI_INCLUDE_PATH}")
    target_link_libraries(test_FTransformer3D PUBLIC ${MPI_LIBRARIES})
endif()
