cmake_minimum_required(VERSION 3.18...4.1)
project(deformer LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(pxr REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(submodules/tracy)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(USD_ROOT "" CACHE PATH "Path to USD installation")
if(USD_ROOT)
    list(APPEND CMAKE_PREFIX_PATH "${USD_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Normal calculation strategies
set(NORMAL_STRATEGIES
    src/DisplaceByNormalEigen.cpp
    src/DisplaceByNormalThreads.cpp
    src/DisplaceByNormalTBB.cpp
    src/DisplaceByNormalCuda.cpp src/DisplaceByNormalCudaKernels.cu
)

# Create executable target
add_executable(deformer
    src/main.cpp
    ${NORMAL_STRATEGIES}
)
set_target_properties(deformer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(deformer PRIVATE
    OUTPUT_USD_PATH="${OUTPUT_USD_PATH}"
    INPUT_USD_PATH="${INPUT_USD_PATH}"
)
target_include_directories(deformer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link 3rd party libraries
target_link_libraries(deformer PRIVATE
    Eigen3::Eigen
    Tracy::TracyClient
    TBB::tbb
    ${PXR_LIBRARIES}
)