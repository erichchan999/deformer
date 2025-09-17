#include "DisplaceByNormalCuda.h"
#include "DisplaceByNormalCudaKernels.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>
#include <tracy/Tracy.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void DisplaceByNormalCuda::displaceByNormal(VtArray<GfVec3f>& usdVertices, VtArray<int>& usdFaceVertexIndices, VtArray<int>& usdFaceVertexCounts)
{
    ZoneScopedNC("displaceByNormalCuda", 0xFF00FF);

    int numVertices = usdVertices.size();
    int numFaces = usdFaceVertexCounts.size();
    int totalFaceVertexIndices = usdFaceVertexIndices.size();

    // Calculate offsets into vertex array for each face
    std::vector<int> faceVertexOffsets(numFaces);
    faceVertexOffsets[0] = 0;
    for (int i = 1; i < numFaces; i++) {
        faceVertexOffsets[i] = faceVertexOffsets[i-1] + usdFaceVertexCounts[i-1];
    }

    // Initialise device memory
    float3* d_vertices;
    float3* d_vertexNormals;
    int* d_faceVertexIndices;
    int* d_faceVertexCounts;
    int* d_faceVertexOffsets;

    size_t verticesSize = numVertices * sizeof(float3);
    size_t normalsSize = numVertices * sizeof(float3);
    size_t faceIndicesSize = totalFaceVertexIndices * sizeof(int);
    size_t faceCountsSize = numFaces * sizeof(int);
    size_t faceVertexOffsetsSize = numFaces * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_vertexNormals, normalsSize));
    CUDA_CHECK(cudaMalloc(&d_vertices, verticesSize));
    CUDA_CHECK(cudaMalloc(&d_faceVertexIndices, faceIndicesSize));
    CUDA_CHECK(cudaMalloc(&d_faceVertexCounts, faceCountsSize));
    CUDA_CHECK(cudaMalloc(&d_faceVertexOffsets, faceVertexOffsetsSize));

    CUDA_CHECK(cudaMemset(d_vertexNormals, 0, normalsSize));
    // VtArray<GfVec3f> is contiguous so can cast directly to float3[]
    CUDA_CHECK(cudaMemcpy(d_vertices, usdVertices.data(), verticesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faceVertexIndices, usdFaceVertexIndices.data(), faceIndicesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faceVertexCounts, usdFaceVertexCounts.data(), faceCountsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faceVertexOffsets, faceVertexOffsets.data(), faceVertexOffsetsSize, cudaMemcpyHostToDevice));

    // Execute cuda kernels
    launchDisplaceByNormalKernels(
        d_vertices,
        d_vertexNormals,
        d_faceVertexIndices,
        d_faceVertexCounts,
        d_faceVertexOffsets,
        numVertices,
        numFaces
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy vertices from device back to USD
    CUDA_CHECK(cudaMemcpy(usdVertices.data(), d_vertices, verticesSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_vertices));
    CUDA_CHECK(cudaFree(d_vertexNormals));
    CUDA_CHECK(cudaFree(d_faceVertexIndices));
    CUDA_CHECK(cudaFree(d_faceVertexCounts));
    CUDA_CHECK(cudaFree(d_faceVertexOffsets));
}
