#include "DisplaceByNormalCudaKernels.h"
#include <cuda_runtime.h>
#include <cstdio>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper functions to make float3 act like math vectors
__device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline void operator+=(float3& a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__device__ inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
__device__ inline float3 normalize(const float3& v)
{
    float magnitude = norm3df(v.x, v.y, v.z);
    if (magnitude < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / magnitude, v.y / magnitude, v.z / magnitude);
}


__global__ void computeFaceNormalsKernel(
    const float3* vertices,
    float3* vertexNormals,
    const int* faceVertexIndices,
    const int* faceVertexCounts,
    const int* faceVertexOffsets,
    int numFaces
)
{
    int faceIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Need to exit for extra threads that occur when faces dont divide evenly into threads per block
    if (faceIdx >= numFaces) {
        return;
    }

    int vertexOffset = faceVertexOffsets[faceIdx];
    int vertexCount = faceVertexCounts[faceIdx];

    int i0 = faceVertexIndices[vertexOffset + 0];
    int i1 = faceVertexIndices[vertexOffset + 1];
    int i2 = faceVertexIndices[vertexOffset + 2];

    float3 v0 = vertices[i0];
    float3 v1 = vertices[i1];
    float3 v2 = vertices[i2];

    float3 e1 = v2 - v0;
    float3 e2 = v1 - v0;
    float3 faceNormal = cross(e1, e2);

    for (int j = 0; j < vertexCount; j++) {
        int vertexIdx = faceVertexIndices[vertexOffset + j];
        atomicAdd(&vertexNormals[vertexIdx].x, faceNormal.x);
        atomicAdd(&vertexNormals[vertexIdx].y, faceNormal.y);
        atomicAdd(&vertexNormals[vertexIdx].z, faceNormal.z);
    }
}

__global__ void normalizeAndDisplaceKernel(
    float3* vertices,
    float3* vertexNormals,
    int numVertices
)
{
    int vertexIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Need to exit for extra threads that occur when faces dont divide evenly into threads per block
    if (vertexIdx >= numVertices) {
        return;
    }

    float3 normal = normalize(vertexNormals[vertexIdx]);

    vertices[vertexIdx] += normal;
}

void launchDisplaceByNormalKernels(
    float3* d_vertices,
    float3* d_vertexNormals,
    const int* d_faceVertexIndices,
    const int* d_faceVertexCounts,
    const int* d_faceVertexOffsets,
    int numVertices,
    int numFaces
)
{
    // Set each thread to handle 1 face
    const int threadsPerBlock = 256;

    // Calculate normals
    int blocksForFaces = (numFaces + threadsPerBlock - 1) / threadsPerBlock;
    computeFaceNormalsKernel<<<blocksForFaces, threadsPerBlock>>>(
        d_vertices,
        d_vertexNormals,
        d_faceVertexIndices,
        d_faceVertexCounts,
        d_faceVertexOffsets,
        numFaces
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Normalize normal and displace
    int blocksForVertices = (numVertices + threadsPerBlock - 1) / threadsPerBlock;
    normalizeAndDisplaceKernel<<<blocksForVertices, threadsPerBlock>>>(
        d_vertices,
        d_vertexNormals,
        numVertices
    );
    CUDA_CHECK(cudaGetLastError());
}
