#pragma once

#include <cuda_runtime.h>

void launchDisplaceByNormalKernels(
    float3* d_vertices,
    float3* d_vertexNormals,
    const int* d_faceVertexIndices,
    const int* d_faceVertexCounts,
    const int* d_faceVertexOffsets,
    int numVertices,
    int numFaces
);
