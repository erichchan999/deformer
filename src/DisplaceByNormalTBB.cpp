#include "DisplaceByNormalTBB.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>
#include <Eigen/Dense>
#include <tracy/Tracy.hpp>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/spin_mutex.h>

PXR_NAMESPACE_USING_DIRECTIVE

void DisplaceByNormalTBB::displaceByNormal(VtArray<GfVec3f>& usdVertices, VtArray<int>& usdFaceVertexIndices, VtArray<int>& usdFaceVertexCounts)
{
    ZoneScopedNC("displaceByNormalTBB", 0x0000FF);

    // Convert USD to Eigen
    Eigen::Matrix3Xf vertices{3, usdVertices.size()};
    for (size_t i = 0; i < vertices.cols(); i++) {
        vertices.col(i) << usdVertices[i][0], usdVertices[i][1], usdVertices[i][2];
    }

    // Normals
    Eigen::Matrix3Xf verticesNormal = Eigen::Matrix3Xf::Zero(3, vertices.cols());
    std::vector<tbb::spin_mutex> mutexes(verticesNormal.cols());

    // Calculate normals
    std::vector<size_t> faceVertexOffsets(usdFaceVertexCounts.size() + 1);
    faceVertexOffsets[0] = 0;
    for (size_t i = 0; i < usdFaceVertexCounts.size(); i++) {
        faceVertexOffsets[i + 1] = faceVertexOffsets[i] + usdFaceVertexCounts[i];
    }
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, usdFaceVertexCounts.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t faceIdx = range.begin(); faceIdx < range.end(); faceIdx++) {
                size_t vertexIndexCursor = faceVertexOffsets[faceIdx];
                size_t vertexCountCurrFace = usdFaceVertexCounts[faceIdx];

                const int i0 = usdFaceVertexIndices[vertexIndexCursor + 0];
                const int i1 = usdFaceVertexIndices[vertexIndexCursor + 1];
                const int i2 = usdFaceVertexIndices[vertexIndexCursor + 2];

                auto v0 = vertices.col(i0);
                auto v1 = vertices.col(i1);
                auto v2 = vertices.col(i2);

                Eigen::Vector3f faceNormal = (v2-v0).cross(v1-v0);
                for (size_t j = 0; j < vertexCountCurrFace; j++) {
                    size_t vertexCol = usdFaceVertexIndices[vertexIndexCursor + j];
                    tbb::spin_mutex::scoped_lock lock(mutexes[vertexCol]);
                    verticesNormal.col(vertexCol).noalias() += faceNormal;
                }
            }
        }
    );

    // Normalize normals and displace
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, verticesNormal.cols()),
        [&](const tbb::blocked_range<size_t>& range) {
            verticesNormal.middleCols(range.begin(), range.size()).colwise().normalize();
            vertices.middleCols(range.begin(), range.size()) += verticesNormal.middleCols(range.begin(), range.size());
        }
    );

    // Write displaced vertices back into USD
    for (int i = 0; i < usdVertices.size(); i++) {
        usdVertices[i][0] = vertices(0, i);
        usdVertices[i][1] = vertices(1, i);
        usdVertices[i][2] = vertices(2, i);
    }

    return;
}