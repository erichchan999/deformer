#include "DisplaceByNormalEigen.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>
#include <Eigen/Dense>
#include <tracy/Tracy.hpp>

PXR_NAMESPACE_USING_DIRECTIVE

void DisplaceByNormalEigen::displaceByNormal(VtArray<GfVec3f>& usdVertices, VtArray<int>& usdFaceVertexIndices, VtArray<int>& usdFaceVertexCounts)
{
    ZoneScopedNC("displaceByNormalEigen", 0x00FF00);
    // Convert USD to Eigen
    Eigen::Matrix3Xf vertices{3, usdVertices.size()};
    for (size_t i = 0; i < vertices.cols(); i++) {
        vertices.col(i) << usdVertices[i][0], usdVertices[i][1], usdVertices[i][2];
    }

    // Face loop
    Eigen::Matrix3Xf verticesNormal = Eigen::Matrix3Xf::Zero(3, vertices.cols());
    size_t idxOffset = 0;
    for (size_t faceIdx = 0; faceIdx < usdFaceVertexCounts.size(); faceIdx++) {
        int vertexCountCurrFace = usdFaceVertexCounts[faceIdx];

        const int i0 = usdFaceVertexIndices[idxOffset + 0];
        const int i1 = usdFaceVertexIndices[idxOffset + 1];
        const int i2 = usdFaceVertexIndices[idxOffset + 2];

        auto v0 = vertices.col(i0);
        auto v1 = vertices.col(i1);
        auto v2 = vertices.col(i2);

        Eigen::Vector3f faceNormal = (v2-v0).cross(v1-v0);

        for (size_t i = 0; i < vertexCountCurrFace; i++) {
            verticesNormal.col(usdFaceVertexIndices[idxOffset + i]).noalias() += faceNormal;
        }

        idxOffset += vertexCountCurrFace;
    }

    // Normalize normals
    verticesNormal.colwise().normalize();

    // Displace vertices by normalized normals
    vertices += verticesNormal;

    // Write displaced vertices back into usd
    for (int i = 0; i < usdVertices.size(); i++) {
        usdVertices[i][0] = vertices(0, i);
        usdVertices[i][1] = vertices(1, i);
        usdVertices[i][2] = vertices(2, i);
    }

    return;
}