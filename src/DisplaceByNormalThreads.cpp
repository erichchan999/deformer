#include "DisplaceByNormalThreads.h"
#include <Eigen/Dense>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/vec3f.h>
#include <tracy/Tracy.hpp>
#include <thread>
#include <mutex>
#include <numeric>
#include <cassert>

PXR_NAMESPACE_USING_DIRECTIVE

void DisplaceByNormalThreads::displaceByNormal(VtArray<GfVec3f>& usdVertices, VtArray<int>& usdFaceVertexIndices, VtArray<int>& usdFaceVertexCounts)
{
    ZoneScopedNC("displaceByNormalThreads", 0xFF0000);
    size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    {
        // ZoneScopedN("Creating threads");
        for (size_t i = 0; i < numThreads; i++) {
            threads.emplace_back();
        }
    }

    // Convert USD to Eigen
    Eigen::Matrix3Xf vertices{3, usdVertices.size()};
    for (size_t i = 0; i < vertices.cols(); i++) {
        vertices.col(i) << usdVertices[i][0], usdVertices[i][1], usdVertices[i][2];
    }

    // Normals matrix
    Eigen::Matrix3Xf verticesNormal = Eigen::Matrix3Xf::Zero(3, vertices.cols());
    std::vector<std::mutex> mutexes(verticesNormal.cols());

    // Calculate normals
    {
        assert(numThreads > 0);
        size_t numFacesPerThread = usdFaceVertexCounts.size() / numThreads;
        size_t remainderNumFaces = usdFaceVertexCounts.size() % numThreads;

        size_t baseVertexIndexCursor = 0;
        size_t faceCursor = 0;
        // ZoneScopedN("Scheduling threads to run 1");
        for (size_t i = 0; i < numThreads; i++) {
            size_t numFacesCurrThread = numFacesPerThread + ((i < remainderNumFaces) ? 1 : 0);
            threads[i] = std::thread(
                [
                    &,
                    faceCursor,
                    baseVertexIndexCursor,
                    numFacesCurrThread
                ]
                () {
                    size_t currThreadVertexIndexCursor = baseVertexIndexCursor;
                    for (size_t idx = 0; idx < numFacesCurrThread; idx++) {
                        size_t vertexCountCurrFace = usdFaceVertexCounts[faceCursor + idx];
                        const int i0 = usdFaceVertexIndices[currThreadVertexIndexCursor + 0];
                        const int i1 = usdFaceVertexIndices[currThreadVertexIndexCursor + 1];
                        const int i2 = usdFaceVertexIndices[currThreadVertexIndexCursor + 2];

                        auto v0 = vertices.col(i0);
                        auto v1 = vertices.col(i1);
                        auto v2 = vertices.col(i2);

                        Eigen::Vector3f faceNormal = (v2-v0).cross(v1-v0);

                        for (size_t j = 0; j < vertexCountCurrFace; j++) {
                            size_t vertexCol = usdFaceVertexIndices[currThreadVertexIndexCursor + j];
                            std::scoped_lock<std::mutex> slock(mutexes[vertexCol]);
                            verticesNormal.col(vertexCol).noalias() += faceNormal;
                        }

                        currThreadVertexIndexCursor += usdFaceVertexCounts[faceCursor + idx];
                    }
                    return;
                }
            );

            baseVertexIndexCursor += std::accumulate(usdFaceVertexCounts.begin() + faceCursor, usdFaceVertexCounts.begin() + faceCursor + numFacesCurrThread, 0);
            faceCursor += numFacesCurrThread;
        }
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    // Normalize normals and displace mesh
    {
        // ZoneScopedN("Scheduling threads to run 2");
        size_t numCols = verticesNormal.cols();
        size_t colsPerThread = numCols / numThreads;
        size_t remainderCols = numCols % numThreads;

        size_t colsCursor = 0;
        for (size_t i = 0; i < numThreads; i++) {
            size_t numColsCurr = colsPerThread + (i < remainderCols ? 1 : 0);
            threads[i] = std::thread(
                [
                    &,
                    colsCursor,
                    numColsCurr
                ]() 
                {
                    verticesNormal.middleCols(colsCursor, numColsCurr).colwise().normalize();
                    vertices.middleCols(colsCursor, numColsCurr) += verticesNormal.middleCols(colsCursor, numColsCurr);
                }
            );
            colsCursor += colsPerThread + (i < remainderCols ? 1 : 0);
        }
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    // Write displaced vertices back into usd
    for (int i = 0; i < usdVertices.size(); i++) {
        usdVertices[i][0] = vertices(0, i);
        usdVertices[i][1] = vertices(1, i);
        usdVertices[i][2] = vertices(2, i);
    }

    return;
}