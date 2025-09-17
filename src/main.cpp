#include "DisplaceByNormalStrategy.h"
#include "DisplaceByNormalEigen.h"
#include "DisplaceByNormalThreads.h"
#include "DisplaceByNormalTBB.h"
#include "DisplaceByNormalCuda.h"
#include <pxr/base/vt/array.h>
#include <pxr/pxr.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <iostream>

PXR_NAMESPACE_USING_DIRECTIVE

int main()
{
    auto sphereStage = UsdStage::Open(INPUT_USD_PATH);
    UsdGeomMesh usdMesh;
    VtArray<GfVec3f> usdVertices;
    VtArray<int> usdFaceVertexIndices;
    VtArray<int> usdFaceVertexCounts;
    assert(sphereStage);
    
    std::cout << "Opened " << INPUT_USD_PATH << std::endl;

    // Find the first geometry in the file, that's our geometry to displace.
    for (const UsdPrim prim : sphereStage->Traverse()) {
        if (prim.IsA<UsdGeomMesh>()) {
            usdMesh = UsdGeomMesh(prim);
            std::cout << "Prim: " << prim.GetPath() << std::endl;
            std::cout << "Type: " << prim.GetTypeName() << std::endl;
        }
    }

    // Extract vertex positions
    const UsdAttribute pointsAttr = usdMesh.GetPointsAttr();
    if (pointsAttr.Get(&usdVertices)) {
        std::cout << "\n=== Vertex Data ===" << std::endl;
        std::cout << "Vertex count: " << usdVertices.size() << std::endl;
        std::cout << "First 5 vertices:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), usdVertices.size()); ++i) {
            const GfVec3f& p = usdVertices[i];
            std::cout << "  v[" << i << "]: (" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
        }
    }

    // Extract face vertex indices
    const UsdAttribute fviAttr = usdMesh.GetFaceVertexIndicesAttr();
    if (fviAttr.Get(&usdFaceVertexIndices)) {
        std::cout << "\n=== Polygon Data ===" << std::endl;
        std::cout << "Total face vertex indices: " << usdFaceVertexIndices.size() << std::endl;
        std::cout << "First 15 indices: ";
        for (size_t i = 0; i < std::min(size_t(15), usdFaceVertexIndices.size()); ++i) {
            std::cout << usdFaceVertexIndices[i] << " ";
        }
        std::cout << std::endl;
    }

    // Extract face vertex counts (how many vertices per face)
    const UsdAttribute fvcAttr = usdMesh.GetFaceVertexCountsAttr();
    if (fvcAttr.Get(&usdFaceVertexCounts)) {
        std::cout << "Face count: " << usdFaceVertexCounts.size() << std::endl;
        std::cout << "First 5 face vertex counts: ";
        for (size_t i = 0; i < std::min(size_t(5), usdFaceVertexCounts.size()); ++i) {
            std::cout << usdFaceVertexCounts[i] << " ";
        }
        std::cout << std::endl;
    }

    // Check whether there are ngons in geometry
    // Not necessary but interesting to know
    if (usdFaceVertexCounts.size() > 0) {
        bool hasNgons = false;
        for (size_t i = 0; i < usdFaceVertexCounts.size(); i++) {
            if (usdFaceVertexCounts[i] > 4) {
                hasNgons = true;
                break;
            }
        }
        std::cout << "\n=== Are there ngons? ===";
        if (hasNgons) {
            std::cout << "\nYes" << std::endl;
        } else {
            std::cout << "\nNo" << std::endl;
        }
    }

    // Run each displace by normal strategy
    std::cout << "\n=== Displacing By Normals ===" << std::endl;
    std::vector<std::pair<std::string, std::unique_ptr<DisplaceByNormalStrategy>>> strategies;
    strategies.push_back({"Eigen", std::make_unique<DisplaceByNormalEigen>()});
    strategies.push_back({"Threads", std::make_unique<DisplaceByNormalThreads>()});
    strategies.push_back({"TBB", std::make_unique<DisplaceByNormalTBB>()});
    strategies.push_back({"CUDA", std::make_unique<DisplaceByNormalCuda>()});

    for (auto& [name, strategy] : strategies) {
        std::cout << "Processing: " << name << std::endl;
        VtArray<GfVec3f> usdVerticesCopy = usdVertices;
        strategy->displaceByNormal(usdVerticesCopy, usdFaceVertexIndices, usdFaceVertexCounts);
        if (dynamic_cast<DisplaceByNormalCuda*>(strategy.get())) {
            usdVertices = usdVerticesCopy;
        }
    }

    usdMesh.GetPointsAttr().Set(usdVertices);
    sphereStage->Export(OUTPUT_USD_PATH);
    std::cout << "Done!" << std::endl;
    return 0;
}