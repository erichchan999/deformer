#pragma once

#include <pxr/base/vt/array.h>

class DisplaceByNormalStrategy
{
public:
    virtual ~DisplaceByNormalStrategy() = default;
    virtual void displaceByNormal(pxr::VtArray<pxr::GfVec3f>& usdVertices, pxr::VtArray<int>& usdFaceVertexIndices, pxr::VtArray<int>& usdFaceVertexCounts) = 0;
};