#pragma once

#include "DisplaceByNormalStrategy.h"
#include <pxr/base/vt/array.h>

class DisplaceByNormalThreads : public DisplaceByNormalStrategy
{
public:
    void displaceByNormal(pxr::VtArray<pxr::GfVec3f>& usdVertices, pxr::VtArray<int>& usdFaceVertexIndices, pxr::VtArray<int>& usdFaceVertexCounts) override;
};
