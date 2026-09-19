#include <metal_stdlib>
using namespace metal;

struct PCUniforms {
    float4x4 mvp;
    float    pointSize;
    float    _pad[3];
};

struct VOut {
    float4 pos [[position]];
    float  psize [[point_size]];
    float3 color;
};

vertex VOut pc_vertex(uint vid [[vertex_id]],
                      device const packed_float3* positions [[buffer(0)]],
                      device const packed_float3* colors    [[buffer(1)]],
                      constant PCUniforms&        U         [[buffer(2)]])
{
    VOut o;
    float3 p = float3(positions[vid]);
    // NaN marks a dropped point (out of range or invalid depth); push it off-screen rather
    // than letting the rasteriser see a NaN position.
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        o.pos = float4(0, 0, -10, 1); o.psize = 0; o.color = float3(0);
        return o;
    }
    o.pos = U.mvp * float4(p, 1.0);
    o.psize = U.pointSize;
    o.color = float3(colors[vid]);
    return o;
}

fragment float4 pc_fragment(VOut in [[stage_in]], float2 pc [[point_coord]])
{
    // round dots read much better than squares at this density
    if (length(pc - float2(0.5)) > 0.5) discard_fragment();
    return float4(in.color, 1.0);
}

/// Marker rendering for measurement endpoints and the line between them.
vertex float4 marker_vertex(uint vid [[vertex_id]],
                            device const packed_float3* pts [[buffer(0)]],
                            constant PCUniforms& U [[buffer(2)]])
{
    return U.mvp * float4(float3(pts[vid]), 1.0);
}

fragment float4 marker_fragment(constant float4& color [[buffer(0)]])
{
    return color;
}
