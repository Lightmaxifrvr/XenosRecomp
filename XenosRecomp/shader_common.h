#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

// Vulkan only: read the shader constants from dynamic uniform buffers (set 4, bindings
// 0/1/2) instead of through 64-bit pointers. NVK promotes dynamic UBOs to hardware
// constant banks, also on Maxwell; the pointer path compiles to global memory loads.
// The renderer ORs this bit into every pipeline after masking (it is intentionally not
// part of any shader's specConstantsMask, so D3D12 and DXIL generation are unaffected).
// High bit so that it never clashes with the per-shader bits below.
#define SPEC_CONSTANT_CONSTANTS_UBO     (1 << 8)

// Texture sizes from the shared constants instead of GetDimensions() queries. The renderer writes,
// for every 2D slot, the size of the texture behind the bound descriptor (all views start at mip 0),
// which is exactly what GetDimensions() returns; the arithmetic that uses it is unchanged, so the
// results are bit-identical. With the VERIFY bit, sizes still come from GetDimensions() and every
// 2D sample whose constant size differs returns magenta, to validate the renderer's table on the
// console. Both are ORed into every pipeline by the renderer, like the UBO bit.
#define SPEC_CONSTANT_TEXTURE_SIZE          (1 << 9)
#define SPEC_CONSTANT_TEXTURE_SIZE_VERIFY   (1 << 10)

// Vulkan only: specialization constants 1-3 (g_SpecUnusedOutputs0-2) hold, per pipeline, one bit per
// vertex shader output component that nothing reads: bit 4 * k + c of the 72, spread over the three
// 32-bit constants, is component c (x, y, z, w) of interpolator k in the translator's order
// (TEXCOORD0-15, then COLOR0-1, which is also their location). Either the pipeline's pixel shader
// never reads that component or the pipeline has no pixel shader. The vertex shader then writes 0 to
// it at every exit, so the driver's compiler drops the math and the vertex fetches that only fed it;
// the image cannot change, since nothing reads it. Set by the renderer per pipeline.
#define SPEC_CONSTANT_UNUSED_OUTPUT_COMPONENT_COUNT 72

// Vulkan only, ORed into every pipeline by the renderer after masking, like the UBO bit: shadow-map
// filters, four point fetches at (+-0.5, +-0.5) texels or nine at (-1..1, -1..1) texels of one
// coordinate, read as one or four gathers wherever those return the very same texels (point-filtered,
// single-mip, power-of-two texture: g_GatherableSlots). VERIFY does both and draws magenta where they
// differ.
#define SPEC_CONSTANT_SHADOW_GATHER         (1 << 11)
#define SPEC_CONSTANT_SHADOW_GATHER_VERIFY  (1 << 12)

// Vertex shaders, part of their mask: constant arrays indexed by a0 (bone palettes, instance data),
// whose index differs from one vertex to the next, read through the constant pointer (L1 cache)
// instead of the constant bank, which serializes different addresses within a warp.
#define SPEC_CONSTANT_RELATIVE_FROM_MEMORY  (1 << 13)

// Vertex shaders that branch on mrgHasBone, part of their mask: with SPECIALIZED, the branch takes
// HAS_BONES, which the renderer sets from that same boolean constant per pipeline, so each pipeline
// only contains the path it runs.
#define SPEC_CONSTANT_BONES_SPECIALIZED     (1 << 14)
#define SPEC_CONSTANT_HAS_BONES             (1 << 15)

// Vulkan only, set by the renderer per pipeline: bit (SHIFT + c) means component c of oC0 is neither
// written (colour write mask, no target) nor used by blending, the alpha test or alpha to coverage.
// The pixel shader writes 0 to it at every exit, so its math is dropped.
#define SPEC_CONSTANT_UNUSED_COLOR_SHIFT    16

// Pixel shaders, part of their mask; ORed into every pipeline by the renderer after masking. With the
// alpha test on, the arithmetic between the last write of oC0.w and the alpha test (which has no texture
// fetch and no derivative in it) only runs for pixels the alpha test keeps: the others skip it and are
// discarded, as before. Kept pixels run exactly the same code.
#define SPEC_CONSTANT_ALPHA_TEST_EARLY_OUT  (1 << 20)

// Vulkan only, set by the renderer per pipeline (after masking, like SPEC_CONSTANT_SHADOW_GATHER): every
// slot the pixel shader's gather rewrites read (its cache entry's gatherSlots) is known to hold a gatherable
// texture and sampler for this draw, so the runtime check of g_GatherableSlots and the point-fetch fallback
// are compiled out. The renderer re-selects the pipeline whenever g_GatherableSlots changes.
#define SPEC_CONSTANT_SHADOW_GATHER_KNOWN   (1 << 21)

// Pixel shaders with the alpha test early-out, part of their mask when the translator moved code; set by
// the renderer (never together with SPEC_CONSTANT_SHADOW_GATHER_VERIFY). The code before the early-out
// that only feeds the colour of kept pixels (the shadow gathers and arithmetic, never a texture fetch with
// an implicit level of detail) runs inside it, so discarded pixels skip it too. Kept pixels compute the
// same values; without the bit that code runs before the early-out, as before.
#define SPEC_CONSTANT_ALPHA_TEST_SINK       (1 << 22)

// Pixel shaders, part of every mask; set by the renderer per pipeline when the blend leaves an 8-bit
// (UNORM) target unchanged for such a pixel: the pixel is discarded instead of going through the blend,
// which saves the read and write of the target. _ALPHA: a source alpha of 0 or less makes the source term
// 0 and the destination factor 1 (SRC_ALPHA or ZERO source factors, INV_SRC_ALPHA or ONE destination
// factors, add or reverse subtract); the colour arithmetic is skipped too. _ZERO: all four outputs 0 or
// less (ONE source factors, same destination factors). NaN outputs are never discarded.
#define SPEC_CONSTANT_BLEND_SKIP_ALPHA      (1 << 23)
#define SPEC_CONSTANT_BLEND_SKIP_ZERO       (1 << 24)

// Pixel shaders with SPEC_CONSTANT_ALPHA_TEST_SINK code, part of their mask when the translator could also
// move fetches with an implicit level of detail or derivatives; set by the renderer only together with the
// sink bit. The code that only feeds the colour of kept pixels, those fetches included, runs for every 2x2
// quad in which at least one pixel passes the early-out's test (all four pixels of such a quad run it, as
// before, so levels of detail and derivatives are unchanged) and is skipped by quads whose four pixels are
// all discarded. Vulkan only (quad operations).
#define SPEC_CONSTANT_ALPHA_TEST_QUAD_SINK  (1 << 25)

// Shader cache entry flags (ShaderCacheEntry::flags), written by the translator.
// Vertex shaders: the position is the POSITION input with w = 1 (plus the half-pixel offset), nothing else.
#define SHADER_FLAG_POSITION_PASS_THROUGH   (1 << 0)
// Pixel shaders: a kill instruction can discard pixels whatever the alpha test.
#define SHADER_FLAG_PIXEL_KILL              (1 << 1)
// Pixel shaders that only copy a texel: oC0 = the 2D fetch of slot SHADER_FLAG_COPY_SLOT at TEXCOORD0.xy, all
// four channels unchanged (_PIXEL_COPY), or oDepth = saturate of its x (_DEPTH_COPY; oC0 = its x too). No
// other output, no kill.
#define SHADER_FLAG_PIXEL_COPY              (1 << 2)
#define SHADER_FLAG_DEPTH_COPY              (1 << 3)
// Vertex shaders with SHADER_FLAG_POSITION_PASS_THROUGH that also pass the TEXCOORD0 input's xy through
// unchanged (swapped as g_SwappedTexcoords says, like every texcoord input).
#define SHADER_FLAG_TEXCOORD_PASS_THROUGH   (1 << 4)
#define SHADER_FLAG_COPY_SLOT_SHIFT         8
#define SHADER_FLAG_COPY_SLOT_MASK          0xF

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;
[[vk::constant_id(1)]] const uint g_SpecUnusedOutputs0 = 0;
[[vk::constant_id(2)]] const uint g_SpecUnusedOutputs1 = 0;
[[vk::constant_id(3)]] const uint g_SpecUnusedOutputs2 = 0;

// The same upload buffer blocks that the push constant pointers address, also bound as
// dynamic UBOs. With -fvk-use-dx-layout a float4 array is 16 bytes per element, so any
// 4-byte word B of a block is v[B / 16][(B % 16) / 4] and asuint() returns it bit-exact.
// Sizes: vertex 256 float4 (4096 B), pixel 224 float4 (3584 B), shared 26 float4 (416 B: the
// original 276 bytes, padding to 288, then 16 texture sizes as float2 at byte 288).
struct UrConstantsVertex { float4 v[256]; };
struct UrConstantsPixel  { float4 v[224]; };
struct UrConstantsShared { float4 v[26]; };
[[vk::binding(0, 4)]] ConstantBuffer<UrConstantsVertex> g_UboVertex;
[[vk::binding(1, 4)]] ConstantBuffer<UrConstantsPixel>  g_UboPixel;
[[vk::binding(2, 4)]] ConstantBuffer<UrConstantsShared> g_UboShared;

#define UR_CONSTANTS_UBO   ((g_SpecConstants & SPEC_CONSTANT_CONSTANTS_UBO) != 0)
#define UR_SHARED_UINT(B)  asuint(g_UboShared.v[(B) / 16][((B) % 16) / 4])
#define UR_SHARED_FLOAT(B) g_UboShared.v[(B) / 16][((B) % 16) / 4]

#define g_Booleans                 (UR_CONSTANTS_UBO ? UR_SHARED_UINT(256) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256))
#define g_SwappedTexcoords         (UR_CONSTANTS_UBO ? UR_SHARED_UINT(260) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260))
#define g_HalfPixelOffset          (UR_CONSTANTS_UBO ? float2(UR_SHARED_FLOAT(264), UR_SHARED_FLOAT(268)) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264))
#define g_AlphaThreshold           (UR_CONSTANTS_UBO ? UR_SHARED_FLOAT(272) : vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272))
// Bit s: the texture and sampler bound to 2D slot s make a gather return exactly the texels of point
// fetches (see SPEC_CONSTANT_SHADOW_GATHER).
#define g_GatherableSlots          (UR_CONSTANTS_UBO ? UR_SHARED_UINT(276) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 276))

#define UR_REL_FROM_MEMORY ((g_SpecConstants & SPEC_CONSTANT_RELATIVE_FROM_MEMORY) != 0)

// mrgHasBone for a branch: the specialized value when the pipeline has one, the constant otherwise.
#define urHasBones(value) (((g_SpecConstants & SPEC_CONSTANT_BONES_SPECIALIZED) != 0) ? \
    (((g_SpecConstants & SPEC_CONSTANT_HAS_BONES) != 0) ? 1u : 0u) : (value))

#define g_SpecConstants() g_SpecConstants

#else

#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);

uint g_SpecConstants();

#define urHasBones(value) (value)

#endif

Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

// The size of a 2D texture as float2: from the shared constants, or queried (the original path, and
// the path used while verifying). [branch] so that the unused side disappears when specialised.
float2 getTexture2DSize(Texture2D<float4> texture, float2 size)
{
    float2 result;
    [branch] if ((g_SpecConstants() & (SPEC_CONSTANT_TEXTURE_SIZE | SPEC_CONSTANT_TEXTURE_SIZE_VERIFY)) == SPEC_CONSTANT_TEXTURE_SIZE)
        result = size;
    else
        result = float2(getTexture2DDimensions(texture));
    return result;
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 size)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];

    [branch] if (g_SpecConstants() & SPEC_CONSTANT_TEXTURE_SIZE_VERIFY)
    {
        if (any(size != float2(getTexture2DDimensions(texture))))
            return float4(1.0, 0.0, 1.0, 1.0);
    }

    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DSize(texture, size));
}

// The red channel of the four texels a bilinear fetch at texCoord + offset would blend, in gather order:
// x = left bottom, y = right bottom, z = right top, w = left top. The coordinate is computed exactly as
// tfetch2D computes it. At offset 0 those are the texels of point fetches at (-0.5, +0.5), (+0.5, +0.5),
// (+0.5, -0.5) and (-0.5, -0.5) texels (for a power-of-two size, where these sums are exact).
float4 tfetch2DGather(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 size)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.GatherRed(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DSize(texture, size));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 size)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DSize(texture, size) + offset - 0.5));
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 size)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    // Same values as the former uint2 (converted to float where it was used).
    float2 dimensions = getTexture2DSize(texture, size);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    // 32-bit mask test (semantic indices are below 16): the same result as the former 64-bit
    // shift, without 64-bit integer operations in every vertex shader that reads texcoords.
    return (swappedTexcoords & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord, float2 size)
{
    return getTexture2DSize(g_Texture2DDescriptorHeap[resourceDescriptorIndex], size) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
