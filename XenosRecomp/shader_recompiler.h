#pragma once

#include "shader.h"
#include "shader_code.h"

struct StringBuffer
{
    std::string out;

    template<class... Args>
    void print(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    }

    template<class... Args>
    void println(fmt::format_string<Args...> fmt, Args&&... args)
    {
        fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
        out += '\n';
    }
};

struct ShaderRecompiler : StringBuffer
{
    uint32_t indentation = 0;
    bool isPixelShader = false;
    const uint8_t* constantTableData = nullptr;
    std::unordered_map<uint32_t, VertexElement> vertexElements;
    std::unordered_map<uint32_t, std::string> interpolators;
    std::unordered_map<uint32_t, const ConstantInfo*> float4Constants;
    std::unordered_map<uint32_t, const char*> boolConstants;
    std::unordered_map<uint32_t, const char*> samplers;
    std::unordered_map<uint32_t, uint32_t> ifEndLabels;
    uint32_t specConstantsMask = 0;

    // Consecutive instructions predicated on the same condition share one `if (p0)` block instead
    // of one block each. The block is closed before any instruction that is unpredicated or uses the
    // other condition, right after any instruction that writes p0 (later instructions must test the
    // new value), and at the end of every exec clause. Inside a block p0 cannot change, so every
    // instruction runs under exactly the same condition as before: the output is identical.
    // -1: no block open, 0: `if (!p0)` open, 1: `if (p0)` open.
    int openPredicate = -1;
    void openPredicateBlock(bool condition);
    void closePredicateBlock();
    void closePredicateBlockIfWritten(size_t mark);

    // Vertex shaders: writes 0 to every output component the pipeline marks unused (g_SpecUnusedOutputs0-2).
    void printUnusedOutputsZeroing();

    // Pixel shaders, at every exit: magenta where a verified gather differed, before the alpha test.
    void printGatherMismatchOutput();
    // Pixel shaders, at every exit: 0 in the oC0 components the pipeline marks unused.
    void printUnusedColorZeroing();

    // Shadow-map filters: four point fetches at (+-0.5, +-0.5) texels of one coordinate (2x2), or nine at
    // (-1..1, -1..1) texels (3x3), read with one or four gathers where the renderer says that is exact
    // (SPEC_CONSTANT_SHADOW_GATHER). A pass over the finished HLSL of a pixel shader, because the fetches
    // of one filter are often split over two exec clauses. Returns the number of filters rewritten.
    uint32_t rewriteShadowFetches();

#ifdef UNLEASHED_RECOMP
    bool hasMtxProjection = false;
    bool hasMtxPrevInvViewProjection = false;

    // Pixel shaders: SPEC_CONSTANT_ALPHA_TEST_EARLY_OUT. A pass over the finished HLSL; returns whether
    // the shader got it, and in shortRegion whether the branch holds little of its own (alpha written late).
    bool rewriteAlphaTestEarlyOut(bool& shortRegion);

    // Pixel shaders: the discards of SPEC_CONSTANT_BLEND_SKIP_ALPHA and _ZERO, in the epilogue.
    void printBlendSkip();

    // Float4 constant registers the shader can read, counted from c0: the end of the last constant it
    // declares, or the whole block (224 pixel / 256 vertex registers) when an array is indexed by a0 or
    // aL, since such a read may run past the array. The renderer uploads only these registers.
    uint32_t float4ConstantRegisters = 0;
    bool float4RelativeAccess = false;

    // Pixel shaders: the texture slots read by gathers of rewriteShadowFetches() (bit s = slot s).
    uint32_t gatherSlots = 0;

    // Pixel shaders: SPEC_CONSTANT_ALPHA_TEST_SINK and _QUAD_SINK, run after rewriteAlphaTestEarlyOut.
    // Returns the number of statements moved out of the discarded pixels' way.
    uint32_t sinkIntoAlphaTestEarlyOut();
    // The shader uses quad operations (the _QUAD_SINK variant): its SPIR-V needs Vulkan 1.1 (SPIR-V 1.3).
    bool usesQuadOperations = false;

    void detectPositionPassThrough();
    // Pixel shaders: SHADER_FLAG_PIXEL_COPY / SHADER_FLAG_DEPTH_COPY and the slot, in copyFlags.
    void detectCopyShader();
    uint32_t copyFlags = 0;
    // Vertex shaders: SHADER_FLAG_TEXCOORD_PASS_THROUGH (detectPositionPassThrough sets it).
    bool texcoordPassThrough = false;
    // XENOS_SINK_DEBUG set: why each statement before the alpha test stays in place (the dump writes it).
    std::string sinkDebug;
#endif

    // SHADER_FLAG_* for the cache entry: a kill instruction was translated (pixel shaders), the position
    // is the POSITION input passed through (vertex shaders, detectPositionPassThrough).
    bool hasKill = false;
    bool positionPassThrough = false;

    void indent()
    {
        for (uint32_t i = 0; i < indentation; i++)
            out += '\t';
    }

    void printDstSwizzle(uint32_t dstSwizzle, bool operand);
    void printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle);

    void recompile(const VertexFetchInstruction& instr, uint32_t address);
    void recompile(const TextureFetchInstruction& instr, bool bicubic);
    void recompile(const AluInstruction& instr);

    void recompile(const uint8_t* shaderData, const std::string_view& include);
};
