#include "shader_recompiler.h"
#include "shader_common.h"

#include <algorithm>
#include <map>
#include <optional>
#include <regex>
#include <set>

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
    "uint4", // NORMAL
    "float4", // PSIZE
    "float4", // TEXCOORD
    "uint4", // TANGENT
    "uint4", // BINORMAL
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 }
};

static constexpr std::string_view TEXTURE_DIMENSIONS[] = 
{
    "2D",
    "3D", 
    "Cube" 
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

void ShaderRecompiler::openPredicateBlock(bool condition)
{
    const int wanted = condition ? 1 : 0;
    if (openPredicate == wanted)
        return;

    closePredicateBlock();

    indent();
    println("if ({}p0)", condition ? "" : "!");

    indent();
    out += "{\n";
    ++indentation;

    openPredicate = wanted;
}

void ShaderRecompiler::closePredicateBlock()
{
    if (openPredicate < 0)
        return;

    --indentation;
    indent();
    out += "}\n";

    openPredicate = -1;
}

void ShaderRecompiler::closePredicateBlockIfWritten(size_t mark)
{
    // Both places that write the predicate (SetpXXPush vector ops and Setp scalar ops) emit "p0 = ".
    if (out.find("p0 = ", mark) != std::string::npos)
        closePredicateBlock();
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
        openPredicateBlock(instr.predicateCondition);
    else
        closePredicateBlock();

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    auto findResult = vertexElements.find(address);
    assert(findResult != vertexElements.end());

    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        specConstantsMask |= SPEC_CONSTANT_R11G11B10_NORMAL;
        print("tfetchR11G11B10(");
        break;

    case DeclUsage::TexCoord:
        print("tfetchTexcoord(g_SwappedTexcoords, ");
        break;
    }

    print("i{}{}", USAGE_VARIABLES[uint32_t(findResult->second.usage)], uint32_t(findResult->second.usageIndex));

    switch (findResult->second.usage)
    {
    case DeclUsage::Normal:
    case DeclUsage::Tangent:
    case DeclUsage::Binormal:
        out += ')';
        break;

    case DeclUsage::TexCoord:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights)
        return;

    if (instr.isPredicated)
        openPredicateBlock(instr.predCondition);
    else
        closePredicateBlock();

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        print(", {}_Texture2DSize);\n", constNamePtr);
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex, ", constNamePtr, dimension);
    printSrcRegister(componentCount);

    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        print(", float2({}, {}), {}_Texture2DSize", instr.offsetX * 0.5f, instr.offsetY * 0.5f, constNamePtr);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    if (instr.isPredicated)
        openPredicateBlock(instr.predicateCondition);
    else
        closePredicateBlock();

    const size_t predicateMark = out.size();

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    auto op = [&](size_t operand)
        {
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                regFormatted = fmt::format("r{}", reg);
            }
            else
            {
            #ifdef UNLEASHED_RECOMP
                // A relative read may reach any register of the block (float4ConstantRegisters).
                if (instr.const0Relative || instr.const1Relative)
                    float4RelativeAccess = true;
            #endif

                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        if (!isPixelShader && instr.const0Relative && instr.constAddressRegisterRelative)
                        {
                            // Indexed by a0, which vertex shaders load from vertex data (bone indices...):
                            // may read through memory instead (SPEC_CONSTANT_RELATIVE_FROM_MEMORY).
                            specConstantsMask |= SPEC_CONSTANT_RELATIVE_FROM_MEMORY;
                            regFormatted = fmt::format("{}_Rel({} + a0)", constantName, reg - findResult->second->registerIndex);
                        }
                        else
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, instr.const0Relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        assert(!instr.const0Relative && !instr.const1Relative);
                        regFormatted = constantName;
                    }
                }
                else
                {
                    assert(!instr.const0Relative && !instr.const1Relative);
                    regFormatted = fmt::format("c{}", reg);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        hasKill = true;
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        hasKill = true;
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        hasKill = true;
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        hasKill = true;
        indent();
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth";
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            default:
            {
                auto findResult = interpolators.find(instr.vectorDest);
                assert(findResult != interpolators.end());
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    else if (instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        indent();
        println("a0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", op(VECTOR_0));
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        // Xenos has no move instruction: assemblers write a move as MAX dst, src, src.
        // max(a, a) == a exactly (NaN and signed zero included), but NIR lowers fmax(a, a)
        // to fcanonicalize(a) -> fmul(a, 1.0) on NAK, so every "move" became a real
        // multiply. Emit the operand directly when both operands are the same expression.
        // (Idea and measurements from nfsmw-nx, docs/shaders.md.) The a0 write of MaxA is
        // emitted separately above and is not affected.
        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
        case AluVectorOpcode::Min:
        {
            const std::string a = op(VECTOR_0);
            const std::string b = op(VECTOR_1);
            if (a == b)
                out += a;
            else
                print("{}({}, {})", instr.vectorOpcode == AluVectorOpcode::Min ? "min" : "max", a, b);
            break;
        }

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            print("cube(r{}, cubeMapData)", instr.src1Register);
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        // Same as the vector case: MAXs x, x is a scalar move.
        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
        case AluScalarOpcode::Mins:
        {
            const std::string a = op(SCALAR_0);
            const std::string b = op(SCALAR_1);
            if (a == b)
                out += a;
            else
                print("{}({}, {})", instr.scalarOpcode == AluScalarOpcode::Mins ? "min" : "max", a, b);
            break;
        }

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(rcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            print("{0} == 0.0 ? 1.0 : {0}", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        hasKill = true;
        indent();
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    // A predicated setp writes p0: the next instructions must test the new value.
    closePredicateBlockIfWritten(predicateMark);
}

#ifdef UNLEASHED_RECOMP
// An if/else in Xenos control flow is a conditional jump over the "then" part that lands right after
// an unconditional jump over the "else" part. When the only jumps the plain if blocks cannot express
// are such unconditional jumps, and every block and loop nests, the shader is emitted with if/else
// blocks instead of as a pc state machine; loops become do-while loops, which is what the state
// machine runs (the body always runs once). The same operations run in the same order; the driver's
// compiler just gets structured control flow. On success, ifEndLabels holds the closings of the if
// and else blocks and elseEnds the end of the else block of each unconditional jump.
static bool StructureIfElse(const std::vector<ControlFlowInstruction>& cf,
    std::unordered_map<uint32_t, uint32_t>& ifEndLabels, std::unordered_map<uint32_t, uint32_t>& elseEnds)
{
    const uint32_t count = uint32_t(cf.size());
    std::unordered_map<uint32_t, uint32_t> labels = ifEndLabels;
    std::unordered_map<uint32_t, uint32_t> heads; // conditional jump -> the unconditional jump ending its "then" part
    std::unordered_map<uint32_t, uint32_t> ends;

    for (uint32_t pc = 0; pc < count; pc++)
    {
        if (cf[pc].opcode != ControlFlowOpcode::CondJmp)
            continue;

        const auto& jmp = cf[pc].condJmp;
        if (jmp.direction)
            return false;

        if (!jmp.isUnconditional)
            continue;

        const uint32_t elseStart = pc + 1;
        const uint32_t end = jmp.address;
        if (end <= elseStart || end >= count)
            return false;

        // Exactly one jump lands right after it: the conditional jump over the "then" part.
        uint32_t head = UINT32_MAX;
        for (uint32_t other = 0; other < count; other++)
        {
            if (cf[other].opcode == ControlFlowOpcode::CondJmp && cf[other].condJmp.address == elseStart)
            {
                if (head != UINT32_MAX || cf[other].condJmp.isUnconditional || other >= pc)
                    return false;

                head = other;
            }
        }

        if (head == UINT32_MAX)
            return false;

        heads[head] = pc;
        ends[pc] = end;

        auto label = labels.find(elseStart);
        if (label == labels.end() || label->second == 0)
            return false;

        if (--label->second == 0)
            labels.erase(label);

        ++labels[end];
    }

    // Every closing has to close the innermost open block.
    enum class Block { If, Then, Else, Loop };
    std::vector<std::pair<Block, uint32_t>> stack; // closing pc (If, Else), else jump (Then), loop start (Loop)

    for (uint32_t pc = 0; pc < count; pc++)
    {
        auto label = labels.find(pc);
        if (label != labels.end())
        {
            for (uint32_t i = 0; i < label->second; i++)
            {
                if (stack.empty() || (stack.back().first != Block::If && stack.back().first != Block::Else) || stack.back().second != pc)
                    return false;

                stack.pop_back();
            }
        }

        const auto& instr = cf[pc];
        switch (instr.opcode)
        {
        case ControlFlowOpcode::LoopStart:
            stack.emplace_back(Block::Loop, pc);
            break;

        case ControlFlowOpcode::LoopEnd:
            // The do-while body starts right after the loop start.
            if (stack.empty() || stack.back().first != Block::Loop || instr.loopEnd.address != stack.back().second + 1)
                return false;

            stack.pop_back();
            break;

        case ControlFlowOpcode::CondJmp:
            if (instr.condJmp.isUnconditional)
            {
                if (stack.empty() || stack.back().first != Block::Then || stack.back().second != pc)
                    return false;

                stack.back() = { Block::Else, ends[pc] };
            }
            else
            {
                auto head = heads.find(pc);
                if (head != heads.end())
                    stack.emplace_back(Block::Then, head->second);
                else
                    stack.emplace_back(Block::If, uint32_t(instr.condJmp.address));
            }
            break;

        default:
            break;
        }
    }

    if (!stack.empty())
        return false;

    ifEndLabels = std::move(labels);
    elseEnds = std::move(ends);
    return true;
}
#endif

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    assert((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100);
    assert(shaderContainer->constantTableOffset != NULL);

    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;

    const auto constantTableContainer = reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset);
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);

    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;

                // Dynamic UBO (a hardware constant bank on Maxwell) or 64-bit pointer, chosen by
                // SPEC_CONSTANT_CONSTANTS_UBO when the pipeline is created. See shader_common.h.
                println("#define {}(INDEX) select((INDEX) < {}, (UR_CONSTANTS_UBO ? g_Ubo{}.v[{} + min(INDEX, {})] : vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10)), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1,
                    shaderName, constantInfo->registerIndex.get(), tailCount - 1);

                // The same read for an a0-relative index in a vertex shader, which may go through the
                // pointer (the L1 cache) instead of the constant bank: same bytes, same result.
                if (!isPixelShader)
                {
                    println("#define {}_Rel(INDEX) select((INDEX) < {}, ((UR_CONSTANTS_UBO && !UR_REL_FROM_MEMORY) ? g_Ubo{}.v[{} + min(INDEX, {})] : vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10)), 0.0)",
                        constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1,
                        shaderName, constantInfo->registerIndex.get(), tailCount - 1);
                }
            }
            else
            {
                println("#define {} (UR_CONSTANTS_UBO ? g_Ubo{}.v[{}] : vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10))",
                    constantName, shaderName, constantInfo->registerIndex.get(), shaderName, constantInfo->registerIndex * 16);
            }
            
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);

        #ifdef UNLEASHED_RECOMP
            float4ConstantRegisters = std::max<uint32_t>(float4ConstantRegisters,
                constantInfo->registerIndex + constantInfo->registerCount);
        #endif

            break;
        }

        case RegisterSet::Sampler:
        {
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex (UR_CONSTANTS_UBO ? UR_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
                    constantName, TEXTURE_DIMENSIONS[j], j * 64 + constantInfo->registerIndex * 4, j * 64 + constantInfo->registerIndex * 4);
            }

            println("#define {}_SamplerDescriptorIndex (UR_CONSTANTS_UBO ? UR_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
                constantName, std::size(TEXTURE_DIMENSIONS) * 64 + constantInfo->registerIndex * 4, std::size(TEXTURE_DIMENSIONS) * 64 + constantInfo->registerIndex * 4);

            // Size of the 2D texture in this slot (float2 at byte 288 + slot * 8 of the shared constants).
            {
                const uint32_t sizeOffset = 288 + constantInfo->registerIndex * 8;
                println("#define {}_Texture2DSize (UR_CONSTANTS_UBO ? float2(UR_SHARED_FLOAT({}), UR_SHARED_FLOAT({})) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + {}))",
                    constantName, sizeOffset, sizeOffset + 4, sizeOffset);
            }

            samplers.emplace(constantInfo->registerIndex, constantName);
            break;
        }

        }
    }

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = (isPixelShader ? 224 : 256) - constantInfo->registerIndex;
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
                if (!isPixelShader)
                    println("#define {0}_Rel(INDEX) {0}(INDEX)", constantName);
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);

            // float2 at byte 288 + slot * 8: register c18 + slot / 2, components xy or zw.
            println("\tfloat2 {}_Texture2DSize : packoffset(c{}.{});",
                constantName, 18 + constantInfo->registerIndex / 2, (constantInfo->registerIndex % 2) != 0 ? "z" : "x");
        }
    }

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            boolConstants.emplace(constantInfo->registerIndex, constantName);
        }
    }

    out += '\n';

    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\tin float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
    }
    else
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
        for (uint32_t i = 0; i < vertexShader->vertexElementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

        #ifdef UNLEASHED_RECOMP
            if ((vertexElement.usage == DeclUsage::TexCoord && vertexElement.usageIndex == 2 && isMetaInstancer) ||
                (vertexElement.usage == DeclUsage::Position && vertexElement.usageIndex == 1))
            {
                usageType = "uint4";
            }
        #endif

            out += '\t';

            for (auto& usageLocation : USAGE_LOCATIONS)
            {
                if (usageLocation.usage == vertexElement.usage && usageLocation.usageIndex == vertexElement.usageIndex)
                {
                    print("[[vk::location({})]] ", usageLocation.location);
                    break;
                }
            }

            println("in {0} i{1}{2} : {3}{2},", usageType, USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex), USAGE_SEMANTICS[uint32_t(vertexElement.usage)]);

            vertexElements.emplace(uint32_t(vertexElement.address), vertexElement);
        }

    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iVertexId : SV_VertexID,\n";
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #endif

        out += "\tout float4 oPos : SV_Position";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
    }

    out += ")\n";
    out += "{\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";

        // Iteration 0 exists to compute oPos with the reverse-Z projection. Without reverse Z every
        // output it writes is written again, identically, by iteration 1, so skip it outright
        // rather than rely on the driver's compiler to find that the whole pass is dead.
        out += "\t\tif (iterationIndex == 0 && (g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0)\n";
        out += "\t\t\tcontinue;\n";
    }
#endif

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                union
                {
                    uint32_t value;
                    struct
                    {
                        int8_t x;
                        int8_t y;
                        int8_t z;
                        int8_t w;
                    };
                };

                value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, x, y, z, w);
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            println("\tfloat4 r{} = i{}{};", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }

    if (!isPixelShader)
    {
    #ifdef UNLEASHED_RECOMP
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #endif

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            print("\tfloat4 r{} = ", i);
            if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                out += "float4((iPos.xy - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #endif
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint aL = 0;\n";
    out += "\tbool p0 = false;\n";
    out += "\tfloat ps = 0.0;\n";
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
        out += "\tbool gatherMismatch = false;\n";
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    union
    {
        ControlFlowInstruction controlFlow[2];
        struct
        {
            uint32_t code0;
            uint32_t code1;
            uint32_t code2;
            uint32_t code3;
        };
    };

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader->size;
    bool simpleControlFlow = true;
#ifdef UNLEASHED_RECOMP
    std::vector<ControlFlowInstruction> cfInstructions;
#endif

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
        #ifdef UNLEASHED_RECOMP
            cfInstructions.push_back(cfInstr);
        #endif
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional || cfInstr.condJmp.direction)
                    simpleControlFlow = false;
                else
                    ++ifEndLabels[cfInstr.condJmp.address];

                break;
            }
            }

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

#ifdef UNLEASHED_RECOMP
    // Only the instructions the emission below visits.
    cfInstructions.resize(std::min<size_t>(cfInstructions.size(), size_t(instrSize / 12) * 2));

    bool structuredIfElse = false;
    std::unordered_map<uint32_t, uint32_t> elseEnds;
    if (!simpleControlFlow && StructureIfElse(cfInstructions, ifEndLabels, elseEnds))
    {
        simpleControlFlow = true;
        structuredIfElse = true;
    }
#endif

    if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
    instrAddress = 0;
    uint32_t pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : controlFlow)
        {
            if (!simpleControlFlow)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd || cfInstr.opcode == ControlFlowOpcode::CondExecEnd);
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                break;

            case ControlFlowOpcode::LoopStart:
            #ifdef UNLEASHED_RECOMP
                if (structuredIfElse)
                {
                    indent();
                    out += "aL = 0;\n";
                    indent();
                    out += "do\n";
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else
            #endif
                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else 
                {
                    out += "\t\t\taL = 0;\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
            #ifdef UNLEASHED_RECOMP
                if (structuredIfElse)
                {
                    indent();
                    out += "++aL;\n";
                    --indentation;
                    indent();
                    println("}} while (aL < i{}.x);", uint32_t(cfInstr.loopEnd.loopId));
                }
                else
            #endif
                if (simpleControlFlow)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional)
                {
                #ifdef UNLEASHED_RECOMP
                    if (structuredIfElse)
                    {
                        // The end of the "then" part (StructureIfElse).
                        --indentation;
                        indent();
                        out += "}\n";
                        indent();
                        out += "else\n";
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                #endif
                    {
                        assert(!simpleControlFlow);
                        println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\tcontinue;\n";
                    }
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                    #ifdef UNLEASHED_RECOMP
                        // The skinning branch can be specialized per pipeline (SPEC_CONSTANT_BONES_SPECIALIZED).
                        // The renderer reads the value from vertex boolean b0.
                        if (findResult != boolConstants.end() && !isPixelShader && cfInstr.condJmp.boolAddress == 0 &&
                            strcmp(findResult->second, "mrgHasBone") == 0)
                        {
                            specConstantsMask |= SPEC_CONSTANT_BONES_SPECIALIZED | SPEC_CONSTANT_HAS_BONES;
                            println("if (urHasBones(g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        }
                        else
                    #endif
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else
                            println("if (b{} {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            auto instructionCode = code + address * 3;
            
            for (uint32_t i = 0; i < count; i++)
            {
                union
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct
                    {
                        uint32_t code0;
                        uint32_t code1;
                        uint32_t code2;
                    };
                };
            
                code0 = instructionCode[0];
                code1 = instructionCode[1];
                code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            // Predicate blocks must not span the braces of this wrapper.
                            closePredicateBlock();

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            closePredicateBlock();
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            closePredicateBlock();
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            // Never let a predicate block span control flow.
            closePredicateBlock();

            if (shouldReturn)
            {
                if (isPixelShader)
                {
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    // Before the alpha test, so that a verification failure is never clipped away.
                    printGatherMismatchOutput();

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    out += "\tclip(oC0.w - g_AlphaThreshold);\n";

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif

                #ifdef UNLEASHED_RECOMP
                    printBlendSkip();
                #endif

                    // After the alpha test and alpha to coverage, which read oC0.w; the renderer never
                    // marks a component unused that either of them (or blending) needs.
                    printUnusedColorZeroing();
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                #endif
                    {
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    }

                    // This exit returns from main. The other exits (the reverse-Z loop's continue,
                    // the switch's break) reach the end of main, which does the same.
                #ifdef UNLEASHED_RECOMP
                    if (simpleControlFlow && !hasMtxProjection)
                #else
                    if (simpleControlFlow)
                #endif
                        printUnusedOutputsZeroing();
                }

                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                indent();
                out += "}\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (!simpleControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    if (!isPixelShader)
    {
        indentation = 1;
        printUnusedOutputsZeroing();
    }

    out += "}";

    if (isPixelShader)
    {
        rewriteShadowFetches();
    #ifdef UNLEASHED_RECOMP
        const std::string withoutEarlyOut = out;
        const uint32_t maskWithoutEarlyOut = specConstantsMask;
        bool shortRegion = false;
        if (rewriteAlphaTestEarlyOut(shortRegion) && sinkIntoAlphaTestEarlyOut() == 0 && shortRegion)
        {
            out = withoutEarlyOut;
            specConstantsMask = maskWithoutEarlyOut;
            usesQuadOperations = false;
        }
        detectCopyShader();
    #endif
    }
#ifdef UNLEASHED_RECOMP
    else
    {
        detectPositionPassThrough();
    }
#endif
}

void ShaderRecompiler::printUnusedOutputsZeroing()
{
    static_assert(std::size(INTERPOLATORS) * 4 == SPEC_CONSTANT_UNUSED_OUTPUT_COMPONENT_COUNT,
        "One spec constant bit per interpolator component, in INTERPOLATORS order (which is also the location order).");

    // Vulkan only: D3D12 links its specialisation constants and must not see these bits.
    if (!out.empty() && out.back() != '\n')
        out += '\n';
    out += "#ifdef __spirv__\n";
    for (size_t i = 0; i < std::size(INTERPOLATORS); i++)
    {
        for (size_t c = 0; c < 4; c++)
        {
            const size_t bit = i * 4 + c;
            indent();
            println("if (g_SpecUnusedOutputs{} & (1u << {})) o{}{}.{} = 0.0;", bit / 32, bit % 32,
                USAGE_VARIABLES[uint32_t(INTERPOLATORS[i].first)], INTERPOLATORS[i].second, SWIZZLES[c]);
        }
    }
    out += "#endif\n";
}

void ShaderRecompiler::printGatherMismatchOutput()
{
    // The shadow filters are only rewritten once the whole shader exists, so the check is always
    // written; without a rewritten filter gatherMismatch stays false and the compiler drops it.
    if (!out.empty() && out.back() != '\n')
        out += '\n';
    out += "#ifdef __spirv__\n";
    indent();
    out += "if (gatherMismatch) oC0 = float4(1.0, 0.0, 1.0, 1.0);\n";
    out += "#endif\n";
}

void ShaderRecompiler::printUnusedColorZeroing()
{
    // The alpha-to-coverage block before this ends without a line break.
    if (!out.empty() && out.back() != '\n')
        out += '\n';
    out += "#ifdef __spirv__\n";
    for (size_t c = 0; c < 4; c++)
    {
        indent();
        println("if (g_SpecConstants() & (1u << {})) oC0.{} = 0.0;", SPEC_CONSTANT_UNUSED_COLOR_SHIFT + c, SWIZZLES[c]);
    }
    out += "#endif\n";
}

// Where one of the gathers of rewriteShadowFetches() returns the texel a point fetch with a given
// offset reads. Offsets are in half texels, relative to the fetch coordinate.
struct ShadowTap
{
    uint32_t gather;   // Which gather (2x2: always 0; 3x3: A, B, C, D = 0..3).
    char component;    // Gather order: x = (i0, j1), y = (i1, j1), z = (i1, j0), w = (i0, j0).
};

static bool shadowTapFor2x2(int32_t offsetX, int32_t offsetY, ShadowTap& tap)
{
    // One gather at the coordinate: its footprint is the four texels around it, which are the texels of
    // the point fetches half a texel away.
    if ((offsetX != -1 && offsetX != 1) || (offsetY != -1 && offsetY != 1))
        return false;

    tap.gather = 0;
    if (offsetY < 0)
        tap.component = offsetX < 0 ? 'w' : 'z';
    else
        tap.component = offsetX < 0 ? 'x' : 'y';
    return true;
}

static bool shadowTapFor3x3(int32_t offsetX, int32_t offsetY, ShadowTap& tap)
{
    // Four gathers half a texel up-left (A), up-right (B), down-left (C) and down-right (D) of the
    // coordinate return the 2x2 blocks [c-1, c] x [c-1, c], [c, c+1] x [c-1, c], [c-1, c] x [c, c+1] and
    // [c, c+1] x [c, c+1] around the texel c the coordinate falls in, which cover the nine texels.
    static constexpr ShadowTap TAPS[3][3] = // [y + 1][x + 1]
    {
        { { 0, 'w' }, { 0, 'z' }, { 1, 'z' } },
        { { 0, 'x' }, { 0, 'y' }, { 1, 'y' } },
        { { 2, 'x' }, { 2, 'y' }, { 3, 'y' } },
    };

    if ((offsetX != -2 && offsetX != 0 && offsetX != 2) || (offsetY != -2 && offsetY != 0 && offsetY != 2))
        return false;

    tap = TAPS[offsetY / 2 + 1][offsetX / 2 + 1];
    return true;
}

// The shadow maps' manual PCF: consecutive lines "rN.m = tfetch2D(S..., rC.ab, float2(x, y), S...).x;"
// of one sampler and one coordinate, with the 2x2 offsets (+-0.5, +-0.5) or the 3x3 offsets
// (-1..1, -1..1) texels in any order. Consecutive lines with nothing in between (no brace, no case
// label) run one after the other under the same predicate, even when they came from two exec clauses.
// No fetch before the last may write the coordinate the following ones read.
//
// Fetches of other textures may sit between them: they move in front of the filter when they neither
// read nor write a register the filter lines before them write, and do not write the coordinate.
uint32_t ShaderRecompiler::rewriteShadowFetches()
{
    static const std::regex FETCH_LINE(
        R"(^(\t*)r(\d+)\.([xyzw]+) = tfetch2D\((\w+)_Texture2DDescriptorIndex, \4_SamplerDescriptorIndex, r(\d+)\.([xyzw])([xyzw]), float2\((-?[0-9.]+), (-?[0-9.]+)\), \4_Texture2DSize\)\.(x+);$)");
    static const std::regex OTHER_FETCH_LINE(
        R"(^(\t*)r(\d+)\.([xyzw]+) = tfetch2D\((\w+)_Texture2DDescriptorIndex, \4_SamplerDescriptorIndex, r(\d+)\.([xyzw]+), float2\(-?[0-9.]+, -?[0-9.]+\), \4_Texture2DSize\)\.[xyzw]+;$)");

    // The alpha-to-coverage coordinate of another texture: writes nothing but pixelCoord.
    static const std::regex PIXEL_COORD_LINE(
        R"(^(\t*)pixelCoord = getPixelCoord\((\w+)_Texture2DDescriptorIndex, r(\d+)\.([xyzw]+), \2_Texture2DSize\);$)");

    // The components of a register named by a mask or swizzle ("xyw").
    auto componentMask = [](const std::string& components)
        {
            uint32_t mask = 0;
            for (char c : components)
                mask |= 1u << (c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : 3);
            return mask;
        };

    struct Fetch
    {
        std::string indentation;
        uint32_t dstRegister = 0;
        std::string dstComponents;
        std::string name;
        uint32_t srcRegister = 0;
        char srcX = 0;
        char srcY = 0;
        std::string offsetXText;
        std::string offsetYText;
        int32_t offsetX = 0; // Half texels.
        int32_t offsetY = 0;
    };

    struct Filter
    {
        int32_t slot = -1;
        std::vector<size_t> fetchLines; // The filter's fetches, in program order.
        std::vector<size_t> otherLines; // Fetches of other textures between them, in program order.
        size_t end = 0;                 // One past the last line of the span.
    };

    std::vector<std::string> lines;
    {
        size_t start = 0;
        while (start < out.size())
        {
            size_t end = out.find('\n', start);
            if (end == std::string::npos)
                end = out.size();
            lines.emplace_back(out, start, end - start);
            start = end + 1;
        }
    }
    const bool endsWithNewLine = !out.empty() && out.back() == '\n';

    std::vector<std::optional<Fetch>> fetches(lines.size());
    for (size_t i = 0; i < lines.size(); i++)
    {
        std::smatch match;
        if (!std::regex_match(lines[i], match, FETCH_LINE))
            continue;

        Fetch fetch;
        fetch.indentation = match[1];
        fetch.dstRegister = uint32_t(std::stoul(match[2]));
        fetch.dstComponents = match[3];
        fetch.name = match[4];
        fetch.srcRegister = uint32_t(std::stoul(match[5]));
        fetch.srcX = match[6].str()[0];
        fetch.srcY = match[7].str()[0];
        fetch.offsetXText = match[8];
        fetch.offsetYText = match[9];
        fetch.offsetX = int32_t(std::lround(std::stod(fetch.offsetXText) * 2.0));
        fetch.offsetY = int32_t(std::lround(std::stod(fetch.offsetYText) * 2.0));

        // Every written component must come from the red channel.
        if (match[10].length() != fetch.dstComponents.size())
            continue;

        fetches[i] = std::move(fetch);
    }

    auto slotOf = [&](const std::string& name) -> int32_t
        {
            for (auto& [registerIndex, samplerName] : samplers)
            {
                if (name == samplerName)
                    return int32_t(registerIndex);
            }
            if (name.size() > 1 && name[0] == 's' && std::all_of(name.begin() + 1, name.end(), ::isdigit))
                return std::stoi(name.substr(1));
            return -1;
        };

    auto tapOf = [](const Fetch& fetch, uint32_t count, ShadowTap& tap)
        {
            return count == 4 ? shadowTapFor2x2(fetch.offsetX, fetch.offsetY, tap) : shadowTapFor3x3(fetch.offsetX, fetch.offsetY, tap);
        };

    auto matches = [&](size_t first, uint32_t count, Filter& filter) -> bool
        {
            const auto& head = fetches[first];
            if (!head)
                return false;

            // Slot 0 feeds the alpha-to-coverage pixel coordinate and slot 10 is the GI texture: both have
            // special handling in the translator, leave them alone.
            filter = {};
            filter.slot = slotOf(head->name);
            if (filter.slot < 0 || filter.slot >= 16 || filter.slot == 0 || filter.slot == 10)
                return false;

            bool seen[9]{};
            // Components of each register written by the filter's fetches so far (in program order).
            std::vector<uint32_t> written(256, 0);

            size_t i = first;
            while (filter.fetchLines.size() < count)
            {
                if (i >= lines.size())
                    return false;

                const auto& fetch = fetches[i];
                if (fetch && fetch->name == head->name)
                {
                    if (fetch->indentation != head->indentation || fetch->srcRegister != head->srcRegister ||
                        fetch->srcX != head->srcX || fetch->srcY != head->srcY)
                    {
                        return false;
                    }

                    ShadowTap tap;
                    if (!tapOf(*fetch, count, tap))
                        return false;

                    const uint32_t index = (count == 4) ? uint32_t((fetch->offsetY > 0 ? 2 : 0) + (fetch->offsetX > 0 ? 1 : 0)) :
                        uint32_t((fetch->offsetY / 2 + 1) * 3 + (fetch->offsetX / 2 + 1));
                    if (seen[index])
                        return false;
                    seen[index] = true;

                    filter.fetchLines.push_back(i);

                    // The coordinate must be the same for every fetch of the filter.
                    if (filter.fetchLines.size() < count && fetch->dstRegister == head->srcRegister &&
                        (fetch->dstComponents.find(head->srcX) != std::string::npos || fetch->dstComponents.find(head->srcY) != std::string::npos))
                    {
                        return false;
                    }

                    if (fetch->dstRegister < written.size())
                        written[fetch->dstRegister] |= componentMask(fetch->dstComponents);
                }
                else
                {
                    std::smatch match;
                    if (std::regex_match(lines[i], match, PIXEL_COORD_LINE) && match[1] == head->indentation)
                    {
                        // Moves in front of the filter too; it must not read a component a tap before it wrote.
                        const uint32_t srcRegister = uint32_t(std::stoul(match[3]));
                        if (srcRegister >= written.size() || (written[srcRegister] & componentMask(match[4])) != 0)
                            return false;

                        filter.otherLines.push_back(i);
                        i++;
                        continue;
                    }

                    if (!std::regex_match(lines[i], match, OTHER_FETCH_LINE) || match[1] != head->indentation || match[4] == head->name)
                        return false;

                    // It moves in front of the filter: it must not read a component a filter fetch before it
                    // wrote, nor write one (the filter's writes, emitted after it, would win instead), nor
                    // write the filter's coordinate register. Other components of the same registers are fine.
                    const uint32_t dstRegister = uint32_t(std::stoul(match[2]));
                    const uint32_t srcRegister = uint32_t(std::stoul(match[5]));
                    if (dstRegister >= written.size() || srcRegister >= written.size() ||
                        dstRegister == head->srcRegister ||
                        (written[dstRegister] & componentMask(match[3])) != 0 ||
                        (written[srcRegister] & componentMask(match[6])) != 0)
                    {
                        return false;
                    }

                    filter.otherLines.push_back(i);
                }

                i++;
            }

            filter.end = i;
            return true;
        };

    static constexpr const char* GATHER_OFFSETS[4] = { "float2(-0.5, -0.5)", "float2(0.5, -0.5)", "float2(-0.5, 0.5)", "float2(0.5, 0.5)" };

    std::string result;
    result.reserve(out.size() + 4096);
    uint32_t rewritten = 0;

    for (size_t i = 0; i < lines.size();)
    {
        uint32_t count = 0;
        Filter filter;
        for (uint32_t candidate : { 9u, 4u })
        {
            if (matches(i, candidate, filter))
            {
                count = candidate;
                break;
            }
        }

        if (count == 0)
        {
            result += lines[i];
            if (i + 1 < lines.size() || endsWithNewLine)
                result += '\n';
            i++;
            continue;
        }

        for (size_t line : filter.otherLines)
        {
            result += lines[line];
            result += '\n';
        }

        const Fetch& head = *fetches[i];
        const std::string& tab = head.indentation;
        const std::string coordinate = fmt::format("r{}.{}{}", head.srcRegister, head.srcX, head.srcY);
        const uint32_t id = rewritten++;
        const uint32_t gatherCount = (count == 4) ? 1 : 4;

        result += "#ifdef __spirv__\n";
        result += fmt::format("{}[branch] if ((g_SpecConstants() & SPEC_CONSTANT_SHADOW_GATHER) != 0 && "
            "((g_SpecConstants() & SPEC_CONSTANT_SHADOW_GATHER_KNOWN) != 0 || (g_GatherableSlots & (1u << {})) != 0))\n", tab, filter.slot);
    #ifdef UNLEASHED_RECOMP
        gatherSlots |= 1u << filter.slot;
    #endif
        result += fmt::format("{}{{\n", tab);

        for (uint32_t g = 0; g < gatherCount; g++)
        {
            result += fmt::format("{0}\tfloat4 gather{1}_{2} = tfetch2DGather({3}_Texture2DDescriptorIndex, {3}_SamplerDescriptorIndex, {4}, {5}, {3}_Texture2DSize);\n",
                tab, id, g, head.name, coordinate, count == 4 ? "float2(0, 0)" : GATHER_OFFSETS[g]);
        }

        result += fmt::format("{}\t[branch] if (g_SpecConstants() & SPEC_CONSTANT_SHADOW_GATHER_VERIFY)\n", tab);
        result += fmt::format("{}\t{{\n", tab);
        for (size_t line : filter.fetchLines)
        {
            const Fetch& fetch = *fetches[line];
            ShadowTap tap;
            tapOf(fetch, count, tap);
            result += fmt::format("{0}\t\tif (asuint(tfetch2D({1}_Texture2DDescriptorIndex, {1}_SamplerDescriptorIndex, {2}, float2({3}, {4}), {1}_Texture2DSize).x) != asuint(gather{5}_{6}.{7})) gatherMismatch = true;\n",
                tab, fetch.name, coordinate, fetch.offsetXText, fetch.offsetYText, id, tap.gather, tap.component);
        }
        result += fmt::format("{}\t}}\n", tab);

        for (size_t line : filter.fetchLines)
        {
            const Fetch& fetch = *fetches[line];
            ShadowTap tap;
            tapOf(fetch, count, tap);
            result += fmt::format("{}\tr{}.{} = gather{}_{}.{};\n", tab, fetch.dstRegister, fetch.dstComponents, id, tap.gather,
                std::string(fetch.dstComponents.size(), tap.component));
        }

        result += fmt::format("{}}}\n", tab);
        result += fmt::format("{}else\n", tab);
        result += "#endif\n";
        result += fmt::format("{}{{\n", tab);
        for (size_t line : filter.fetchLines)
        {
            result += '\t';
            result += lines[line];
            result += '\n';
        }
        result += fmt::format("{}}}\n", tab);

        i = filter.end;
    }

    out = std::move(result);
    return rewritten;
}

#ifdef UNLEASHED_RECOMP
// The branch of rewriteAlphaTestEarlyOut. The pixels it skips are discarded at the end: by the alpha test
// (the same comparison as the clip, so a NaN alpha, kept by the clip, also runs the code) or by the blend
// skip (SPEC_CONSTANT_BLEND_SKIP_ALPHA, the same comparison as its discard).
static constexpr std::string_view ALPHA_TEST_EARLY_OUT_BRANCH =
    "\t[branch] if (((g_SpecConstants() & (SPEC_CONSTANT_ALPHA_TEST | SPEC_CONSTANT_ALPHA_TEST_EARLY_OUT)) != "
    "(SPEC_CONSTANT_ALPHA_TEST | SPEC_CONSTANT_ALPHA_TEST_EARLY_OUT) || !((oC0.w - g_AlphaThreshold) < 0.0)) && "
    "((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ALPHA) == 0 || !(oC0.w <= 0.0)))";

void ShaderRecompiler::printBlendSkip()
{
    // After the alpha test (a pixel it keeps may still change nothing) and alpha to coverage (the renderer
    // never combines it with a blend skip), before the unused colour channels are zeroed: the blend reads
    // the alpha the shader wrote.
    specConstantsMask |= SPEC_CONSTANT_BLEND_SKIP_ALPHA | SPEC_CONSTANT_BLEND_SKIP_ZERO;
    if (!out.empty() && out.back() != '\n')
        out += '\n';
    out += "#ifdef __spirv__\n";
    indent();
    out += "if ((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ALPHA) != 0 && oC0.w <= 0.0) discard;\n";
    indent();
    out += "if ((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ZERO) != 0 && all(oC0 <= 0.0)) discard;\n";
    out += "#endif\n";
}

bool ShaderRecompiler::rewriteAlphaTestEarlyOut(bool& shortRegion)
{
    static constexpr std::string_view ALPHA_TEST = "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
    static constexpr std::string_view CLIP = "clip(oC0.w - g_AlphaThreshold);";

    std::vector<std::string> lines;
    for (size_t start = 0; start <= out.size();)
    {
        size_t end = out.find('\n', start);
        if (end == std::string::npos)
            end = out.size();
        lines.emplace_back(out, start, end - start);
        start = end + 1;
    }

    // One exit, hence one alpha test: its line holds the test and the clip.
    size_t testLine = SIZE_MAX;
    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find(CLIP) != std::string::npos)
        {
            if (testLine != SIZE_MAX)
                return false;
            testLine = i;
        }
    }

    if (testLine == SIZE_MAX || lines[testLine].find(ALPHA_TEST) == std::string::npos)
        return false;

    auto trimmed = [](const std::string& line)
        {
            size_t first = line.find_first_not_of(" \t");
            return first == std::string::npos ? std::string() : line.substr(first);
        };

    // The gather verification output stays outside, so that it still shows on every pixel.
    size_t regionEnd = testLine;
    if (regionEnd >= 3 && trimmed(lines[regionEnd - 1]) == "#endif" &&
        lines[regionEnd - 2].find("gatherMismatch") != std::string::npos && trimmed(lines[regionEnd - 3]) == "#ifdef __spirv__")
    {
        regionEnd -= 3;
    }

    // The last write of oC0.w (oC0, or a write mask with w).
    auto writesAlpha = [&](const std::string& line)
        {
            const std::string text = trimmed(line);
            if (text.rfind("oC0", 0) != 0)
                return false;

            size_t position = 3;
            bool hasW = true;
            if (position < text.size() && text[position] == '.')
            {
                size_t maskEnd = position + 1;
                while (maskEnd < text.size() && strchr("xyzw", text[maskEnd]) != nullptr)
                    maskEnd++;
                hasW = text.find('w', position + 1) < maskEnd;
                position = maskEnd;
            }

            while (position < text.size() && text[position] == ' ')
                position++;
            if (position < text.size() && strchr("+-*/", text[position]) != nullptr)
                position++;

            return hasW && position + 1 < text.size() && text[position] == '=' && text[position + 1] != '=';
        };

    size_t regionStart = SIZE_MAX;
    for (size_t i = regionEnd; i-- > 0;)
    {
        if (lines[i].find("void main(") != std::string::npos)
            return false;
        if (writesAlpha(lines[i]))
        {
            regionStart = i + 1;
            break;
        }
    }

    if (regionStart == SIZE_MAX || regionEnd < regionStart)
        return false;

    // Alpha written late: the branch skips little by itself, only what sinkIntoAlphaTestEarlyOut moves into
    // it. The caller undoes the rewrite when that is nothing.
    shortRegion = regionEnd - regionStart < 16;

    // Straight arithmetic only: nothing that needs the neighbouring pixels (texture fetches with implicit
    // level of detail, derivatives), no other exit, no loop, nothing unbalanced.
    static constexpr const char* FORBIDDEN[] =
    {
        "tfetch", "getWeights", "getPixelCoord", "computeMipLevel", "ddx", "ddy", "fwidth", "Sample", "Gather",
        "Load(", "texture", "clip(", "discard", "return", "for (", "for(", "while", "gatherMismatch", "#",
    };

    int depth = 0;
    for (size_t i = regionStart; i < regionEnd; i++)
    {
        const std::string& line = lines[i];
        for (const char* token : FORBIDDEN)
        {
            if (line.find(token) != std::string::npos)
                return false;
        }

        if (trimmed(line) == "do")
            return false;

        for (char c : line)
        {
            if (c == '{')
                depth++;
            else if (c == '}' && --depth < 0)
                return false;
        }
    }

    if (depth != 0)
        return false;

    // The same comparison as the clip, so a NaN alpha (kept by the clip) also runs the code.
    lines.insert(lines.begin() + regionEnd, "\t}");
    lines.insert(lines.begin() + regionStart, "\t{");
    lines.insert(lines.begin() + regionStart, std::string(ALPHA_TEST_EARLY_OUT_BRANCH));

    out.clear();
    for (size_t i = 0; i < lines.size(); i++)
    {
        out += lines[i];
        if (i + 1 < lines.size())
            out += '\n';
    }

    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST_EARLY_OUT;
    return true;
}
#endif

#ifdef UNLEASHED_RECOMP
namespace
{
    // The components a statement writes or reads, per variable ("r12", "oC0", "ps", "_t3"...): x, y, z, w are
    // bits 0-3; scalars use bit 0.
    using ComponentSets = std::map<std::string, uint32_t>;

    uint32_t componentMask(std::string_view swizzle)
    {
        if (swizzle.empty())
            return 0xF;

        uint32_t mask = 0;
        for (char c : swizzle)
        {
            if (c == 'x') mask |= 1;
            else if (c == 'y') mask |= 2;
            else if (c == 'z') mask |= 4;
            else if (c == 'w') mask |= 8;
        }
        return mask;
    }

    bool overlaps(const ComponentSets& a, const ComponentSets& b)
    {
        for (const auto& [name, mask] : a)
        {
            auto found = b.find(name);
            if (found != b.end() && (found->second & mask) != 0)
                return true;
        }
        return false;
    }

    void addComponents(ComponentSets& to, const ComponentSets& from)
    {
        for (const auto& [name, mask] : from)
            to[name] |= mask;
    }

    void removeComponents(ComponentSets& from, const ComponentSets& what)
    {
        for (const auto& [name, mask] : what)
        {
            auto found = from.find(name);
            if (found != from.end() && (found->second &= ~mask) == 0)
                from.erase(found);
        }
    }

    std::string trimmedLine(const std::string& line)
    {
        const size_t first = line.find_first_not_of(" \t");
        return first == std::string::npos ? std::string() : line.substr(first);
    }

    // The registers and variables translated code reads and writes, and the sinking's own temporaries (_tN,
    // each written once). Constants (cN), inputs (iN...) and uniforms never change; locals declared inside a
    // block (gather0_0...) stay in it.
    const std::regex& variableReference()
    {
        static const std::regex REFERENCE(R"(\b(r\d+|oC0|ps|p0|pixelCoord|_t\d+)\b(?:\.([xyzw]{1,4})\b)?)");
        return REFERENCE;
    }

    const std::regex& assignment()
    {
        static const std::regex ASSIGNMENT(R"((?:^|[\s{};])(r\d+|oC0|ps|p0|pixelCoord|_t\d+)(?:\.([xyzw]{1,4}))?\s*([-+*/]?=)(?!=))");
        return ASSIGNMENT;
    }

    struct SinkUnit
    {
        std::vector<std::string> lines;
        ComponentSets uses;
        ComponentSets mayDefs;
        ComponentSets mustDefs;
        bool pinned = false;        // must run where it is: side effects, loops, address registers, cube maps
        bool derivatives = false;   // implicit level of detail or derivatives: needs its whole 2x2 quad
        bool gatherPath = false;    // the gather half of a split shadow filter
        bool copy = false;          // a copy of a renamed value back into its register
        size_t construct = SIZE_MAX; // index of the original gather construct (both halves)
        bool loop = false;          // round 9: a loop with its own counter (aL) that nothing else reads
    };

    void analyseUnit(SinkUnit& unit)
    {
        unit.uses.clear();
        unit.mayDefs.clear();
        unit.mustDefs.clear();
        unit.pinned = false;
        unit.derivatives = false;

        std::string text;
        for (const auto& line : unit.lines)
        {
            // The gather verification (SPEC_CONSTANT_SHADOW_GATHER_VERIFY) compares with point fetches; the
            // renderer never combines it with the sinking bit, so those fetches do not pin the unit.
            if (line.find("gatherMismatch = true") != std::string::npos)
                continue;
            text += line;
            text += '\n';
        }

        // Exact in a branch only when the whole 2x2 quad of pixels runs it (the quad-uniform block of
        // SPEC_CONSTANT_ALPHA_TEST_QUAD_SINK): fetches with an implicit level of detail and derivatives.
        static constexpr const char* DERIVATIVES[] =
        {
            "tfetch2D(", "tfetch2DBicubic(", "tfetch3D(", ".Sample", "computeMipLevel", "ddx", "ddy", "fwidth",
        };
        for (const char* token : DERIVATIVES)
        {
            if (text.find(token) != std::string::npos)
                unit.derivatives = true;
        }

        // Never moved. Cube map fetches read cubeMapData, which the analysis does not follow.
        static constexpr const char* PINNING[] =
        {
            "tfetchCube(", "cube(", "cubeMapData", "clip(", "discard", "return", "for (", "for(", "while",
        };
        for (const char* token : PINNING)
        {
            // Round 9: a self-contained loop (SinkUnit::loop) moves as one statement: it computes the same values
            // from the same inputs wherever it runs (the analysis sees every register it reads and writes).
            if (unit.loop && (std::string_view(token) == "for (" || std::string_view(token) == "for("))
                continue;
            if (text.find(token) != std::string::npos)
                unit.pinned = true;
        }

        static const std::regex ADDRESS(R"(\b(a0|aL)\b)");
        static const std::regex ADDRESS_A0(R"(\ba0\b)");
        if (std::regex_search(text, unit.loop ? ADDRESS_A0 : ADDRESS))
            unit.pinned = true;

        for (const auto& line : unit.lines)
        {
            const std::string trimmed = trimmedLine(line);
            if (!trimmed.empty() && trimmed[0] == '#' && !unit.gatherPath)
                unit.pinned = true;
        }

        const bool compound = text.find('{') != std::string::npos || text.find("if (") != std::string::npos;

        // Left-hand sides of plain assignments are not reads.
        std::vector<std::pair<size_t, size_t>> writes;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), assignment()); it != std::sregex_iterator(); ++it)
        {
            const auto& match = *it;
            const uint32_t mask = componentMask(match[2].matched ? match[2].str() : std::string());
            unit.mayDefs[match[1].str()] |= mask;
            if (!compound)
                unit.mustDefs[match[1].str()] |= mask;

            if (match[3].str() == "=")
                writes.emplace_back(size_t(match.position(1)), size_t(match.position(1) + match[1].length() + (match[2].matched ? match[2].length() + 1 : 0)));
            else
                unit.uses[match[1].str()] |= mask;
        }

        for (auto it = std::sregex_iterator(text.begin(), text.end(), variableReference()); it != std::sregex_iterator(); ++it)
        {
            const auto& match = *it;
            const size_t position = size_t(match.position(0));
            bool written = false;
            for (const auto& [start, end] : writes)
            {
                if (position >= start && position < end)
                    written = true;
            }

            if (!written)
            {
                const uint32_t mask = componentMask(match[2].matched ? match[2].str() : std::string());
                unit.uses[match[1].str()] |= mask;
            }
        }
    }

    // Round 9: the right-hand side of a translated statement as a tree, to write one of its components on its own
    // (SinkRenamer::SplitComponents). Only what the ALU translation produces is understood: numbers, registers,
    // constants and temporaries with or without a swizzle, the constant arrays' accessors (g_..., mrg...) with one,
    // floatN(...) constructors, unary minus, + - * /, comparisons, ?: and the component-wise intrinsics; dot and
    // the like give a scalar. Anything else makes the parse fail, and the statement stays whole.
    struct SplitNode
    {
        enum class Kind { NUMBER, VARIABLE, ATOM, CONSTRUCTOR, UNARY, BINARY, TERNARY, CALL, SWIZZLE };

        Kind kind = Kind::NUMBER;
        std::string text;    // the number, variable name, whole accessor text, operator or function name
        std::string swizzle; // VARIABLE, ATOM and SWIZZLE
        std::vector<std::unique_ptr<SplitNode>> children;
        uint32_t width = 0;  // components; 1 is a scalar, which HLSL broadcasts
        bool componentwise = true; // CALL: per component; false: a scalar result (dot, length...)
    };

    class SplitParser
    {
    public:
        SplitParser(const std::string& text, const std::map<std::string, uint32_t>& temporaryWidths)
            : text(text), temporaryWidths(temporaryWidths)
        {
        }

        std::unique_ptr<SplitNode> Parse()
        {
            auto node = ParseTernary();
            SkipSpaces();
            if (node == nullptr || position != text.size())
                return nullptr;
            return node;
        }

    private:
        const std::string& text;
        const std::map<std::string, uint32_t>& temporaryWidths;
        size_t position = 0;

        void SkipSpaces()
        {
            while (position < text.size() && isspace(uint8_t(text[position])))
                position++;
        }

        bool Accept(std::string_view token)
        {
            SkipSpaces();
            if (text.compare(position, token.size(), token) == 0)
            {
                // "=" must not take the first half of "==", nor "<" of "<=".
                const size_t end = position + token.size();
                if ((token == "<" || token == ">" || token == "=") && end < text.size() && text[end] == '=')
                    return false;
                position = end;
                return true;
            }
            return false;
        }

        static uint32_t ComponentIndex(char c)
        {
            switch (c)
            {
            case 'x': return 0;
            case 'y': return 1;
            case 'z': return 2;
            case 'w': return 3;
            default: return 4;
            }
        }

        static bool IsSwizzle(const std::string& s)
        {
            if (s.empty() || s.size() > 4)
                return false;
            for (char c : s)
            {
                if (ComponentIndex(c) == 4)
                    return false;
            }
            return true;
        }

        std::string ParseIdentifier()
        {
            SkipSpaces();
            const size_t start = position;
            while (position < text.size() && (isalnum(uint8_t(text[position])) || text[position] == '_'))
                position++;
            return text.substr(start, position - start);
        }

        uint32_t VariableWidth(const std::string& name) const
        {
            static const std::regex REGISTER(R"(^[rc]\d+$)");
            if (std::regex_match(name, REGISTER) || name == "oC0")
                return 4;
            if (name == "ps" || name == "p0" || name == "FLT_MIN" || name == "FLT_MAX")
                return 1;
            auto found = temporaryWidths.find(name);
            return found != temporaryWidths.end() ? found->second : 0;
        }

        static bool IsComponentwise(const std::string& name)
        {
            static const std::set<std::string> NAMES =
            {
                "saturate", "abs", "rcp", "rsqrt", "sqrt", "log2", "exp2", "frac", "floor", "ceil", "trunc", "round",
                "sign", "min", "max", "clamp", "select", "isnan", "isinf", "asfloat", "asuint", "asint",
            };
            return NAMES.count(name) != 0;
        }

        static bool IsScalarResult(const std::string& name)
        {
            return name == "dot" || name == "length" || name == "distance" || name == "any" || name == "all";
        }

        std::unique_ptr<SplitNode> ParsePostfix(std::unique_ptr<SplitNode> node)
        {
            SkipSpaces();
            if (position < text.size() && text[position] == '.' && position + 1 < text.size() && isalpha(uint8_t(text[position + 1])))
            {
                const size_t save = position;
                position++;
                const std::string swizzle = ParseIdentifier();
                if (!IsSwizzle(swizzle))
                {
                    position = save;
                    return node;
                }

                if (node->kind == SplitNode::Kind::VARIABLE || node->kind == SplitNode::Kind::ATOM)
                {
                    if (!node->swizzle.empty())
                        return nullptr;
                    node->swizzle = swizzle;
                    node->width = uint32_t(swizzle.size());
                    return node;
                }

                auto result = std::make_unique<SplitNode>();
                result->kind = SplitNode::Kind::SWIZZLE;
                result->swizzle = swizzle;
                result->width = uint32_t(swizzle.size());
                for (char c : swizzle)
                {
                    if (node->width != 1 && ComponentIndex(c) >= node->width)
                        return nullptr;
                }
                result->children.push_back(std::move(node));
                return result;
            }
            return node;
        }

        std::unique_ptr<SplitNode> ParsePrimary()
        {
            SkipSpaces();
            if (position >= text.size())
                return nullptr;

            const char c = text[position];
            if (c == '(')
            {
                position++;
                auto inner = ParseTernary();
                if (inner == nullptr || !Accept(")"))
                    return nullptr;
                return ParsePostfix(std::move(inner));
            }

            if (isdigit(uint8_t(c)) || (c == '.' && position + 1 < text.size() && isdigit(uint8_t(text[position + 1]))))
            {
                const size_t start = position;
                while (position < text.size() && (isdigit(uint8_t(text[position])) || text[position] == '.'))
                    position++;
                if (position < text.size() && (text[position] == 'e' || text[position] == 'E'))
                {
                    position++;
                    if (position < text.size() && (text[position] == '+' || text[position] == '-'))
                        position++;
                    while (position < text.size() && isdigit(uint8_t(text[position])))
                        position++;
                }
                if (position < text.size() && (text[position] == 'f' || text[position] == 'u'))
                    position++;

                auto node = std::make_unique<SplitNode>();
                node->kind = SplitNode::Kind::NUMBER;
                node->text = text.substr(start, position - start);
                node->width = 1;
                return node;
            }

            if (!isalpha(uint8_t(c)) && c != '_')
                return nullptr;

            const size_t start = position;
            const std::string name = ParseIdentifier();
            SkipSpaces();
            if (position < text.size() && text[position] == '(')
            {
                // A constant array's accessor: kept as written, with the swizzle it must have.
                if (name.rfind("g_", 0) == 0 || name.rfind("mrg", 0) == 0)
                {
                    int depth = 0;
                    for (; position < text.size(); position++)
                    {
                        if (text[position] == '(')
                            depth++;
                        else if (text[position] == ')' && --depth == 0)
                            break;
                    }
                    if (position >= text.size())
                        return nullptr;
                    position++;

                    auto node = std::make_unique<SplitNode>();
                    node->kind = SplitNode::Kind::ATOM;
                    node->text = text.substr(start, position - start);
                    node->width = 0;
                    return ParsePostfix(std::move(node));
                }

                position++;
                std::vector<std::unique_ptr<SplitNode>> arguments;
                if (!Accept(")"))
                {
                    while (true)
                    {
                        auto argument = ParseTernary();
                        if (argument == nullptr || argument->width == 0)
                            return nullptr;
                        arguments.push_back(std::move(argument));
                        if (Accept(")"))
                            break;
                        if (!Accept(","))
                            return nullptr;
                    }
                }

                auto node = std::make_unique<SplitNode>();
                static const std::regex CONSTRUCTOR(R"(^(float|bool|int|uint)([1-4]?)$)");
                std::smatch match;
                if (std::regex_match(name, match, CONSTRUCTOR))
                {
                    const uint32_t width = match[2].length() != 0 ? uint32_t(std::stoi(match[2].str())) : 1;
                    uint32_t total = 0;
                    for (const auto& argument : arguments)
                        total += argument->width;
                    if (!(total == width || (arguments.size() == 1 && arguments[0]->width == 1)))
                        return nullptr;

                    node->kind = SplitNode::Kind::CONSTRUCTOR;
                    node->text = name;
                    node->width = width;
                }
                else if (IsComponentwise(name) || IsScalarResult(name))
                {
                    uint32_t width = 1;
                    for (const auto& argument : arguments)
                    {
                        if (argument->width != 1)
                        {
                            if (width != 1 && width != argument->width)
                                return nullptr;
                            width = argument->width;
                        }
                    }

                    node->kind = SplitNode::Kind::CALL;
                    node->text = name;
                    node->componentwise = IsComponentwise(name);
                    node->width = node->componentwise ? width : 1;
                }
                else
                {
                    return nullptr;
                }

                node->children = std::move(arguments);
                return ParsePostfix(std::move(node));
            }

            auto node = std::make_unique<SplitNode>();
            node->kind = SplitNode::Kind::VARIABLE;
            node->text = name;
            node->width = VariableWidth(name);
            return ParsePostfix(std::move(node));
        }

        std::unique_ptr<SplitNode> ParseUnary()
        {
            if (Accept("-"))
            {
                auto operand = ParseUnary();
                if (operand == nullptr)
                    return nullptr;
                auto node = std::make_unique<SplitNode>();
                node->kind = SplitNode::Kind::UNARY;
                node->text = "-";
                node->width = operand->width;
                node->children.push_back(std::move(operand));
                return node;
            }
            return ParsePrimary();
        }

        std::unique_ptr<SplitNode> Binary(std::string op, std::unique_ptr<SplitNode> left, std::unique_ptr<SplitNode> right)
        {
            if (left == nullptr || right == nullptr || left->width == 0 || right->width == 0)
                return nullptr;
            if (left->width != 1 && right->width != 1 && left->width != right->width)
                return nullptr;

            auto node = std::make_unique<SplitNode>();
            node->kind = SplitNode::Kind::BINARY;
            node->text = std::move(op);
            node->width = std::max(left->width, right->width);
            node->children.push_back(std::move(left));
            node->children.push_back(std::move(right));
            return node;
        }

        std::unique_ptr<SplitNode> ParseMultiplicative()
        {
            auto node = ParseUnary();
            while (node != nullptr)
            {
                if (Accept("*"))
                    node = Binary("*", std::move(node), ParseUnary());
                else if (Accept("/"))
                    node = Binary("/", std::move(node), ParseUnary());
                else
                    break;
            }
            return node;
        }

        std::unique_ptr<SplitNode> ParseAdditive()
        {
            auto node = ParseMultiplicative();
            while (node != nullptr)
            {
                if (Accept("+"))
                    node = Binary("+", std::move(node), ParseMultiplicative());
                else if (Accept("-"))
                    node = Binary("-", std::move(node), ParseMultiplicative());
                else
                    break;
            }
            return node;
        }

        std::unique_ptr<SplitNode> ParseComparison()
        {
            auto node = ParseAdditive();
            if (node == nullptr)
                return nullptr;

            static constexpr std::string_view OPERATORS[] = { ">=", "<=", "==", "!=", ">", "<" };
            for (auto op : OPERATORS)
            {
                if (Accept(op))
                    return Binary(std::string(op), std::move(node), ParseAdditive());
            }
            return node;
        }

        std::unique_ptr<SplitNode> ParseTernary()
        {
            auto condition = ParseComparison();
            if (condition == nullptr || !Accept("?"))
                return condition;

            auto whenTrue = ParseTernary();
            if (whenTrue == nullptr || !Accept(":"))
                return nullptr;
            auto whenFalse = ParseTernary();
            if (whenFalse == nullptr || condition->width == 0 || whenTrue->width == 0 || whenFalse->width == 0)
                return nullptr;

            const uint32_t width = std::max({ condition->width, whenTrue->width, whenFalse->width });
            for (uint32_t w : { condition->width, whenTrue->width, whenFalse->width })
            {
                if (w != 1 && w != width)
                    return nullptr;
            }

            auto node = std::make_unique<SplitNode>();
            node->kind = SplitNode::Kind::TERNARY;
            node->width = width;
            node->children.push_back(std::move(condition));
            node->children.push_back(std::move(whenTrue));
            node->children.push_back(std::move(whenFalse));
            return node;
        }
    };

    // Whether a scalar sub-expression HLSL broadcasts to several components would be written once per component:
    // only numbers and plain variables may be (a function call, for one, would run once per component).
    bool SplitBroadcastsOnlySimple(const SplitNode& node)
    {
        // The arguments of floatN(a, b, ...) each make components of their own: none is broadcast.
        const bool broadcasts = node.width > 1 && !(node.kind == SplitNode::Kind::CONSTRUCTOR && node.children.size() > 1);
        for (const auto& child : node.children)
        {
            if (broadcasts && child->width == 1 && child->kind != SplitNode::Kind::NUMBER &&
                child->kind != SplitNode::Kind::VARIABLE && child->kind != SplitNode::Kind::ATOM)
            {
                return false;
            }
            if (!SplitBroadcastsOnlySimple(*child))
                return false;
        }

        // A call with a scalar result under a node of several components is broadcast too.
        return node.kind != SplitNode::Kind::CALL || node.componentwise || node.width == 1;
    }

    // The whole expression again (all its components), each operator parenthesised as parsed.
    std::string SplitWhole(const SplitNode& node)
    {
        switch (node.kind)
        {
        case SplitNode::Kind::NUMBER:
            return node.text;

        case SplitNode::Kind::VARIABLE:
        case SplitNode::Kind::ATOM:
            return node.swizzle.empty() ? node.text : node.text + "." + node.swizzle;

        case SplitNode::Kind::SWIZZLE:
            return "(" + SplitWhole(*node.children[0]) + ")." + node.swizzle;

        case SplitNode::Kind::UNARY:
            return "(" + node.text + SplitWhole(*node.children[0]) + ")";

        case SplitNode::Kind::BINARY:
            return "(" + SplitWhole(*node.children[0]) + " " + node.text + " " + SplitWhole(*node.children[1]) + ")";

        case SplitNode::Kind::TERNARY:
            return "(" + SplitWhole(*node.children[0]) + " ? " + SplitWhole(*node.children[1]) + " : " + SplitWhole(*node.children[2]) + ")";

        case SplitNode::Kind::CONSTRUCTOR:
        case SplitNode::Kind::CALL:
        {
            std::string result = node.text + "(";
            for (size_t k = 0; k < node.children.size(); k++)
                result += (k != 0 ? ", " : "") + SplitWhole(*node.children[k]);
            return result + ")";
        }
        }

        return std::string();
    }

    // Component `index` of the expression, as an expression of its own that computes the same value with the same
    // operations in the same order (each operator parenthesised as parsed).
    std::string SplitComponent(const SplitNode& node, uint32_t index)
    {
        static constexpr char COMPONENTS[] = "xyzw";
        const uint32_t i = node.width == 1 ? 0 : index;
        switch (node.kind)
        {
        case SplitNode::Kind::NUMBER:
            return node.text;

        case SplitNode::Kind::VARIABLE:
        case SplitNode::Kind::ATOM:
            if (!node.swizzle.empty())
                return node.text + "." + node.swizzle[node.swizzle.size() == 1 ? 0 : i];
            return node.width == 1 ? node.text : node.text + "." + COMPONENTS[i];

        case SplitNode::Kind::SWIZZLE:
        {
            const char c = node.swizzle[node.swizzle.size() == 1 ? 0 : i];
            const uint32_t inner = c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : 3;
            return SplitComponent(*node.children[0], inner);
        }

        case SplitNode::Kind::CONSTRUCTOR:
        {
            if (node.children.size() == 1 && node.children[0]->width == 1)
                return "(" + node.text.substr(0, node.text.find_first_of("1234")) + ")(" + SplitComponent(*node.children[0], 0) + ")";

            uint32_t offset = 0;
            for (const auto& child : node.children)
            {
                if (i < offset + child->width)
                    return SplitComponent(*child, i - offset);
                offset += child->width;
            }
            return std::string();
        }

        case SplitNode::Kind::UNARY:
            return "(" + node.text + SplitComponent(*node.children[0], i) + ")";

        case SplitNode::Kind::BINARY:
            return "(" + SplitComponent(*node.children[0], i) + " " + node.text + " " + SplitComponent(*node.children[1], i) + ")";

        case SplitNode::Kind::TERNARY:
            return "(" + SplitComponent(*node.children[0], i) + " ? " + SplitComponent(*node.children[1], i) + " : " +
                SplitComponent(*node.children[2], i) + ")";

        case SplitNode::Kind::CALL:
        {
            // A call with a scalar result (dot...) is written whole.
            if (!node.componentwise)
                return SplitWhole(node);

            std::string result = node.text + "(";
            for (size_t k = 0; k < node.children.size(); k++)
            {
                if (k != 0)
                    result += ", ";
                result += SplitComponent(*node.children[k], i);
            }
            return result + ")";
        }
        }

        return std::string();
    }

    // Gives every value a temporary register (rN) or the scalar result (ps) receives in the statements before
    // the early-out a name of its own (_tN, written once), so that reusing a register no longer ties a statement
    // to the ones that write that register before or after it: what is left is the real flow of values. A
    // statement that is not a single plain assignment reads and writes the registers themselves; the values it
    // may read or write are copied back into them first (copy units, which the sinking moves like any other).
    // The renamed statements compute exactly what they did: same operations, same operands.
    class SinkRenamer
    {
    public:
        std::vector<std::string> declarations;

        static bool Renamable(const std::string& name)
        {
            return name == "ps" || (name.size() > 1 && name[0] == 'r' && isdigit(uint8_t(name[1])));
        }

        // The components a reference names: its swizzle, or the whole register (ps is a scalar).
        static std::string Components(const std::string& name, const std::string& swizzle)
        {
            if (!swizzle.empty())
                return swizzle;
            return name == "ps" ? "x" : "xyzw";
        }

        static std::string Original(const std::string& name, char component)
        {
            return name == "ps" ? name : name + "." + component;
        }

        std::string RewriteReads(const std::string& text) const
        {
            static const std::regex REFERENCE(R"(\b(r\d+|ps)\b(?:\.([xyzw]{1,4})\b)?)");

            std::string result;
            size_t last = 0;
            for (auto it = std::sregex_iterator(text.begin(), text.end(), REFERENCE); it != std::sregex_iterator(); ++it)
            {
                const auto& match = *it;
                const std::string name = match[1].str();
                const std::string components = Components(name, match[2].matched ? match[2].str() : std::string());

                bool renamed = false;
                std::vector<std::string> names;
                for (char component : components)
                {
                    auto found = current.find({ name, component });
                    if (found != current.end())
                    {
                        renamed = true;
                        names.push_back(found->second);
                    }
                    else
                    {
                        names.push_back(Original(name, component));
                    }
                }

                result.append(text, last, size_t(match.position(0)) - last);
                if (!renamed)
                {
                    result += match.str(0);
                }
                else if (names.size() == 1)
                {
                    result += names[0];
                }
                else
                {
                    result += fmt::format("float{}(", names.size());
                    for (size_t i = 0; i < names.size(); i++)
                    {
                        if (i != 0)
                            result += ", ";
                        result += names[i];
                    }
                    result += ')';
                }

                last = size_t(match.position(0) + match.length(0));
            }

            result.append(text, last, std::string::npos);
            return result;
        }

        // A single plain assignment "dst[.mask] = expression;" on one line, rewritten; false for anything else.
        bool RewriteStatement(std::string& line)
        {
            static const std::regex STATEMENT(R"(^(\s*)(r\d+|oC0|ps|p0|pixelCoord)(?:\.([xyzw]{1,4}))?\s*=(?!=)\s*(.*);\s*$)");

            if (line.find('{') != std::string::npos || line.find('}') != std::string::npos || line.find('#') != std::string::npos)
                return false;

            std::smatch match;
            if (!std::regex_match(line, match, STATEMENT))
                return false;

            const std::string indentation = match[1].str();
            const std::string destination = match[2].str();
            const std::string swizzle = match[3].matched ? match[3].str() : std::string();
            const std::string expression = RewriteReads(match[4].str());

            if (!Renamable(destination))
            {
                line = indentation + destination + (swizzle.empty() ? "" : "." + swizzle) + " = " + expression + ";";
                return true;
            }

            const std::string components = Components(destination, swizzle);
            const std::string temporary = fmt::format("_t{}", next++);
            declarations.push_back(fmt::format("\t{} {} = 0.0;",
                components.size() == 1 ? std::string("float") : fmt::format("float{}", components.size()), temporary));
            widths[temporary] = uint32_t(components.size());

            line = indentation + temporary + " = " + expression + ";";

            for (size_t i = 0; i < components.size(); i++)
            {
                current[{ destination, components[i] }] = components.size() == 1 ? temporary :
                    fmt::format("{}.{}", temporary, "xyzw"[i]);
            }

            return true;
        }

        // Round 9: a statement RewriteStatement made ("_tN = expression;" with _tN of several components) as one
        // statement per component ("_tN.x = ...;"), when the expression is component-wise (SplitParser): each
        // component is then a statement of its own, which moves or stays by what reads that component alone. A
        // vector statement that feeds, say, the environment map's coordinates with two components and a lighting
        // value with the third no longer keeps the lighting (and all it reads) before the alpha test. Each
        // component is computed with the same operations on the same operands as before. Empty: not split.
        std::vector<std::string> SplitComponents(const std::string& line) const
        {
            static const std::regex STATEMENT(R"(^(\s*)(_t\d+) = (.*);$)");
            std::smatch match;
            if (!std::regex_match(line, match, STATEMENT))
                return {};

            const std::string temporary = match[2].str();
            auto found = widths.find(temporary);
            if (found == widths.end() || found->second < 2)
                return {};

            const std::string expression = match[3].str();
            SplitParser parser(expression, widths);
            auto tree = parser.Parse();
            if (tree == nullptr || tree->width != found->second || !SplitBroadcastsOnlySimple(*tree))
                return {};

            std::vector<std::string> parts;
            for (uint32_t i = 0; i < found->second; i++)
            {
                const std::string component = SplitComponent(*tree, i);
                if (component.empty())
                    return {};
                parts.push_back(fmt::format("{}{}.{} = {};", match[1].str(), temporary, "xyzw"[i], component));
            }
            return parts;
        }

        // Round 9: the two halves of a split shadow filter (exactly one of them runs). Each half's plain
        // one-component assignments to registers ("r10.y = gather0_1.y;", "r10.y = tfetch2D(...).x;") write
        // temporaries of their own instead, the same ones in both halves, and every read in them is renamed:
        // the filter no longer writes the registers, so a later statement that reuses one of them (or a copy
        // back into it) no longer ties it to its place. False, with nothing changed, unless both halves write
        // the same components, each once and in no other way, and read none of them.
        bool RewriteConstruct(std::vector<std::string>& first, std::vector<std::string>& second)
        {
            static const std::regex ONE(R"(^(\s*)(r\d+)\.([xyzw])\s*=(?!=)\s*(.*);\s*$)");
            static const std::regex REFERENCE(R"(\b(r\d+|ps)\b(?:\.([xyzw]{1,4})\b)?)");

            using Component = std::pair<std::string, char>;
            auto collect = [&](const std::vector<std::string>& lines, std::set<Component>& written)
                {
                    for (const auto& line : lines)
                    {
                        std::smatch match;
                        if (std::regex_match(line, match, ONE))
                        {
                            if (!written.insert({ match[2].str(), match[3].str()[0] }).second)
                                return false;
                            continue;
                        }

                        // Any other write of a register, oC0, p0, pixelCoord or a temporary.
                        if (std::regex_search(line, assignment()))
                            return false;
                    }
                    return true;
                };

            std::set<Component> writtenFirst;
            std::set<Component> writtenSecond;
            if (!collect(first, writtenFirst) || !collect(second, writtenSecond) || writtenFirst.empty() || writtenFirst != writtenSecond)
                return false;

            // No read of a component the filter writes (the right-hand sides and every other line).
            auto readsWritten = [&](const std::vector<std::string>& lines)
                {
                    for (const auto& line : lines)
                    {
                        std::smatch match;
                        const std::string text = std::regex_match(line, match, ONE) ? match[4].str() : line;
                        for (auto it = std::sregex_iterator(text.begin(), text.end(), REFERENCE); it != std::sregex_iterator(); ++it)
                        {
                            const std::string name = (*it)[1].str();
                            for (char component : Components(name, (*it)[2].matched ? (*it)[2].str() : std::string()))
                            {
                                if (writtenFirst.count({ name, component }) != 0)
                                    return true;
                            }
                        }
                    }
                    return false;
                };

            if (readsWritten(first) || readsWritten(second))
                return false;

            std::map<Component, std::string> temporaries;
            for (const auto& component : writtenFirst)
            {
                const std::string temporary = fmt::format("_t{}", next++);
                declarations.push_back(fmt::format("\tfloat {} = 0.0;", temporary));
                widths[temporary] = 1;
                temporaries[component] = temporary;
            }

            auto rewrite = [&](std::vector<std::string>& lines)
                {
                    for (auto& line : lines)
                    {
                        std::smatch match;
                        if (std::regex_match(line, match, ONE))
                            line = match[1].str() + temporaries[{ match[2].str(), match[3].str()[0] }] + " = " + RewriteReads(match[4].str()) + ";";
                        else
                            line = RewriteReads(line);
                    }
                };

            rewrite(first);
            rewrite(second);

            for (const auto& [component, temporary] : temporaries)
                current[component] = temporary;

            return true;
        }

        // Copies the renamed values of these components back into their registers, one statement per register.
        std::vector<std::string> CopyBack(const ComponentSets& which)
        {
            std::vector<std::string> copies;
            for (const auto& [name, mask] : which)
            {
                if (!Renamable(name))
                    continue;

                std::string components;
                std::vector<std::string> names;
                for (uint32_t i = 0; i < 4; i++)
                {
                    const char component = "xyzw"[i];
                    if ((mask & (1u << i)) == 0 || (name == "ps" && i != 0))
                        continue;

                    auto found = current.find({ name, component });
                    if (found == current.end())
                        continue;

                    components += component;
                    names.push_back(found->second);
                }

                if (names.empty())
                    continue;

                if (name == "ps")
                {
                    copies.push_back(fmt::format("\tps = {};", names[0]));
                }
                else if (names.size() == 1)
                {
                    copies.push_back(fmt::format("\t{}.{} = {};", name, components, names[0]));
                }
                else
                {
                    std::string list;
                    for (size_t i = 0; i < names.size(); i++)
                        list += (i != 0 ? ", " : "") + names[i];
                    copies.push_back(fmt::format("\t{}.{} = float{}({});", name, components, names.size(), list));
                }
            }
            return copies;
        }

        // After a statement that may have written these components of the registers themselves.
        void Forget(const ComponentSets& which)
        {
            for (const auto& [name, mask] : which)
            {
                for (uint32_t i = 0; i < 4; i++)
                {
                    if ((mask & (1u << i)) != 0)
                        current.erase({ name, "xyzw"[i] });
                }
            }
        }

    private:
        std::map<std::pair<std::string, char>, std::string> current;
        std::map<std::string, uint32_t> widths; // components of each temporary
        uint32_t next = 0;
    };
}

// [Switch] SPEC_CONSTANT_ALPHA_TEST_SINK and SPEC_CONSTANT_ALPHA_TEST_QUAD_SINK. Discarded pixels skip the
// arithmetic between the last write of oC0.w and the alpha test (rewriteAlphaTestEarlyOut); this pass also
// moves out of their way what comes before it and only feeds the kept pixels' colour.
//
// The values the statements before the early-out compute are first given names of their own (SinkRenamer),
// so only the real flow of values ties statements together. A statement is then moved when nothing that
// stays after it reads what it writes, or writes what it reads or writes, so it computes the same values
// where it goes; everything the alpha, the alpha test and the code after the branch read stays.
//
// - With the SINK bit alone, the moved statements run inside the per-pixel branch: the arithmetic and the
//   shadow gathers (no implicit level of detail), but no fetch with an implicit level of detail and no
//   derivative, which need the neighbouring pixels of the quad.
// - With the QUAD bit too, they also include those fetches and derivatives, and run right before the branch
//   for every 2x2 quad in which at least one pixel is kept: all four pixels of such a quad run them as before
//   (so the implicit levels of detail and derivatives are the same), and a quad whose four pixels are all
//   discarded skips them, fetches included.
// - Without the SINK bit, the moved statements run before the branch, where they were.
uint32_t ShaderRecompiler::sinkIntoAlphaTestEarlyOut()
{
    std::vector<std::string> lines;
    for (size_t start = 0; start <= out.size();)
    {
        size_t end = out.find('\n', start);
        if (end == std::string::npos)
            end = out.size();
        lines.emplace_back(out, start, end - start);
        start = end + 1;
    }

    size_t branch = SIZE_MAX;
    size_t bodyStart = SIZE_MAX;
    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i] == ALPHA_TEST_EARLY_OUT_BRANCH)
        {
            if (branch != SIZE_MAX)
                return 0;
            branch = i;
        }
        else if (lines[i] == "\tbool gatherMismatch = false;" && bodyStart == SIZE_MAX)
        {
            bodyStart = i + 1;
        }
    }

    if (branch == SIZE_MAX || bodyStart == SIZE_MAX || bodyStart >= branch || branch + 1 >= lines.size() || lines[branch + 1] != "\t{")
        return 0;

    // The end of the branch.
    size_t close = SIZE_MAX;
    int depth = 0;
    for (size_t i = branch + 1; i < lines.size(); i++)
    {
        for (char c : lines[i])
        {
            if (c == '{')
                depth++;
            else if (c == '}')
                depth--;
        }
        if (depth == 0)
        {
            close = i;
            break;
        }
    }
    if (close == SIZE_MAX)
        return 0;

    // "}<tab>statement;" (the end of an if/else the translator prints on the line of the next statement)
    // becomes two lines, so that the statement is a unit of its own.
    std::vector<std::string> body;
    for (size_t i = bodyStart; i < branch; i++)
    {
        const std::string trimmed = trimmedLine(lines[i]);
        if (trimmed.size() > 1 && trimmed[0] == '}')
        {
            const std::string rest = trimmedLine(trimmed.substr(1));
            if (!rest.empty() && rest.rfind("else", 0) != 0)
            {
                body.push_back("\t}");
                body.push_back("\t" + rest);
                continue;
            }
        }
        body.push_back(lines[i]);
    }

    // The statements before the branch, each with its braces and preprocessor lines.
    std::vector<SinkUnit> units;
    std::vector<std::vector<std::string>> constructs;
    for (size_t i = 0; i < body.size();)
    {
        if (trimmedLine(body[i]).empty())
        {
            i++;
            continue;
        }

        std::vector<std::string> unitLines;
        int braces = 0;
        int preprocessor = 0;
        while (true)
        {
            if (i >= body.size())
                return 0;

            const std::string trimmed = trimmedLine(body[i]);
            if (trimmed.rfind("#if", 0) == 0)
                preprocessor++;
            else if (trimmed.rfind("#endif", 0) == 0)
                preprocessor--;
            for (char c : body[i])
            {
                if (c == '{')
                    braces++;
                else if (c == '}')
                    braces--;
            }

            unitLines.push_back(body[i]);
            i++;

            if (braces == 0 && preprocessor == 0 && !trimmed.empty() && (trimmed.back() == ';' || trimmed.back() == '}'))
            {
                const bool continues = i < body.size() && trimmedLine(body[i]).rfind("else", 0) == 0;
                if (!continues)
                    break;
            }
        }

        // A shadow filter rewritten as gathers (rewriteShadowFetches):
        //     #ifdef __spirv__ / [branch] if (C) / { gathers } / else / #endif / { point fetches }
        // becomes two statements that never both run: the point fetches under !(C), which stay (implicit
        // level of detail), then the gathers under C, which may move.
        static constexpr std::string_view GATHER_IF = "\t[branch] if ((g_SpecConstants() & SPEC_CONSTANT_SHADOW_GATHER) != 0";
        if (unitLines.size() > 6 && trimmedLine(unitLines[0]) == "#ifdef __spirv__" && unitLines[1].rfind(GATHER_IF, 0) == 0)
        {
            size_t elseLine = SIZE_MAX;
            for (size_t l = 2; l < unitLines.size(); l++)
            {
                if (trimmedLine(unitLines[l]) == "else")
                {
                    elseLine = l;
                    break;
                }
            }

            if (elseLine != SIZE_MAX && elseLine + 2 < unitLines.size() && trimmedLine(unitLines[elseLine + 1]) == "#endif")
            {
                const std::string condition = unitLines[1].substr(std::string_view("\t[branch] if ").size());

                SinkUnit fallback;
                fallback.construct = constructs.size();
                fallback.lines.push_back("#ifdef __spirv__");
                fallback.lines.push_back("\t[branch] if (!" + condition + ")");
                fallback.lines.push_back("#endif");
                fallback.lines.insert(fallback.lines.end(), unitLines.begin() + elseLine + 2, unitLines.end());

                SinkUnit gathers;
                gathers.construct = constructs.size();
                gathers.gatherPath = true;
                gathers.lines.push_back("#ifdef __spirv__");
                gathers.lines.insert(gathers.lines.end(), unitLines.begin() + 1, unitLines.begin() + elseLine);
                gathers.lines.push_back("#endif");

                constructs.push_back(unitLines);
                units.push_back(std::move(fallback));
                units.push_back(std::move(gathers));
                continue;
            }
        }

        SinkUnit unit;
        unit.lines = std::move(unitLines);
        units.push_back(std::move(unit));
    }

    // Round 9: loops whose counter (aL, which each loop sets to 0 first) nothing outside them reads: the local
    // light loops, which only compute. Each moves as one statement (SinkUnit::loop).
    {
        static const std::regex LOOP(R"(^\s*(?:\[unroll\]\s*|\[loop\]\s*)?for \(aL = 0; aL < i\d+\.[xyzw]; aL\+\+\)\s*$)");
        static const std::regex COUNTER(R"(\baL\b)");
        bool counterElsewhere = false;
        std::vector<size_t> loops;
        for (size_t i = 0; i < units.size(); i++)
        {
            if (!units[i].lines.empty() && std::regex_match(units[i].lines[0], LOOP))
            {
                loops.push_back(i);
                continue;
            }

            for (const auto& line : units[i].lines)
            {
                if (std::regex_search(line, COUNTER))
                    counterElsewhere = true;
            }
        }

        for (size_t i = branch; i < lines.size(); i++)
        {
            if (std::regex_search(lines[i], COUNTER))
                counterElsewhere = true;
        }

        if (!counterElsewhere)
        {
            for (size_t i : loops)
                units[i].loop = true;
        }
    }

    // Renaming. What the rest of the shader (the branch and the code after it) reads of the registers is
    // copied back at the end.
    SinkRenamer renamer;
    std::vector<SinkUnit> renamed;
    renamed.reserve(units.size() * 2);

    auto pushCopies = [&](const ComponentSets& which)
        {
            for (auto& copy : renamer.CopyBack(which))
            {
                SinkUnit unit;
                unit.copy = true;
                unit.lines.push_back(std::move(copy));
                renamed.push_back(std::move(unit));
            }
        };

    for (size_t i = 0; i < units.size(); i++)
    {
        SinkUnit& unit = units[i];

        if (unit.construct == SIZE_MAX && unit.lines.size() == 1 && renamer.RewriteStatement(unit.lines[0]))
        {
            // Round 9: one statement per component where the expression allows (SplitComponents).
            std::vector<std::string> parts = renamer.SplitComponents(unit.lines[0]);
            if (parts.empty())
            {
                renamed.push_back(std::move(unit));
            }
            else
            {
                for (auto& part : parts)
                {
                    SinkUnit component;
                    component.lines.push_back(std::move(part));
                    renamed.push_back(std::move(component));
                }
            }
            continue;
        }

        // Both halves of a shadow filter together.
        const size_t count = (unit.construct != SIZE_MAX && i + 1 < units.size() && units[i + 1].construct == unit.construct) ? 2 : 1;
        if (count == 2 && renamer.RewriteConstruct(units[i].lines, units[i + 1].lines))
        {
            renamed.push_back(std::move(units[i]));
            renamed.push_back(std::move(units[i + 1]));
            i++;
            continue;
        }

        // Read and written through the registers themselves: the values it reads or may write go back into
        // them first.
        ComponentSets touched;
        ComponentSets written;
        for (size_t k = i; k < i + count; k++)
        {
            SinkUnit analysed = units[k];
            analyseUnit(analysed);
            addComponents(touched, analysed.uses);
            addComponents(touched, analysed.mayDefs);
            addComponents(written, analysed.mayDefs);
        }

        pushCopies(touched);
        for (size_t k = i; k < i + count; k++)
            renamed.push_back(std::move(units[k]));
        renamer.Forget(written);
        i += count - 1;
    }

    {
        SinkUnit rest;
        rest.lines.assign(lines.begin() + branch, lines.end());
        analyseUnit(rest);
        pushCopies(rest.uses);
    }

    for (auto& unit : renamed)
        analyseUnit(unit);

    // What the branch condition and the code after the branch read.
    ComponentSets liveAfter;
    liveAfter["oC0"] |= 0x8;
    {
        SinkUnit after;
        after.lines.assign(lines.begin() + close + 1, lines.end());
        analyseUnit(after);
        addComponents(liveAfter, after.uses);
    }

    // Backward: a statement stays when it is pinned or writes something read after it by code that stays.
    // Then forward checks may pin more (a statement that stays after a moved one and writes what the moved
    // one reads or writes); repeat until nothing changes.
    auto computeKept = [&](bool quadUniform)
        {
            std::vector<bool> kept(renamed.size(), false);
            for (size_t i = 0; i < renamed.size(); i++)
                kept[i] = renamed[i].pinned || (!quadUniform && renamed[i].derivatives);

            for (bool changed = true; changed;)
            {
                changed = false;

                ComponentSets live = liveAfter;
                for (size_t i = renamed.size(); i-- > 0;)
                {
                    if (!kept[i] && overlaps(renamed[i].mayDefs, live))
                    {
                        kept[i] = true;
                        changed = true;
                    }

                    if (kept[i])
                    {
                        removeComponents(live, renamed[i].mustDefs);
                        addComponents(live, renamed[i].uses);
                    }
                }

                for (size_t i = 0; i < renamed.size(); i++)
                {
                    if (kept[i])
                        continue;

                    for (size_t k = i + 1; k < renamed.size(); k++)
                    {
                        // The two halves of one shadow filter never both run.
                        if (!kept[k] || (renamed[k].construct != SIZE_MAX && renamed[k].construct == renamed[i].construct))
                            continue;

                        if (overlaps(renamed[k].mayDefs, renamed[i].uses) || overlaps(renamed[k].mayDefs, renamed[i].mayDefs) ||
                            overlaps(renamed[k].uses, renamed[i].mayDefs))
                        {
                            kept[i] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }

            return kept;
        };

    std::vector<bool> keptPixel = computeKept(false);

    // XENOS_SINK_DEBUG: why each statement stays in the per-pixel variant (the first kept statement after it
    // that reads what it writes, or what else pins it).
    if (getenv("XENOS_SINK_DEBUG") != nullptr)
    {
        for (size_t i = 0; i < renamed.size(); i++)
        {
            std::string first = trimmedLine(renamed[i].lines.empty() ? std::string() : renamed[i].lines[0]);
            if (renamed[i].lines.size() > 1)
                first += fmt::format(" (+{} lines)", renamed[i].lines.size() - 1);
            std::string reason;
            if (!keptPixel[i])
                reason = "moved";
            else if (renamed[i].pinned)
                reason = "pinned";
            else if (renamed[i].derivatives)
                reason = "derivatives";
            else
            {
                for (size_t k = i + 1; k < renamed.size() && reason.empty(); k++)
                {
                    if (keptPixel[k] && overlaps(renamed[i].mayDefs, renamed[k].uses))
                        reason = fmt::format("read by kept #{}", k);
                }
                if (reason.empty() && overlaps(renamed[i].mayDefs, liveAfter))
                    reason = "read after the branch";
                for (size_t k = i + 1; k < renamed.size() && reason.empty(); k++)
                {
                    if (keptPixel[k] && (overlaps(renamed[k].mayDefs, renamed[i].uses) || overlaps(renamed[k].mayDefs, renamed[i].mayDefs)))
                        reason = fmt::format("overwritten by kept #{}", k);
                }
                if (reason.empty())
                    reason = "?";
            }
            sinkDebug += fmt::format("#{} {}: {}\n", i, reason, first);
        }
    }
    std::vector<bool> keptQuad = computeKept(true);

    // Worth a variant only for the gathers or a real amount of arithmetic (copies back not counted); the
    // quad variant for a moved fetch or derivative, or a real amount more.
    uint32_t movedPixel = 0;
    uint32_t movedQuadOnly = 0;
    bool movedGathers = false;
    bool movedDerivatives = false;
    for (size_t i = 0; i < renamed.size(); i++)
    {
        // The quad analysis moves a superset: fewer statements are pinned.
        assert(keptPixel[i] || !keptQuad[i]);
        if (!keptPixel[i])
        {
            movedPixel += renamed[i].copy ? 0 : 1;
            movedGathers |= renamed[i].gatherPath;
        }
        else if (!keptQuad[i])
        {
            movedQuadOnly += renamed[i].copy ? 0 : 1;
            movedDerivatives |= renamed[i].derivatives;
        }
    }

    const bool pixelVariant = movedGathers || movedPixel >= 8;
    if (!pixelVariant)
    {
        // Nothing moves for the per-pixel variant: whatever the quad variant moves is taken where it is.
        movedQuadOnly += movedPixel;
        movedPixel = 0;
        std::fill(keptPixel.begin(), keptPixel.end(), true);
    }

    const bool quadVariant = movedDerivatives || movedQuadOnly >= 8;
    if (!quadVariant)
        keptQuad = keptPixel;

    if (!pixelVariant && !quadVariant)
        return 0;

    // Statements by where they run: kept (everywhere in place), moved in the quad variant only (in place
    // without it), moved in both.
    enum class Place { KEPT, QUAD_ONLY, MOVED };
    std::vector<Place> places(renamed.size());
    for (size_t i = 0; i < renamed.size(); i++)
        places[i] = keptQuad[i] ? Place::KEPT : (keptPixel[i] ? Place::QUAD_ONLY : Place::MOVED);

    auto appendUnits = [&](std::vector<std::string>& to, auto&& filter)
        {
            for (size_t i = 0; i < renamed.size(); i++)
            {
                if (filter(places[i]))
                    to.insert(to.end(), renamed[i].lines.begin(), renamed[i].lines.end());
            }
        };

    static constexpr std::string_view QUAD_BOTH = "(SPEC_CONSTANT_ALPHA_TEST_SINK | SPEC_CONSTANT_ALPHA_TEST_QUAD_SINK)";

    std::vector<std::string> result(lines.begin(), lines.begin() + bodyStart);
    result.insert(result.end(), renamer.declarations.begin(), renamer.declarations.end());

    std::vector<bool> constructEmitted(constructs.size(), false);
    for (size_t i = 0; i < renamed.size(); i++)
    {
        const SinkUnit& unit = renamed[i];
        if (places[i] == Place::MOVED)
            continue;

        if (places[i] == Place::QUAD_ONLY)
        {
            result.push_back(fmt::format("\t[branch] if ((g_SpecConstants() & {}) != {})", QUAD_BOTH, QUAD_BOTH));
            result.push_back("\t{");
            result.insert(result.end(), unit.lines.begin(), unit.lines.end());
            result.push_back("\t}");
            continue;
        }

        if (unit.construct != SIZE_MAX)
        {
            // Both halves kept: the original construct, unchanged.
            bool bothKept = true;
            for (size_t k = 0; k < renamed.size(); k++)
            {
                if (renamed[k].construct == unit.construct && places[k] != Place::KEPT)
                    bothKept = false;
            }

            if (bothKept)
            {
                if (!constructEmitted[unit.construct])
                {
                    result.insert(result.end(), constructs[unit.construct].begin(), constructs[unit.construct].end());
                    constructEmitted[unit.construct] = true;
                }
                continue;
            }
        }

        result.insert(result.end(), unit.lines.begin(), unit.lines.end());
    }

    if (pixelVariant)
    {
        result.push_back("\t[branch] if ((g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST_SINK) == 0)");
        result.push_back("\t{");
        appendUnits(result, [](Place place) { return place == Place::MOVED; });
        result.push_back("\t}");
    }

    if (quadVariant)
    {
        // The branch's condition, per pixel, then for its quad (quad operations include the helper pixels).
        const std::string_view branchLine = ALPHA_TEST_EARLY_OUT_BRANCH;
        const std::string_view prefix = "\t[branch] if (";
        const std::string condition(branchLine.substr(prefix.size(), branchLine.size() - prefix.size() - 1));

        result.push_back("#ifdef __spirv__");
        result.push_back(fmt::format("\t[branch] if ((g_SpecConstants() & {}) == {})", QUAD_BOTH, QUAD_BOTH));
        result.push_back("\t{");
        result.push_back(fmt::format("\t\tuint quadKept = ({}) ? 1u : 0u;", condition));
        result.push_back("\t\tquadKept |= QuadReadAcrossX(quadKept);");
        result.push_back("\t\tquadKept |= QuadReadAcrossY(quadKept);");
        result.push_back("\t\t[branch] if (quadKept != 0u)");
        result.push_back("\t\t{");
        appendUnits(result, [](Place place) { return place != Place::KEPT; });
        result.push_back("\t\t}");
        result.push_back("\t}");
        result.push_back("#endif");
    }

    result.push_back(lines[branch]);
    result.push_back(lines[branch + 1]);
    if (pixelVariant)
    {
        result.push_back(fmt::format("\t[branch] if ((g_SpecConstants() & {}) == SPEC_CONSTANT_ALPHA_TEST_SINK)", QUAD_BOTH));
        result.push_back("\t{");
        appendUnits(result, [](Place place) { return place == Place::MOVED; });
        result.push_back("\t}");
    }
    result.insert(result.end(), lines.begin() + branch + 2, lines.end());

    out.clear();
    for (size_t i = 0; i < result.size(); i++)
    {
        out += result[i];
        if (i + 1 < result.size())
            out += '\n';
    }

    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST_SINK;
    if (quadVariant)
    {
        specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST_QUAD_SINK;
        usesQuadOperations = true;
    }

    return movedPixel + movedQuadOnly;
}

// SHADER_FLAG_POSITION_PASS_THROUGH: the position is (POSITION.xyz, 1) plus the half-pixel offset, so the
// renderer can work out from the vertices alone which pixels a draw covers.
void ShaderRecompiler::detectPositionPassThrough()
{
    positionPassThrough = false;
    if (hasMtxProjection)
        return;

    std::vector<std::string> lines;
    for (size_t start = 0; start <= out.size();)
    {
        size_t end = out.find('\n', start);
        if (end == std::string::npos)
            end = out.size();
        lines.emplace_back(out, start, end - start);
        start = end + 1;
    }

    // The body of main: after its signature (which declares oPos) and its opening brace.
    size_t bodyStart = SIZE_MAX;
    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].rfind("void main(", 0) == 0)
        {
            for (size_t j = i + 1; j < lines.size(); j++)
            {
                if (lines[j] == "{")
                {
                    bodyStart = j + 1;
                    break;
                }
            }
            break;
        }
    }
    if (bodyStart == SIZE_MAX)
        return;

    static const std::regex POSITION_OUTPUT(R"(^\toPos\.xyzw = r(\d+)\.xyzw;$)");
    size_t outputLine = SIZE_MAX;
    std::string registerName;
    for (size_t i = bodyStart; i < lines.size(); i++)
    {
        const std::string& line = lines[i];
        if (line.find("oPos") == std::string::npos)
            continue;

        std::smatch match;
        if (line == "\toPos = 0.0;" || line == "\toPos.xy += g_HalfPixelOffset * oPos.w;")
            continue;
        if (outputLine == SIZE_MAX && std::regex_match(line, match, POSITION_OUTPUT))
        {
            outputLine = i;
            registerName = "r" + match[1].str();
            continue;
        }
        return;
    }

    if (outputLine == SIZE_MAX)
        return;

    // No control flow, and the register written exactly twice before: xyz from POSITION, w = 1.
    const std::regex writesRegister("^\\t" + registerName + "(\\.[xyzw]+)?\\s*[-+*/]?=[^=]");
    bool xyz = false;
    bool w = false;
    for (size_t i = bodyStart; i < outputLine; i++)
    {
        const std::string& line = lines[i];
        if (line.find("if (") != std::string::npos || line.find("for (") != std::string::npos || line.find("switch") != std::string::npos ||
            line.find("[branch]") != std::string::npos || line.find("while") != std::string::npos)
        {
            return;
        }

        if (!std::regex_search(line, writesRegister))
            continue;

        if (line == "\t" + registerName + ".xyz = iPosition0.xyz;" && !xyz)
            xyz = true;
        else if (line == "\t" + registerName + ".w = 1.0;" && !w)
            w = true;
        else
            return;
    }

    positionPassThrough = xyz && w;
    if (!positionPassThrough)
        return;

    // SHADER_FLAG_TEXCOORD_PASS_THROUGH: oTexCoord0.xy is written once (its z and w may come along, as in
    // "oTexCoord0.xyzw = r0.xyzz"), from the x and y of a register whose x and y were written once before, from
    // the TEXCOORD0 input (tfetchTexcoord, which applies g_SwappedTexcoords), with no control flow. The other
    // writes of oTexCoord0 are its zeroing at the start and the unused-component zeroing at the end (never for
    // components the pixel shader reads).
    static const std::regex TEXCOORD_OUTPUT(R"(^\toTexCoord0\.(xy[zw]{0,2}) = (r\d+)\.(xy[xyzw]{0,2});$)");
    size_t texcoordLine = SIZE_MAX;
    std::string texcoordRegister;
    for (size_t i = bodyStart; i < lines.size(); i++)
    {
        const std::string& line = lines[i];
        if (line.find("oTexCoord0") == std::string::npos)
            continue;

        std::smatch match;
        if (line == "\toTexCoord0 = 0.0;" || line.find("g_SpecUnusedOutputs") != std::string::npos)
            continue;
        if (texcoordLine == SIZE_MAX && std::regex_match(line, match, TEXCOORD_OUTPUT) && match[1].length() == match[3].length())
        {
            texcoordLine = i;
            texcoordRegister = match[2].str();
            continue;
        }
        return;
    }

    if (texcoordLine == SIZE_MAX)
        return;

    // Writes of the register's x or y before the output: only the fetch.
    const std::regex writesTexcoordRegister("^\\t" + texcoordRegister + "(?:\\.([xyzw]+))?\\s*[-+*/]?=[^=]");
    bool fetched = false;
    for (size_t i = bodyStart; i < texcoordLine; i++)
    {
        const std::string& line = lines[i];
        if (line.find("if (") != std::string::npos || line.find("for (") != std::string::npos || line.find("switch") != std::string::npos ||
            line.find("[branch]") != std::string::npos || line.find("while") != std::string::npos)
        {
            return;
        }

        std::smatch write;
        if (!std::regex_search(line, write, writesTexcoordRegister))
            continue;

        const std::string mask = write[1].matched ? write[1].str() : std::string("xyzw");
        if (mask.find_first_of("xy") == std::string::npos)
            continue;

        if (line == "\t" + texcoordRegister + ".xy = tfetchTexcoord(g_SwappedTexcoords, iTexCoord0, 0).xy;" && !fetched)
            fetched = true;
        else
            return;
    }

    texcoordPassThrough = fetched;
}

// SHADER_FLAG_PIXEL_COPY / SHADER_FLAG_DEPTH_COPY: the whole body of main, up to the alpha test and the other
// endings every translated pixel shader has, is one of these (r0 holds TEXCOORD0 from the start):
//     [pixelCoord = getPixelCoord(S..., r0.xy, S...);]  rN.xyzw = tfetch2D(S..., r0.xy, float2(0, 0), S...).xyzw;
//     oC0.xyzw = rN.xyzw;
//     [pixelCoord = ...]  rN.x = tfetch2D(S..., r0.xy, float2(0, 0), S...).x;  ps = saturate(rN.x);  oDepth.x = ps;
//     [oC0.xyzw = rN.xxxx;]
// The slot is the one whose descriptor index S names.
void ShaderRecompiler::detectCopyShader()
{
    copyFlags = 0;

    std::vector<std::string> lines;
    for (size_t start = 0; start <= out.size();)
    {
        size_t end = out.find('\n', start);
        if (end == std::string::npos)
            end = out.size();
        lines.emplace_back(out, start, end - start);
        start = end + 1;
    }

    size_t bodyStart = SIZE_MAX;
    bool texcoordInR0 = false;
    for (size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i] == "\tfloat4 r0 = iTexCoord0;")
            texcoordInR0 = true;
        else if (lines[i] == "\tbool gatherMismatch = false;")
        {
            bodyStart = i + 1;
            break;
        }
    }
    if (bodyStart == SIZE_MAX || !texcoordInR0)
        return;

    std::vector<std::string> body;
    for (size_t i = bodyStart; i < lines.size(); i++)
    {
        if (lines[i] == "#ifdef __spirv__" && i + 1 < lines.size() && lines[i + 1].find("if (gatherMismatch)") != std::string::npos)
            break;
        if (!lines[i].empty())
            body.push_back(lines[i]);
    }

    size_t next = 0;
    std::string sampler;
    static const std::regex PIXEL_COORD(R"(^\tpixelCoord = getPixelCoord\((\w+)_Texture2DDescriptorIndex, r0\.xy, \w+_Texture2DSize\);$)");
    std::smatch match;
    if (next < body.size() && std::regex_match(body[next], match, PIXEL_COORD))
        next++;

    static const std::regex FETCH(R"(^\t(r\d+)\.(xyzw|x) = tfetch2D\((\w+)_Texture2DDescriptorIndex, (\w+)_SamplerDescriptorIndex, r0\.xy, float2\(0, 0\), (\w+)_Texture2DSize\)\.(xyzw|x);$)");
    if (next >= body.size() || !std::regex_match(body[next], match, FETCH) || match[2].str() != match[6].str() ||
        match[3].str() != match[4].str() || match[3].str() != match[5].str())
    {
        return;
    }

    const std::string fetched = match[1].str();
    const bool wholeTexel = match[2].str() == "xyzw";
    sampler = match[3].str();
    next++;

    uint32_t flag = 0;
    if (wholeTexel)
    {
        if (next < body.size() && body[next] == "\toC0.xyzw = " + fetched + ".xyzw;")
        {
            next++;
            flag = SHADER_FLAG_PIXEL_COPY;
        }
    }
    else if (next + 1 < body.size() && body[next] == "\tps = saturate(" + fetched + ".x);" && body[next + 1] == "\toDepth.x = ps;")
    {
        next += 2;
        if (next < body.size() && body[next] == "\toC0.xyzw = " + fetched + ".xxxx;")
            next++;
        flag = SHADER_FLAG_DEPTH_COPY;
    }

    if (flag == 0 || next != body.size())
        return;

    // The slot: "#define S_Texture2DDescriptorIndex (UR_CONSTANTS_UBO ? UR_SHARED_UINT(4 * slot) : ...".
    const std::regex DEFINE("^#define " + sampler + R"(_Texture2DDescriptorIndex \(UR_CONSTANTS_UBO \? UR_SHARED_UINT\((\d+)\))");
    for (const auto& line : lines)
    {
        if (std::regex_search(line, match, DEFINE))
        {
            const uint32_t offset = uint32_t(std::stoul(match[1].str()));
            if ((offset % 4) == 0 && offset / 4 <= SHADER_FLAG_COPY_SLOT_MASK)
                copyFlags = flag | ((offset / 4) << SHADER_FLAG_COPY_SLOT_SHIFT);
            return;
        }
    }
}
#endif
