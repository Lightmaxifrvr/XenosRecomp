#pragma once

struct DxcCompiler
{
    IDxcCompiler3* dxcCompiler = nullptr;

    DxcCompiler();
    ~DxcCompiler();

    // vulkan11: SPIR-V for Vulkan 1.1 (SPIR-V 1.3), which quad operations need; otherwise Vulkan 1.0 as before.
    IDxcBlob* compile(const std::string& shaderSource, bool compilePixelShader, bool compileLibrary, bool compileSpirv, bool vulkan11 = false);
};
