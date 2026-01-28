#pragma once
// ============== D3D11/D3D12 BASE SHADERS ==============
// Vertex/Pixel shaders for basic cube rendering and text overlay

static const char* g_d3d11ShaderCode = R"HLSL(
cbuffer CB : register(b0) { float Time; float3 _pad; };

static const matrix View = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,4,1 };
static const matrix Proj = { 1.81066f,0,0,0, 0,2.41421f,0,0, 0,0,1.001f,1, 0,0,-0.1001f,0 };

static const float4 Colors[8] = {
    float4(0.95f,0.2f,0.15f,1), float4(0.2f,0.7f,0.3f,1),
    float4(0.15f,0.5f,0.95f,1), float4(1.0f,0.85f,0.0f,1),
    float4(1.0f,0.85f,0.0f,1), float4(0.15f,0.5f,0.95f,1),
    float4(0.2f,0.7f,0.3f,1), float4(0.95f,0.2f,0.15f,1)
};

static const float3 LightDir = normalize(float3(0.2f, 1.0f, 0.3f));

struct VSIn { float3 pos : POSITION; float3 norm : NORMAL; uint cubeID : CUBEID; };
struct PSIn { float4 pos : SV_POSITION; float3 worldNorm : NORMAL; float4 color : COLOR; };

float3x3 RotY(float a) { float c=cos(a),s=sin(a); return float3x3(c,0,s,0,1,0,-s,0,c); }
float3x3 RotX(float a) { float c=cos(a),s=sin(a); return float3x3(1,0,0,0,c,-s,0,s,c); }

PSIn VS(VSIn i) {
    PSIn o;
    float3x3 rot = mul(RotY(Time*1.2f), RotX(Time*0.7f));
    float3 worldPos = mul(i.pos, rot);
    o.pos = mul(mul(float4(worldPos,1), View), Proj);
    o.worldNorm = mul(i.norm, rot);
    o.color = Colors[i.cubeID];
    return o;
}

float4 PS(PSIn i) : SV_TARGET {
    float3 n = normalize(i.worldNorm);
    float d = max(dot(n, LightDir), 0) * 0.65f + 0.35f;
    return float4(i.color.rgb * d, 1);
}

// ============== TEXT RENDERING SHADERS ==============
struct TextVSIn { float2 pos : POSITION; float2 uv : TEXCOORD; float4 color : COLOR; };
struct TextPSIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 color : COLOR; };

Texture2D fontTexture : register(t0);
SamplerState fontSampler : register(s0);

TextPSIn TextVS(TextVSIn i) {
    TextPSIn o;
    o.pos = float4(i.pos.x/320.0f-1.0f, 1.0f-i.pos.y/240.0f, 0, 1);
    o.uv = i.uv;
    o.color = i.color;
    return o;
}

float4 TextPS(TextPSIn i) : SV_TARGET {
    float alpha = fontTexture.Sample(fontSampler, i.uv).r;
    if (alpha < 0.5f) discard;
    return float4(i.color.rgb, alpha);
}
)HLSL";
