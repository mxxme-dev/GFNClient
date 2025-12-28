// NV12 YUV-to-RGB Video Shader
// Optimized for low-latency 240fps streaming
// Uses BT.709 with FULL RANGE YUV (0-255) - common for streaming

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

// Y plane texture (luma) - R8 format, full resolution
@group(0) @binding(0)
var t_y: texture_2d<f32>;

// UV plane texture (chroma) - RG8 format, half resolution
@group(0) @binding(1)
var t_uv: texture_2d<f32>;

// Sampler for both textures
@group(0) @binding(2)
var s_video: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample Y plane (luma) - already normalized [0, 1]
    let y = textureSample(t_y, s_video, in.tex_coords).r;

    // Sample UV plane (chroma) - convert from [0, 1] to [-0.5, 0.5]
    let uv = textureSample(t_uv, s_video, in.tex_coords).rg;
    let u = uv.r - 0.5;
    let v = uv.g - 0.5;

    // BT.709 full range YUV to RGB conversion
    // R = Y + 1.5748 * V
    // G = Y - 0.1873 * U - 0.4681 * V
    // B = Y + 1.8556 * U
    let r = y + 1.5748 * v;
    let g = y - 0.1873 * u - 0.4681 * v;
    let b = y + 1.8556 * u;

    return vec4<f32>(
        saturate(r),
        saturate(g),
        saturate(b),
        1.0
    );
}
