//! GPU Renderer Module
//!
//! Handles hardware-accelerated rendering of video frames using wgpu.
//! Supports Vulkan, Metal, DX12, and WebGPU backends.

use std::sync::Arc;

use anyhow::{Result, Context};
use bytemuck::{Pod, Zeroable};
use log::{info, debug};
use wgpu::util::DeviceExt;
use raw_window_handle::{RawWindowHandle, RawDisplayHandle, HasWindowHandle, HasDisplayHandle, WindowHandle, DisplayHandle};

use super::decoder::DecodedFrame;
use super::stats::StreamStats;
use super::overlay::Overlay;

/// Wrapper to make raw window handles work with wgpu
struct RawWindowWrapper {
    window: RawWindowHandle,
    display: RawDisplayHandle,
}

unsafe impl Send for RawWindowWrapper {}
unsafe impl Sync for RawWindowWrapper {}

impl HasWindowHandle for RawWindowWrapper {
    fn window_handle(&self) -> std::result::Result<WindowHandle<'_>, raw_window_handle::HandleError> {
        // SAFETY: The window handle is valid for the lifetime of the wrapper
        Ok(unsafe { WindowHandle::borrow_raw(self.window) })
    }
}

impl HasDisplayHandle for RawWindowWrapper {
    fn display_handle(&self) -> std::result::Result<DisplayHandle<'_>, raw_window_handle::HandleError> {
        // SAFETY: The display handle is valid for the lifetime of the wrapper
        Ok(unsafe { DisplayHandle::borrow_raw(self.display) })
    }
}

/// Vertex for fullscreen quad
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Fullscreen quad vertices
const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0], tex_coords: [0.0, 1.0] }, // Bottom-left
    Vertex { position: [1.0, -1.0], tex_coords: [1.0, 1.0] },  // Bottom-right
    Vertex { position: [1.0, 1.0], tex_coords: [1.0, 0.0] },   // Top-right
    Vertex { position: [-1.0, 1.0], tex_coords: [0.0, 0.0] },  // Top-left
];

const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

/// GPU renderer for video frames with NV12 YUV-to-RGB conversion
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    // Y plane texture (luma) - R8 format
    y_texture: wgpu::Texture,
    y_texture_view: wgpu::TextureView,
    // UV plane texture (chroma) - RG8 format (U and V interleaved)
    uv_texture: wgpu::Texture,
    uv_texture_view: wgpu::TextureView,
    texture_sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    current_texture_size: (u32, u32),
    // Stats overlay
    overlay: Overlay,
}

impl Renderer {
    /// Create a new renderer from raw window handles (for SDL2)
    pub async fn new_from_raw(
        window_handle: RawWindowHandle,
        display_handle: RawDisplayHandle,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // Create wrapper for raw handles
        let wrapper = Arc::new(RawWindowWrapper {
            window: window_handle,
            display: display_handle,
        });

        // Create wgpu instance with all backends
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface from raw handles
        let surface = instance.create_surface(wrapper.clone())
            .context("Failed to create surface from raw handles")?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find suitable GPU adapter")?;

        let backend = adapter.get_info().backend;
        info!("Using GPU: {} ({:?})", adapter.get_info().name, backend);

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GFN Streamer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("Failed to create device")?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Select lowest latency present mode available for 240fps streaming
        // Priority: Immediate (lowest latency, may tear) > Mailbox > Fifo (VSync)
        let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            info!("Using Immediate present mode (lowest latency for 240fps)");
            wgpu::PresentMode::Immediate
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            info!("Using Mailbox present mode (low latency, no tearing)");
            wgpu::PresentMode::Mailbox
        } else {
            info!("Using Fifo present mode (VSync - will limit to 60fps)");
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1, // Minimize frame latency
        };
        surface.configure(&device, &config);

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Video Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create NV12 textures for video frame (start with 1920x1080)
        // Y plane: full resolution, R8 format (luma only)
        // UV plane: half resolution, RG8 format (U and V interleaved)
        let initial_size = (1920u32, 1080u32);

        let y_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Y Plane Texture"),
            size: wgpu::Extent3d {
                width: initial_size.0,
                height: initial_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let uv_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UV Plane Texture"),
            size: wgpu::Extent3d {
                width: initial_size.0 / 2,
                height: initial_size.1 / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let y_texture_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_texture_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout for NV12 (Y texture + UV texture + sampler)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NV12 Bind Group Layout"),
            entries: &[
                // Y plane texture (luma)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // UV plane texture (chroma)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NV12 Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&uv_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create stats overlay
        let overlay = Overlay::new(&device, &queue, surface_format);
        info!("Stats overlay initialized (F3 to toggle)");

        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            y_texture,
            y_texture_view,
            uv_texture,
            uv_texture_view,
            texture_sampler,
            bind_group,
            bind_group_layout,
            current_texture_size: initial_size,
            overlay,
        })
    }

    /// Resize the surface
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            debug!("Surface resized to {}x{}", width, height);
        }
        Ok(())
    }

    /// Upload a decoded NV12 frame to the GPU
    pub fn upload_frame(&mut self, frame: &DecodedFrame) -> Result<()> {
        // Recreate textures if size changed
        if frame.width != self.current_texture_size.0
            || frame.height != self.current_texture_size.1
        {
            info!("Frame size changed to {}x{}, recreating NV12 textures", frame.width, frame.height);
            self.recreate_texture(frame.width, frame.height)?;
        }

        // Validate Y plane size
        let expected_y_size = (frame.width * frame.height) as usize;
        if frame.y_data.len() != expected_y_size {
            anyhow::bail!(
                "Y plane size mismatch: expected {} bytes, got {} bytes",
                expected_y_size,
                frame.y_data.len()
            );
        }

        // Validate UV plane size
        let expected_uv_size = (frame.width * frame.height / 2) as usize;
        if frame.uv_data.len() != expected_uv_size {
            anyhow::bail!(
                "UV plane size mismatch: expected {} bytes, got {} bytes",
                expected_uv_size,
                frame.uv_data.len()
            );
        }

        // Upload Y plane (luma) - R8 format, full resolution
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.y_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            frame.y_data.as_slice(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(frame.width), // 1 byte per pixel (R8)
                rows_per_image: Some(frame.height),
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );

        // Upload UV plane (chroma) - RG8 format, half resolution
        // UV data is interleaved: U0 V0 U1 V1 ...
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.uv_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            frame.uv_data.as_slice(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(frame.width), // 2 bytes per pixel (RG8), but half width
                rows_per_image: Some(frame.height / 2),
            },
            wgpu::Extent3d {
                width: frame.width / 2,
                height: frame.height / 2,
                depth_or_array_layers: 1,
            },
        );

        debug!("Uploaded NV12 frame {}x{} to GPU", frame.width, frame.height);
        Ok(())
    }

    /// Recreate NV12 textures with new size
    fn recreate_texture(&mut self, width: u32, height: u32) -> Result<()> {
        info!("Recreating NV12 textures: {}x{}", width, height);

        // Create Y plane texture (full resolution, R8)
        self.y_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Y Plane Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create UV plane texture (half resolution, RG8)
        self.uv_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("UV Plane Texture"),
            size: wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.y_texture_view = self.y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.uv_texture_view = self.uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bind group with new texture views
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("NV12 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.y_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.uv_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                },
            ],
        });

        self.current_texture_size = (width, height);
        Ok(())
    }

    /// Render a frame with stats overlay
    pub fn render(&mut self, width: u32, height: u32, stats: &StreamStats) -> Result<()> {
        let output = self.surface.get_current_texture()
            .context("Failed to get surface texture")?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Render video frame
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Video Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        // Update and render stats overlay
        self.overlay.update(stats);
        if let Err(e) = self.overlay.render(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            width,
            height,
        ) {
            debug!("Overlay render error: {}", e);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Toggle stats overlay visibility
    pub fn toggle_overlay(&mut self) {
        self.overlay.toggle();
    }

    /// Check if overlay is visible
    pub fn is_overlay_visible(&self) -> bool {
        self.overlay.is_visible()
    }
}
