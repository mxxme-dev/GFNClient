//! Stats Overlay Module
//!
//! Renders performance statistics overlay using glyphon for text rendering.

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::{MultisampleState, TextureFormat};
use std::time::Instant;

use super::stats::StreamStats;

/// Stats overlay renderer
pub struct Overlay {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    buffer: Buffer,
    visible: bool,
    last_update: Instant,
    needs_prepare: bool,
}

impl Overlay {
    /// Create a new overlay renderer
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: TextureFormat,
    ) -> Self {
        // Initialize font system with system fonts
        let mut font_system = FontSystem::new();

        // Create text rendering components
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            None,
        );

        // Create text buffer
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(16.0, 20.0));
        buffer.set_size(&mut font_system, Some(400.0), Some(200.0));

        Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            buffer,
            visible: true,
            last_update: Instant::now(),
            needs_prepare: true,
        }
    }

    /// Toggle overlay visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Check if overlay is visible
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Set overlay visibility
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Update overlay text with current stats (throttled to 10 updates/sec)
    pub fn update(&mut self, stats: &StreamStats) {
        if !self.visible {
            return;
        }

        // Only update text 10 times per second to reduce CPU/GPU load
        if self.last_update.elapsed().as_millis() < 100 {
            return;
        }
        self.last_update = Instant::now();

        // Format stats text
        let text = format!(
            "FPS: {:.0}/{} (decode: {:.0})\n\
             Latency: {:.1}ms | Jitter: {:.1}ms\n\
             Bitrate: {} Mbps | Loss: {:.2}%\n\
             Video: {} | Audio: {} | Input: {}\n\
             Dropped: {} | {}",
            stats.render_fps,
            stats.target_fps,
            stats.decode_fps,
            stats.latency_ms,
            stats.jitter_ms,
            stats.bitrate_kbps / 1000,
            stats.packet_loss * 100.0,
            stats.video_packets,
            stats.audio_packets,
            stats.input_packets_sent,
            stats.frames_dropped,
            stats.resolution,
        );

        // Update buffer with new text
        self.buffer.set_text(
            &mut self.font_system,
            &text,
            Attrs::new().family(Family::Monospace).color(Color::rgb(255, 255, 255)),
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
        self.needs_prepare = true;
    }

    /// Render the overlay
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if !self.visible {
            return Ok(());
        }

        // Update viewport
        self.viewport.update(
            queue,
            Resolution { width, height },
        );

        // Only prepare text when it changed (throttled in update())
        if self.needs_prepare {
            self.text_renderer
                .prepare(
                    device,
                    queue,
                    &mut self.font_system,
                    &mut self.atlas,
                    &self.viewport,
                    [TextArea {
                        buffer: &self.buffer,
                        left: 10.0,
                        top: 10.0,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: 0,
                            top: 0,
                            right: width as i32,
                            bottom: height as i32,
                        },
                        default_color: Color::rgb(255, 255, 255),
                        custom_glyphs: &[],
                    }],
                    &mut self.swash_cache,
                )
                .map_err(|e| format!("Failed to prepare text: {:?}", e))?;
            self.needs_prepare = false;
        }

        // Create render pass for overlay (no clear - render on top)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Overlay Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear - render on top
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)
                .map_err(|e| format!("Failed to render text: {:?}", e))?;
        }

        Ok(())
    }
}
