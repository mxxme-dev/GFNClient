//! Video Decoder Module
//!
//! Hardware-accelerated H.264/H.265 video decoding using FFmpeg with NVDEC (CUVID).
//! Optimized for low-latency streaming with minimal CPU copies.

use anyhow::{Result, Context};
use log::{info, warn, debug};
use std::sync::Arc;

/// Video codec type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoCodec {
    H264,
    H265,
    Unknown,
}

/// Decoded video frame ready for rendering
#[derive(Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    pub y_data: Arc<Vec<u8>>,
    pub uv_data: Arc<Vec<u8>>,
    pub pts: u64,
}

/// FFmpeg NVDEC decoder with optimized frame handling
pub struct VideoDecoder {
    decoder: ffmpeg_next::decoder::Video,
    frames_decoded: u64,
    nal_buffer: Vec<u8>,
    width: u32,
    height: u32,
    is_hardware: bool,
    // Pre-allocated buffers to avoid allocation per frame
    y_buffer: Vec<u8>,
    uv_buffer: Vec<u8>,
}

unsafe impl Send for VideoDecoder {}
unsafe impl Sync for VideoDecoder {}

impl VideoDecoder {
    /// Create a new video decoder with NVDEC hardware acceleration
    pub fn new(hw_accel: bool) -> Result<Self> {
        // Initialize FFmpeg
        ffmpeg_next::init().context("Failed to initialize FFmpeg")?;

        // Try NVDEC first, then software
        let (decoder, is_hardware) = if hw_accel {
            match Self::create_nvdec_decoder() {
                Ok(dec) => {
                    info!("NVDEC hardware decoder initialized (h264_cuvid)");
                    (dec, true)
                }
                Err(e) => {
                    warn!("NVDEC not available ({}), using software decoder", e);
                    (Self::create_software_decoder()?, false)
                }
            }
        } else {
            (Self::create_software_decoder()?, false)
        };

        // Pre-allocate buffers for 1080p (will resize if needed)
        let y_size = 1920 * 1080;
        let uv_size = 1920 * 1080 / 2;

        Ok(Self {
            decoder,
            frames_decoded: 0,
            nal_buffer: Vec::with_capacity(512 * 1024), // 512KB for NAL assembly
            width: 1920,
            height: 1080,
            is_hardware,
            y_buffer: vec![0u8; y_size],
            uv_buffer: vec![0u8; uv_size],
        })
    }

    /// Create hardware decoder (NVDEC on Windows/Linux, VideoToolbox on Mac)
    fn create_nvdec_decoder() -> Result<ffmpeg_next::decoder::Video> {
        // Try platform-specific hardware decoders
        #[cfg(target_os = "macos")]
        let decoder_name = "h264_videotoolbox";
        #[cfg(not(target_os = "macos"))]
        let decoder_name = "h264_cuvid";

        let codec = ffmpeg_next::decoder::find_by_name(decoder_name)
            .context(format!("{} not found", decoder_name))?;

        info!("Found hardware decoder: {}", codec.name());

        let mut decoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .decoder()
            .video()
            .context("Failed to create NVDEC decoder")?;

        unsafe {
            let ctx = decoder.as_mut_ptr();
            // Low latency flags
            (*ctx).flags |= ffmpeg_next::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
            (*ctx).flags2 |= ffmpeg_next::ffi::AV_CODEC_FLAG2_FAST as i32;
            // Single thread for NVDEC (it's already parallel on GPU)
            (*ctx).thread_count = 1;
        }

        Ok(decoder)
    }

    /// Create software decoder
    fn create_software_decoder() -> Result<ffmpeg_next::decoder::Video> {
        let codec = ffmpeg_next::decoder::find(ffmpeg_next::codec::Id::H264)
            .context("H.264 decoder not found")?;

        info!("Using software decoder: {}", codec.name());

        let mut decoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .decoder()
            .video()
            .context("Failed to create software decoder")?;

        unsafe {
            let ctx = decoder.as_mut_ptr();
            (*ctx).flags |= ffmpeg_next::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
            (*ctx).flags2 |= ffmpeg_next::ffi::AV_CODEC_FLAG2_FAST as i32;
            (*ctx).thread_count = 4;
        }

        Ok(decoder)
    }

    /// Decode a video packet (RTP payload)
    pub fn decode(&mut self, data: &[u8]) -> Result<Option<DecodedFrame>> {
        if data.is_empty() {
            return Ok(None);
        }

        // Parse RTP H.264 payload
        let nal_data = self.parse_rtp_h264(data)?;
        if nal_data.is_empty() {
            return Ok(None);
        }

        // Create packet and decode
        let mut packet = ffmpeg_next::Packet::copy(&nal_data);
        packet.set_pts(Some(self.frames_decoded as i64));
        packet.set_dts(Some(self.frames_decoded as i64));

        if self.decoder.send_packet(&packet).is_err() {
            return Ok(None);
        }

        // Try to receive decoded frame
        let mut frame = ffmpeg_next::util::frame::Video::empty();
        if self.decoder.receive_frame(&mut frame).is_err() {
            return Ok(None);
        }

        self.frames_decoded += 1;
        self.width = frame.width();
        self.height = frame.height();

        // Extract frame data with optimized copy
        let decoded = self.extract_frame_fast(&frame)?;

        if self.frames_decoded == 1 {
            info!("First {} decoded frame! {}x{}",
                  if self.is_hardware { "NVDEC" } else { "software" },
                  self.width, self.height);
        }

        Ok(Some(decoded))
    }

    /// Fast frame extraction with pre-allocated buffers
    fn extract_frame_fast(&mut self, frame: &ffmpeg_next::util::frame::Video) -> Result<DecodedFrame> {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let y_size = width * height;
        let uv_size = width * height / 2;

        // Resize buffers if needed
        if self.y_buffer.len() < y_size {
            self.y_buffer.resize(y_size, 0);
        }
        if self.uv_buffer.len() < uv_size {
            self.uv_buffer.resize(uv_size, 0);
        }

        let format = frame.format();

        match format {
            ffmpeg_next::format::Pixel::NV12 => {
                // NV12 - direct copy
                let y_plane = frame.data(0);
                let uv_plane = frame.data(1);
                let y_stride = frame.stride(0);
                let uv_stride = frame.stride(1);

                // Copy Y plane (row by row if stride != width)
                if y_stride == width {
                    self.y_buffer[..y_size].copy_from_slice(&y_plane[..y_size]);
                } else {
                    for row in 0..height {
                        let src_start = row * y_stride;
                        let dst_start = row * width;
                        self.y_buffer[dst_start..dst_start + width]
                            .copy_from_slice(&y_plane[src_start..src_start + width]);
                    }
                }

                // Copy UV plane
                let uv_height = height / 2;
                if uv_stride == width {
                    self.uv_buffer[..uv_size].copy_from_slice(&uv_plane[..uv_size]);
                } else {
                    for row in 0..uv_height {
                        let src_start = row * uv_stride;
                        let dst_start = row * width;
                        self.uv_buffer[dst_start..dst_start + width]
                            .copy_from_slice(&uv_plane[src_start..src_start + width]);
                    }
                }
            }
            ffmpeg_next::format::Pixel::YUV420P => {
                // I420 - need to interleave U and V
                let y_plane = frame.data(0);
                let u_plane = frame.data(1);
                let v_plane = frame.data(2);
                let y_stride = frame.stride(0);
                let u_stride = frame.stride(1);
                let v_stride = frame.stride(2);

                // Copy Y plane
                if y_stride == width {
                    self.y_buffer[..y_size].copy_from_slice(&y_plane[..y_size]);
                } else {
                    for row in 0..height {
                        let src_start = row * y_stride;
                        let dst_start = row * width;
                        self.y_buffer[dst_start..dst_start + width]
                            .copy_from_slice(&y_plane[src_start..src_start + width]);
                    }
                }

                // Interleave U and V
                let uv_width = width / 2;
                let uv_height = height / 2;
                let mut uv_idx = 0;
                for row in 0..uv_height {
                    let u_row_start = row * u_stride;
                    let v_row_start = row * v_stride;
                    for col in 0..uv_width {
                        self.uv_buffer[uv_idx] = u_plane[u_row_start + col];
                        self.uv_buffer[uv_idx + 1] = v_plane[v_row_start + col];
                        uv_idx += 2;
                    }
                }
            }
            _ => {
                warn!("Unsupported pixel format: {:?}, using fallback", format);
                // Just fill with black
                self.y_buffer[..y_size].fill(0);
                self.uv_buffer[..uv_size].fill(128);
            }
        }

        Ok(DecodedFrame {
            width: width as u32,
            height: height as u32,
            y_data: Arc::new(self.y_buffer[..y_size].to_vec()),
            uv_data: Arc::new(self.uv_buffer[..uv_size].to_vec()),
            pts: self.frames_decoded,
        })
    }

    /// Parse RTP H.264 payload and assemble NAL units
    fn parse_rtp_h264(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let nal_header = data[0];
        let nal_type = nal_header & 0x1F;

        match nal_type {
            1..=23 => {
                // Single NAL unit
                let mut result = Vec::with_capacity(4 + data.len());
                result.extend_from_slice(&[0, 0, 0, 1]);
                result.extend_from_slice(data);
                Ok(result)
            }
            24 => {
                // STAP-A
                let mut result = Vec::with_capacity(data.len() + 16);
                let mut offset = 1;
                while offset + 2 <= data.len() {
                    let nal_size = ((data[offset] as usize) << 8) | (data[offset + 1] as usize);
                    offset += 2;
                    if offset + nal_size <= data.len() {
                        result.extend_from_slice(&[0, 0, 0, 1]);
                        result.extend_from_slice(&data[offset..offset + nal_size]);
                        offset += nal_size;
                    } else {
                        break;
                    }
                }
                Ok(result)
            }
            28 => {
                // FU-A
                if data.len() < 2 {
                    return Ok(Vec::new());
                }
                let fu_header = data[1];
                let start_bit = (fu_header & 0x80) != 0;
                let end_bit = (fu_header & 0x40) != 0;
                let nal_unit_type = fu_header & 0x1F;

                if start_bit {
                    self.nal_buffer.clear();
                    let reconstructed_header = (nal_header & 0xE0) | nal_unit_type;
                    self.nal_buffer.extend_from_slice(&[0, 0, 0, 1, reconstructed_header]);
                    self.nal_buffer.extend_from_slice(&data[2..]);
                } else if !self.nal_buffer.is_empty() {
                    self.nal_buffer.extend_from_slice(&data[2..]);
                }

                if end_bit && !self.nal_buffer.is_empty() {
                    Ok(std::mem::take(&mut self.nal_buffer))
                } else {
                    Ok(Vec::new())
                }
            }
            _ => {
                debug!("Unsupported NAL type: {}", nal_type);
                Ok(Vec::new())
            }
        }
    }
}
