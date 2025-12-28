//! Audio Playback Module
//!
//! Handles audio decoding and low-latency playback using cpal.
//! Currently supports raw PCM audio; Opus decoding can be added with system libopus.

use anyhow::{Result, Context};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{info, warn, debug};
use parking_lot::Mutex;
use ringbuf::{HeapRb, traits::{Producer, Consumer, Split}};
use std::sync::Arc;

/// Audio player with low-latency playback
pub struct AudioPlayer {
    /// Audio output stream
    _stream: cpal::Stream,
    /// Ring buffer producer for audio samples
    producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u16,
}

impl AudioPlayer {
    /// Create a new audio player
    pub fn new() -> Result<Self> {
        // Get default audio host and output device
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device available")?;

        info!("Audio device: {}", device.name().unwrap_or_default());

        // Log all supported configs for debugging
        if let Ok(configs) = device.supported_output_configs() {
            for cfg in configs {
                info!(
                    "  Supported: {} channels, {}-{}Hz, {:?}",
                    cfg.channels(),
                    cfg.min_sample_rate().0,
                    cfg.max_sample_rate().0,
                    cfg.sample_format()
                );
            }
        }

        // Get supported config (prefer 48kHz stereo for GFN audio)
        let supported_config = device
            .supported_output_configs()
            .context("Failed to get supported audio configs")?
            .find(|c| {
                c.channels() == 2
                    && c.min_sample_rate().0 <= 48000
                    && c.max_sample_rate().0 >= 48000
            })
            .or_else(|| {
                // Fallback: find any stereo config
                device
                    .supported_output_configs()
                    .ok()?
                    .find(|c| c.channels() == 2)
            })
            .or_else(|| {
                // Last resort: any config
                device
                    .supported_output_configs()
                    .ok()?
                    .next()
            })
            .context("No suitable audio config found")?;

        info!(
            "Selected config: {} channels, {}-{}Hz",
            supported_config.channels(),
            supported_config.min_sample_rate().0,
            supported_config.max_sample_rate().0
        );

        // Determine sample rate - must be within the supported range
        let min_rate = supported_config.min_sample_rate().0;
        let max_rate = supported_config.max_sample_rate().0;
        let sample_rate = if min_rate <= 48000 && max_rate >= 48000 {
            48000
        } else if min_rate <= 44100 && max_rate >= 44100 {
            44100
        } else {
            // Use whatever is supported
            min_rate.max(max_rate.min(48000))
        };

        info!("Using sample rate: {}Hz", sample_rate);

        // Ensure sample rate is in valid range before calling with_sample_rate
        let clamped_rate = sample_rate.max(min_rate).min(max_rate);
        if clamped_rate != sample_rate {
            warn!("Sample rate {} clamped to {}", sample_rate, clamped_rate);
        }

        let channels = supported_config.channels();
        let config = supported_config
            .with_sample_rate(cpal::SampleRate(clamped_rate))
            .config();

        info!(
            "Audio config: {}Hz, {} channels",
            sample_rate, channels
        );

        // Create ring buffer for audio samples (20ms buffer for low latency)
        // Lower buffer = lower latency but higher risk of underruns
        let buffer_size = (sample_rate as usize * channels as usize) / 50; // 20ms
        let ring = HeapRb::<f32>::new(buffer_size);
        let (producer, mut consumer) = ring.split();

        // Wrap producer in Arc<Mutex> for thread-safe access
        let producer = Arc::new(Mutex::new(producer));

        // Create audio output stream
        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Fill output buffer from ring buffer
                for sample in data.iter_mut() {
                    *sample = consumer.try_pop().unwrap_or(0.0);
                }
            },
            |err| {
                warn!("Audio stream error: {}", err);
            },
            None,
        ).context("Failed to create audio stream")?;

        // Start playback
        stream.play().context("Failed to start audio playback")?;

        Ok(Self {
            _stream: stream,
            producer,
            sample_rate,
            channels,
        })
    }

    /// Play raw PCM audio data (f32 samples, interleaved stereo)
    pub fn play(&self, pcm_data: &[u8]) -> Result<()> {
        if pcm_data.is_empty() {
            return Ok(());
        }

        // Convert bytes to f32 samples
        // Assuming the data is already f32 PCM (4 bytes per sample)
        let samples: Vec<f32> = pcm_data
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(bytes)
            })
            .collect();

        // Push samples to ring buffer
        let mut producer = self.producer.lock();

        for &sample in &samples {
            // Drop samples if buffer is full (prevents latency buildup)
            let _ = producer.try_push(sample);
        }

        debug!("Played {} audio samples", samples.len());
        Ok(())
    }

    /// Play raw PCM audio data (i16 samples, interleaved stereo)
    pub fn play_i16(&self, pcm_data: &[u8]) -> Result<()> {
        if pcm_data.is_empty() {
            return Ok(());
        }

        // Convert i16 samples to f32
        let samples: Vec<f32> = pcm_data
            .chunks_exact(2)
            .map(|chunk| {
                let bytes: [u8; 2] = chunk.try_into().unwrap();
                let sample = i16::from_le_bytes(bytes);
                sample as f32 / 32768.0
            })
            .collect();

        // Push samples to ring buffer
        let mut producer = self.producer.lock();

        for &sample in &samples {
            let _ = producer.try_push(sample);
        }

        debug!("Played {} audio samples (i16)", samples.len());
        Ok(())
    }

    /// Get audio statistics
    pub fn stats(&self) -> AudioStats {
        AudioStats {
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

/// Audio statistics
pub struct AudioStats {
    pub sample_rate: u32,
    pub channels: u16,
}
