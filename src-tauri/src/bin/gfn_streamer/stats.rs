//! Streaming Statistics Module
//!
//! Tracks and reports streaming performance metrics with low-latency focus.

use std::collections::VecDeque;
use std::time::Instant;

/// Rolling window size for averaging
const WINDOW_SIZE: usize = 60;

/// Streaming statistics with proper tracking
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Frames decoded
    pub frames_decoded: u64,
    /// Frames displayed (rendered)
    pub frames_rendered: u64,
    /// Frames dropped (arrived too late)
    pub frames_dropped: u64,
    /// Bytes received (video)
    pub bytes_received: u64,
    /// Audio packets received
    pub audio_packets: u64,
    /// Video packets received
    pub video_packets: u64,
    /// Render FPS (frames actually displayed)
    pub render_fps: f64,
    /// Decode FPS (frames from decoder)
    pub decode_fps: f64,
    /// Network latency in milliseconds (RTT/2)
    pub latency_ms: f64,
    /// Packet loss percentage (0.0-1.0)
    pub packet_loss: f64,
    /// Current bitrate in kbps
    pub bitrate_kbps: u64,
    /// Video resolution
    pub resolution: String,
    /// Video codec
    pub codec: String,
    /// Jitter in milliseconds
    pub jitter_ms: f64,
    /// Input packets sent
    pub input_packets_sent: u64,
    /// Stream start time
    pub stream_start: Option<Instant>,
    /// Last RTP sequence number (for packet loss detection)
    pub last_video_seq: u16,
    /// Missing packets count
    pub missing_packets: u64,
    /// Target FPS (from server)
    pub target_fps: u32,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            frames_decoded: 0,
            frames_rendered: 0,
            frames_dropped: 0,
            bytes_received: 0,
            audio_packets: 0,
            video_packets: 0,
            render_fps: 0.0,
            decode_fps: 0.0,
            latency_ms: 0.0,
            packet_loss: 0.0,
            bitrate_kbps: 0,
            resolution: String::new(),
            codec: "H.264".to_string(),
            jitter_ms: 0.0,
            input_packets_sent: 0,
            stream_start: None,
            last_video_seq: 0,
            missing_packets: 0,
            target_fps: 60,
        }
    }
}

impl StreamStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Update packet loss based on sequence numbers
    pub fn update_video_seq(&mut self, seq: u16) {
        if self.video_packets > 0 {
            let expected = self.last_video_seq.wrapping_add(1);
            if seq != expected && seq != self.last_video_seq {
                // Handle wraparound
                let gap = if seq > expected {
                    (seq - expected) as u64
                } else {
                    // Wrapped around
                    (65536 - expected as u64) + seq as u64
                };
                if gap < 1000 {
                    // Sanity check
                    self.missing_packets += gap;
                }
            }
        }
        self.last_video_seq = seq;
        self.video_packets += 1;

        // Calculate packet loss as percentage
        if self.video_packets > 0 {
            self.packet_loss = self.missing_packets as f64 / (self.video_packets + self.missing_packets) as f64;
        }
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> f64 {
        self.stream_start
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Format stats for overlay display (compact)
    pub fn format_overlay(&self) -> Vec<String> {
        vec![
            format!("FPS: {:.0}/{} | Decode: {:.0}", self.render_fps, self.target_fps, self.decode_fps),
            format!("Latency: {:.0}ms | Jitter: {:.1}ms", self.latency_ms, self.jitter_ms),
            format!("Bitrate: {} Mbps | Loss: {:.2}%", self.bitrate_kbps / 1000, self.packet_loss * 100.0),
            format!("Video: {} | Audio: {} | Input: {}",
                self.video_packets, self.audio_packets, self.input_packets_sent),
            format!("Dropped: {} | {}", self.frames_dropped, self.resolution),
        ]
    }

    /// Format single-line stats
    pub fn format(&self) -> String {
        format!(
            "FPS: {:.0}/{} | {:.0}ms | {:.1}% loss | {} Mbps",
            self.render_fps,
            self.target_fps,
            self.latency_ms,
            self.packet_loss * 100.0,
            self.bitrate_kbps / 1000
        )
    }
}

/// Real-time stats calculator with rolling windows
pub struct StatsCalculator {
    /// Frame timestamps for FPS calculation
    frame_times: VecDeque<Instant>,
    /// Decode timestamps
    decode_times: VecDeque<Instant>,
    /// Bytes received in last second for bitrate
    bytes_window: VecDeque<(Instant, u64)>,
    /// Latency samples
    latency_samples: VecDeque<f64>,
    /// Jitter samples
    jitter_samples: VecDeque<f64>,
    /// Last update time
    last_update: Instant,
}

impl StatsCalculator {
    pub fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(WINDOW_SIZE),
            decode_times: VecDeque::with_capacity(WINDOW_SIZE),
            bytes_window: VecDeque::with_capacity(1000),
            latency_samples: VecDeque::with_capacity(WINDOW_SIZE),
            jitter_samples: VecDeque::with_capacity(WINDOW_SIZE),
            last_update: Instant::now(),
        }
    }

    /// Record a rendered frame
    pub fn record_frame(&mut self) {
        let now = Instant::now();
        self.frame_times.push_back(now);
        if self.frame_times.len() > WINDOW_SIZE {
            self.frame_times.pop_front();
        }
    }

    /// Record a decoded frame
    pub fn record_decode(&mut self) {
        let now = Instant::now();
        self.decode_times.push_back(now);
        if self.decode_times.len() > WINDOW_SIZE {
            self.decode_times.pop_front();
        }
    }

    /// Record bytes received
    pub fn record_bytes(&mut self, bytes: u64) {
        let now = Instant::now();
        self.bytes_window.push_back((now, bytes));
        // Remove entries older than 1 second
        while let Some((t, _)) = self.bytes_window.front() {
            if now.duration_since(*t).as_secs_f64() > 1.0 {
                self.bytes_window.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record latency sample (in ms)
    pub fn record_latency(&mut self, latency_ms: f64) {
        self.latency_samples.push_back(latency_ms);
        if self.latency_samples.len() > WINDOW_SIZE {
            self.latency_samples.pop_front();
        }
    }

    /// Record jitter sample
    pub fn record_jitter(&mut self, jitter_ms: f64) {
        self.jitter_samples.push_back(jitter_ms);
        if self.jitter_samples.len() > WINDOW_SIZE {
            self.jitter_samples.pop_front();
        }
    }

    /// Calculate render FPS from recent frames
    pub fn calculate_render_fps(&self) -> f64 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }
        let first = self.frame_times.front().unwrap();
        let last = self.frame_times.back().unwrap();
        let duration = last.duration_since(*first).as_secs_f64();
        if duration > 0.0 {
            (self.frame_times.len() - 1) as f64 / duration
        } else {
            0.0
        }
    }

    /// Calculate decode FPS
    pub fn calculate_decode_fps(&self) -> f64 {
        if self.decode_times.len() < 2 {
            return 0.0;
        }
        let first = self.decode_times.front().unwrap();
        let last = self.decode_times.back().unwrap();
        let duration = last.duration_since(*first).as_secs_f64();
        if duration > 0.0 {
            (self.decode_times.len() - 1) as f64 / duration
        } else {
            0.0
        }
    }

    /// Calculate bitrate in kbps
    pub fn calculate_bitrate_kbps(&self) -> u64 {
        let total_bytes: u64 = self.bytes_window.iter().map(|(_, b)| *b).sum();
        // bytes per second * 8 / 1000 = kbps
        (total_bytes * 8) / 1000
    }

    /// Calculate average latency
    pub fn calculate_avg_latency(&self) -> f64 {
        if self.latency_samples.is_empty() {
            return 0.0;
        }
        self.latency_samples.iter().sum::<f64>() / self.latency_samples.len() as f64
    }

    /// Calculate average jitter
    pub fn calculate_avg_jitter(&self) -> f64 {
        if self.jitter_samples.is_empty() {
            return 0.0;
        }
        self.jitter_samples.iter().sum::<f64>() / self.jitter_samples.len() as f64
    }

    /// Update stats structure with calculated values
    pub fn update_stats(&mut self, stats: &mut StreamStats) {
        stats.render_fps = self.calculate_render_fps();
        stats.decode_fps = self.calculate_decode_fps();
        stats.bitrate_kbps = self.calculate_bitrate_kbps();
        stats.latency_ms = self.calculate_avg_latency();
        stats.jitter_ms = self.calculate_avg_jitter();
    }
}

impl Default for StatsCalculator {
    fn default() -> Self {
        Self::new()
    }
}
