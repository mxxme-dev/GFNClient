//! GFN Native Streamer - High-Performance Hardware-Accelerated Streaming Client
//!
//! This is a standalone binary that handles GFN streaming with:
//! - SDL2-based input handling (matching official NVIDIA client)
//! - Hardware-accelerated video decoding (DXVA2/D3D11VA on Windows, VideoToolbox on macOS, VAAPI on Linux)
//! - GPU-accelerated rendering via wgpu (Vulkan/Metal/DX12)
//! - Low-latency input pipeline with direct WebRTC sending
//! - Opus audio decoding with minimal jitter buffering
//!
//! Usage:
//!   gfn-streamer --server <ip> --session-id <id> [--width 1920] [--height 1080] [--fullscreen]

mod decoder;
mod renderer;
mod input;
mod audio;
mod signaling;
mod stats;
mod overlay;
// mod vulkan_decoder; // TODO: Re-enable when Vulkan Video implementation is complete

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use std::io::Write;

use anyhow::Result;
use clap::Parser;
use log::{info, warn, error, debug};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::mouse::MouseButton;

use decoder::VideoDecoder;
use renderer::Renderer;
use input::{InputEncoder, InputEvent};
use audio::AudioPlayer;
use signaling::SignalingClient;
use stats::{StreamStats, StatsCalculator};

/// GFN Native Streamer - High-Performance Streaming Client
#[derive(Parser, Debug, Clone)]
#[command(name = "gfn-streamer")]
#[command(author, version, about = "High-performance native GFN streaming client", long_about = None)]
struct Args {
    /// GFN streaming server address (IP or hostname)
    #[arg(short, long)]
    server: String,

    /// Session ID from GFN session setup
    #[arg(short = 'i', long)]
    session_id: String,

    /// Authentication token (optional, for reconnection)
    #[arg(short, long)]
    token: Option<String>,

    /// Window width (ignored in fullscreen mode)
    #[arg(long, default_value = "1920")]
    width: u32,

    /// Window height (ignored in fullscreen mode)
    #[arg(long, default_value = "1080")]
    height: u32,

    /// Start in fullscreen mode
    #[arg(short, long)]
    fullscreen: bool,

    /// Enable debug logging
    #[arg(short, long)]
    debug: bool,

    /// Deprecated - hardware decoding is now mandatory
    #[arg(long, hide = true)]
    no_hwaccel: bool,

    /// Target framerate (for stats display)
    #[arg(long, default_value = "60")]
    fps: u32,
}

/// Shared state between the render thread and network tasks
pub struct SharedState {
    /// Current video frame (YUV or RGB data ready for upload)
    pub pending_frame: Option<decoder::DecodedFrame>,
    /// Stream statistics
    pub stats: StreamStats,
    /// Connection status
    pub connected: bool,
    /// Input channel ready
    pub input_ready: bool,
    /// Status message for overlay
    pub status: String,
    /// Should exit
    pub should_exit: bool,
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            pending_frame: None,
            stats: StreamStats::default(),
            connected: false,
            input_ready: false,
            status: "Initializing...".to_string(),
            should_exit: false,
        }
    }
}

/// Direct input sender - sends input directly to WebRTC without channel hop
pub struct DirectInputSender {
    /// WebRTC data channel reference
    input_channel: Arc<Mutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
    /// Input protocol version (>2 needs wrapper)
    protocol_version: Arc<Mutex<u16>>,
    /// Stream start time for relative timestamps
    stream_start_time: Arc<Mutex<Option<Instant>>>,
    /// Input encoder (wrapped in Mutex for thread safety)
    encoder: Mutex<InputEncoder>,
    /// Send count for logging
    send_count: AtomicU64,
    /// Tokio runtime handle for async sending
    runtime: tokio::runtime::Handle,
    /// Shared state for updating stats
    state: Arc<Mutex<SharedState>>,
}

impl DirectInputSender {
    fn new(
        input_channel: Arc<Mutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
        protocol_version: Arc<Mutex<u16>>,
        stream_start_time: Arc<Mutex<Option<Instant>>>,
        runtime: tokio::runtime::Handle,
        state: Arc<Mutex<SharedState>>,
    ) -> Self {
        Self {
            input_channel,
            protocol_version,
            stream_start_time,
            encoder: Mutex::new(InputEncoder::new()),
            send_count: AtomicU64::new(0),
            runtime,
            state,
        }
    }

    /// Send input event directly to WebRTC - called from SDL event loop
    fn send(&self, event: InputEvent) {
        // Encode the input event
        let data = self.encoder.lock().encode(&event);

        // Get the input channel
        let channel = {
            let storage = self.input_channel.lock();
            storage.clone()
        };

        if let Some(dc) = channel {
            // Check if channel is open
            if dc.ready_state() == webrtc::data_channel::data_channel_state::RTCDataChannelState::Open {
                // Get protocol version
                let version = {
                    let pv = self.protocol_version.lock();
                    *pv
                };

                // Build the final packet - wrap if protocol version > 2
                let final_packet = if version > 2 {
                    // v3+ protocol requires 10-byte header wrapper
                    let timestamp_us = {
                        let st = self.stream_start_time.lock();
                        if let Some(start) = *st {
                            start.elapsed().as_micros() as u64
                        } else {
                            0
                        }
                    };

                    let mut wrapped = Vec::with_capacity(10 + data.len());
                    wrapped.push(0x23);
                    wrapped.extend_from_slice(&timestamp_us.to_be_bytes());
                    wrapped.push(0x22);
                    wrapped.extend_from_slice(&data);
                    wrapped
                } else {
                    data.clone()
                };

                let packet = bytes::Bytes::copy_from_slice(&final_packet);
                let dc_clone = dc.clone();

                // Use spawn_blocking to send without blocking SDL event loop
                self.runtime.spawn(async move {
                    let _ = dc_clone.send(&packet).await;
                });

                let count = self.send_count.fetch_add(1, Ordering::Relaxed);

                // Update stats
                {
                    let mut s = self.state.lock();
                    s.stats.input_packets_sent = count + 1;
                }

                if count < 10 || count % 500 == 0 {
                    debug!("INPUT SENT #{}: {} bytes", count + 1, final_packet.len());
                }
            }
        }
    }
}

/// Dual writer that writes to both stdout and a file
struct DualWriter {
    file: std::sync::Mutex<std::fs::File>,
}

impl DualWriter {
    fn new(file: std::fs::File) -> Self {
        Self {
            file: std::sync::Mutex::new(file),
        }
    }
}

impl Write for DualWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let _ = std::io::stderr().write_all(buf);
        let _ = std::io::stderr().flush();
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(buf);
            let _ = file.sync_all();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let _ = std::io::stderr().flush();
        if let Ok(mut file) = self.file.lock() {
            let _ = file.sync_all();
        }
        Ok(())
    }
}

/// Convert SDL scancode to Windows virtual key and scan code
fn sdl_scancode_to_vk(scancode: Scancode) -> (u16, u16) {
    use Scancode::*;
    match scancode {
        A => (0x41, 0x1E),
        B => (0x42, 0x30),
        C => (0x43, 0x2E),
        D => (0x44, 0x20),
        E => (0x45, 0x12),
        F => (0x46, 0x21),
        G => (0x47, 0x22),
        H => (0x48, 0x23),
        I => (0x49, 0x17),
        J => (0x4A, 0x24),
        K => (0x4B, 0x25),
        L => (0x4C, 0x26),
        M => (0x4D, 0x32),
        N => (0x4E, 0x31),
        O => (0x4F, 0x18),
        P => (0x50, 0x19),
        Q => (0x51, 0x10),
        R => (0x52, 0x13),
        S => (0x53, 0x1F),
        T => (0x54, 0x14),
        U => (0x55, 0x16),
        V => (0x56, 0x2F),
        W => (0x57, 0x11),
        X => (0x58, 0x2D),
        Y => (0x59, 0x15),
        Z => (0x5A, 0x2C),
        Num1 => (0x31, 0x02),
        Num2 => (0x32, 0x03),
        Num3 => (0x33, 0x04),
        Num4 => (0x34, 0x05),
        Num5 => (0x35, 0x06),
        Num6 => (0x36, 0x07),
        Num7 => (0x37, 0x08),
        Num8 => (0x38, 0x09),
        Num9 => (0x39, 0x0A),
        Num0 => (0x30, 0x0B),
        Return => (0x0D, 0x1C),
        Escape => (0x1B, 0x01),
        Backspace => (0x08, 0x0E),
        Tab => (0x09, 0x0F),
        Space => (0x20, 0x39),
        LShift => (0x10, 0x2A),
        RShift => (0x10, 0x36),
        LCtrl => (0x11, 0x1D),
        RCtrl => (0x11, 0x1D),
        LAlt => (0x12, 0x38),
        RAlt => (0x12, 0x38),
        Left => (0x25, 0x4B),
        Right => (0x27, 0x4D),
        Up => (0x26, 0x48),
        Down => (0x28, 0x50),
        F1 => (0x70, 0x3B),
        F2 => (0x71, 0x3C),
        F3 => (0x72, 0x3D),
        F4 => (0x73, 0x3E),
        F5 => (0x74, 0x3F),
        F6 => (0x75, 0x40),
        F7 => (0x76, 0x41),
        F8 => (0x77, 0x42),
        F9 => (0x78, 0x43),
        F10 => (0x79, 0x44),
        F11 => (0x7A, 0x57),
        F12 => (0x7B, 0x58),
        _ => (0, 0),
    }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging to both console and file
    let log_level = if args.debug { "debug" } else { "info" };
    let log_file = std::fs::File::create("C:\\log.log").expect("Failed to create log file");

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .target(env_logger::Target::Pipe(Box::new(DualWriter::new(log_file))))
        .init();

    // Set panic hook
    std::panic::set_hook(Box::new(|panic_info| {
        let msg = format!("PANIC: {}\n", panic_info);
        eprintln!("{}", msg);
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open("C:\\log.log") {
            let _ = f.write_all(msg.as_bytes());
            let _ = f.sync_all();
        }
    }));

    info!("GFN Native Streamer v{} (SDL2 input)", env!("CARGO_PKG_VERSION"));
    info!("Server: {}", args.server);
    info!("Session: {}", args.session_id);
    info!("Resolution: {}x{}", args.width, args.height);

    // Initialize SDL2
    let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!("SDL2 init failed: {}", e))?;
    let video_subsystem = sdl_context.video().map_err(|e| anyhow::anyhow!("SDL2 video init failed: {}", e))?;

    info!("SDL2 initialized");

    // Create SDL2 window
    let mut window_builder = video_subsystem.window("GFN Streamer", args.width, args.height);
    window_builder.position_centered().resizable().allow_highdpi();

    if args.fullscreen {
        window_builder.fullscreen_desktop();
    }

    let mut window = window_builder.build().map_err(|e| anyhow::anyhow!("Window creation failed: {}", e))?;

    info!("Window created: {}x{}", window.size().0, window.size().1);

    // Get raw window handle for wgpu
    use raw_window_handle::{HasWindowHandle, HasDisplayHandle};
    let raw_window = window.window_handle().unwrap().as_raw();
    let raw_display = window.display_handle().unwrap().as_raw();

    // Create shared state
    let state = Arc::new(Mutex::new(SharedState::default()));

    // Create tokio runtime for async networking
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Create GPU renderer using SDL2 window
    info!("Initializing GPU renderer...");
    let renderer = runtime.block_on(async {
        Renderer::new_from_raw(raw_window, raw_display, args.width, args.height).await
    })?;
    let renderer = Arc::new(Mutex::new(renderer));
    info!("Renderer initialized");

    // Create channels for streaming events
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();

    // Spawn streaming task and get access to input channel refs
    let streaming_args = args.clone();
    let streaming_state = state.clone();

    // We need to share the input channel reference with the SDL event loop
    let input_channel_ref: Arc<Mutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>> = Arc::new(Mutex::new(None));
    let protocol_version_ref: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
    let stream_start_ref: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

    let input_channel_for_stream = input_channel_ref.clone();
    let protocol_version_for_stream = protocol_version_ref.clone();
    let stream_start_for_stream = stream_start_ref.clone();

    runtime.spawn(async move {
        info!("Spawning streaming task...");
        match run_streaming(
            streaming_args,
            streaming_state,
            input_rx,
            audio_tx,
            input_channel_for_stream,
            protocol_version_for_stream,
            stream_start_for_stream,
        ).await {
            Ok(()) => info!("Streaming task completed normally"),
            Err(e) => {
                error!("Streaming error: {}", e);
                error!("Cause: {:?}", e.source());
            }
        }
    });

    // Spawn audio processing
    std::thread::spawn(move || {
        let audio_player = match AudioPlayer::new() {
            Ok(player) => {
                info!("Audio player initialized");
                player
            }
            Err(e) => {
                warn!("Failed to initialize audio player: {}", e);
                return;
            }
        };

        while let Some(data) = audio_rx.blocking_recv() {
            if let Err(e) = audio_player.play(&data) {
                debug!("Audio play error: {}", e);
            }
        }
    });

    // Create direct input sender (bypasses mpsc channel for minimum latency)
    let direct_sender = DirectInputSender::new(
        input_channel_ref.clone(),
        protocol_version_ref.clone(),
        stream_start_ref.clone(),
        runtime.handle().clone(),
        state.clone(),
    );

    // SDL2 event loop with polling (matching NVIDIA SDLEventProcessor)
    let mut event_pump = sdl_context.event_pump().map_err(|e| anyhow::anyhow!("Event pump failed: {}", e))?;

    // Disable mouse acceleration for raw input (matching NVIDIA: accel=0)
    sdl_context.mouse().set_relative_mouse_mode(false);

    let mut mouse_captured = false;
    let mut last_stats_update = Instant::now();
    let mut stats_calc = StatsCalculator::new();
    let start_time = Instant::now();
    let running = Arc::new(AtomicBool::new(true));
    let mut last_title_update = Instant::now();

    // On Windows, set timer resolution to 1ms for better sleep precision
    #[cfg(target_os = "windows")]
    unsafe {
        extern "system" {
            fn timeBeginPeriod(uPeriod: u32) -> u32;
        }
        timeBeginPeriod(1);
    }

    // Set target FPS in stats
    {
        let mut s = state.lock();
        s.stats.target_fps = args.fps;
        s.stats.stream_start = Some(Instant::now());
    }

    info!("Starting SDL2 event loop (polling mode, target {}fps)", args.fps);

    'main_loop: loop {
        // Poll SDL events (matching NVIDIA SDLEventProcessor tight polling)
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    info!("Quit event received");
                    break 'main_loop;
                }

                Event::KeyDown { scancode: Some(scancode), repeat: false, .. } => {
                    // ESC releases mouse capture
                    if scancode == Scancode::Escape && mouse_captured {
                        sdl_context.mouse().set_relative_mouse_mode(false);
                        sdl_context.mouse().show_cursor(true);
                        mouse_captured = false;
                        info!("Mouse released (ESC)");
                        continue;
                    }

                    // F3 toggles stats overlay
                    if scancode == Scancode::F3 {
                        let mut r = renderer.lock();
                        r.toggle_overlay();
                        info!("Stats overlay: {}", if r.is_overlay_visible() { "ON" } else { "OFF" });
                        continue;
                    }

                    // F11 toggles fullscreen
                    if scancode == Scancode::F11 {
                        // Toggle fullscreen (simplified)
                        continue;
                    }


                    if mouse_captured {
                        let (vk, scan) = sdl_scancode_to_vk(scancode);
                        if vk != 0 {
                            let event = InputEvent::KeyDown {
                                keycode: vk,
                                scancode: scan,
                                modifiers: 0,
                                timestamp_us: start_time.elapsed().as_micros() as u64,
                            };
                            direct_sender.send(event);
                        }
                    }
                }

                Event::KeyUp { scancode: Some(scancode), repeat: false, .. } => {
                    if mouse_captured {
                        let (vk, scan) = sdl_scancode_to_vk(scancode);
                        if vk != 0 {
                            let event = InputEvent::KeyUp {
                                keycode: vk,
                                scancode: scan,
                                modifiers: 0,
                                timestamp_us: start_time.elapsed().as_micros() as u64,
                            };
                            direct_sender.send(event);
                        }
                    }
                }

                Event::MouseButtonDown { mouse_btn, .. } => {
                    if !mouse_captured {
                        // Click to capture mouse
                        sdl_context.mouse().set_relative_mouse_mode(true);
                        mouse_captured = true;
                        info!("Mouse captured (click)");
                        continue;
                    }

                    let btn = match mouse_btn {
                        MouseButton::Left => 1,
                        MouseButton::Right => 2,
                        MouseButton::Middle => 3,
                        MouseButton::X1 => 4,
                        MouseButton::X2 => 5,
                        _ => 0,
                    };

                    if btn != 0 {
                        let event = InputEvent::MouseButtonDown {
                            button: btn,
                            timestamp_us: start_time.elapsed().as_micros() as u64,
                        };
                        direct_sender.send(event);
                    }
                }

                Event::MouseButtonUp { mouse_btn, .. } => {
                    if mouse_captured {
                        let btn = match mouse_btn {
                            MouseButton::Left => 1,
                            MouseButton::Right => 2,
                            MouseButton::Middle => 3,
                            MouseButton::X1 => 4,
                            MouseButton::X2 => 5,
                            _ => 0,
                        };

                        if btn != 0 {
                            let event = InputEvent::MouseButtonUp {
                                button: btn,
                                timestamp_us: start_time.elapsed().as_micros() as u64,
                            };
                            direct_sender.send(event);
                        }
                    }
                }

                Event::MouseMotion { xrel, yrel, .. } => {
                    if mouse_captured && (xrel != 0 || yrel != 0) {
                        let event = InputEvent::MouseMove {
                            dx: xrel as i16,
                            dy: yrel as i16,
                            timestamp_us: start_time.elapsed().as_micros() as u64,
                        };
                        direct_sender.send(event);
                    }
                }

                Event::MouseWheel { y, .. } => {
                    if mouse_captured && y != 0 {
                        let event = InputEvent::MouseWheel {
                            delta_x: 0,
                            delta_y: (y * 120) as i16,
                            timestamp_us: start_time.elapsed().as_micros() as u64,
                        };
                        direct_sender.send(event);
                    }
                }

                Event::Window { win_event, .. } => {
                    use sdl2::event::WindowEvent;
                    match win_event {
                        WindowEvent::FocusLost => {
                            if mouse_captured {
                                sdl_context.mouse().set_relative_mouse_mode(false);
                                sdl_context.mouse().show_cursor(true);
                                mouse_captured = false;
                                info!("Mouse released (focus lost)");
                            }
                        }
                        WindowEvent::Resized(w, h) => {
                            if w > 0 && h > 0 {
                                let mut r = renderer.lock();
                                let _ = r.resize(w as u32, h as u32);
                            }
                        }
                        _ => {}
                    }
                }

                _ => {}
            }
        }

        // Only render when we have a new frame (reduces GPU 3D usage)
        let frame_start = Instant::now();
        let frame = {
            let mut s = state.lock();
            s.pending_frame.take()
        };

        if let Some(frame) = frame {
            // Upload and render in one go
            let (w, h) = window.size();
            if w > 0 && h > 0 {
                let mut r = renderer.lock();
                if let Err(e) = r.upload_frame(&frame) {
                    warn!("Failed to upload frame: {}", e);
                } else {
                    let stats = {
                        let s = state.lock();
                        s.stats.clone()
                    };
                    if let Err(e) = r.render(w, h, &stats) {
                        debug!("Render error: {}", e);
                    }
                    stats_calc.record_frame();
                }
            }
        }

        // Update stats every 100ms (10 times/sec for smooth display)
        if last_stats_update.elapsed().as_millis() >= 100 {
            let mut s = state.lock();
            stats_calc.update_stats(&mut s.stats);
            last_stats_update = Instant::now();
        }

        // Update window title with stats (every 250ms to avoid flicker)
        if last_title_update.elapsed().as_millis() >= 250 {
            let stats = {
                let s = state.lock();
                s.stats.clone()
            };
            let title = format!(
                "GFN Streamer | FPS: {:.0}/{} | {:.0}ms | {:.1}% loss | {} Mbps | V:{} A:{} I:{}",
                stats.render_fps,
                stats.target_fps,
                stats.latency_ms,
                stats.packet_loss * 100.0,
                stats.bitrate_kbps / 1000,
                stats.video_packets,
                stats.audio_packets,
                stats.input_packets_sent
            );
            let _ = window.set_title(&title);
            last_title_update = Instant::now();
        }

        // Check exit flag
        {
            let s = state.lock();
            if s.should_exit {
                break 'main_loop;
            }
        }

        // Yield briefly to prevent busy-waiting
        std::thread::yield_now();
    }

    info!("Shutting down...");
    running.store(false, Ordering::SeqCst);

    // Restore Windows timer resolution
    #[cfg(target_os = "windows")]
    unsafe {
        extern "system" {
            fn timeEndPeriod(uPeriod: u32) -> u32;
        }
        timeEndPeriod(1);
    }

    drop(runtime);

    Ok(())
}

/// Run the streaming network tasks
async fn run_streaming(
    args: Args,
    state: Arc<Mutex<SharedState>>,
    mut input_rx: mpsc::UnboundedReceiver<InputEvent>,
    audio_tx: mpsc::UnboundedSender<Vec<u8>>,
    input_channel: Arc<Mutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
    protocol_version: Arc<Mutex<u16>>,
    stream_start_time: Arc<Mutex<Option<Instant>>>,
) -> Result<()> {
    info!("Starting streaming connection to {}", args.server);

    // Initialize hardware video decoder (required for low-latency streaming)
    info!("Initializing video decoder...");
    let mut decoder = VideoDecoder::new(true)?;
    info!("Video decoder initialized successfully");

    // Create signaling client with shared input channel refs
    info!("Creating signaling client...");
    let mut signaling = SignalingClient::new_with_refs(
        args.server.clone(),
        args.session_id.clone(),
        args.width,
        args.height,
        args.fps,
        input_channel,
        protocol_version,
        stream_start_time,
    );
    info!("Signaling client created");

    {
        let mut s = state.lock();
        s.status = "Connecting to server...".to_string();
    }

    info!("Connecting to signaling server...");
    let mut event_rx = signaling.connect().await?;
    info!("Signaling connection established");

    {
        let mut s = state.lock();
        s.status = "Waiting for stream...".to_string();
        s.connected = true;
    }

    info!("Connected to signaling server");

    // Main event loop (no longer needs to handle input - it's direct now)
    loop {
        tokio::select! {
            event = event_rx.recv() => {
                match event {
                    Some(signaling::SignalingEvent::VideoPacket(data)) => {
                        // Track video packet stats
                        {
                            let mut s = state.lock();
                            s.stats.video_packets += 1;
                            s.stats.bytes_received += data.len() as u64;
                            // Update resolution from frame if we haven't yet
                            if s.stats.resolution.is_empty() {
                                s.stats.resolution = format!("{}x{}", args.width, args.height);
                            }
                        }

                        match decoder.decode(&data) {
                            Ok(Some(frame)) => {
                                let mut s = state.lock();
                                let frame_num = s.stats.frames_decoded + 1;
                                s.pending_frame = Some(frame);
                                s.stats.frames_decoded = frame_num;

                                if frame_num == 1 {
                                    info!("First decoded frame!");
                                    s.status = "Streaming".to_string();
                                }
                            }
                            Ok(None) => {
                                // Partial frame, still counted in video_packets above
                            }
                            Err(e) => {
                                debug!("Decode error: {}", e);
                            }
                        }
                    }
                    Some(signaling::SignalingEvent::AudioPacket(data)) => {
                        let _ = audio_tx.send(data.to_vec());
                        let mut s = state.lock();
                        s.stats.audio_packets += 1;
                    }
                    Some(signaling::SignalingEvent::InputReady) => {
                        let mut s = state.lock();
                        s.input_ready = true;
                        s.status = "Streaming".to_string();
                        info!("Input channel ready - direct sending enabled");
                    }
                    Some(signaling::SignalingEvent::Connected) => {
                        info!("WebRTC connection established");
                        let mut s = state.lock();
                        s.status = "Connected, waiting for media...".to_string();
                    }
                    Some(signaling::SignalingEvent::Disconnected(reason)) => {
                        warn!("Disconnected: {}", reason);
                        let mut s = state.lock();
                        s.connected = false;
                        s.status = format!("Disconnected: {}", reason);
                        break;
                    }
                    Some(signaling::SignalingEvent::Error(e)) => {
                        error!("Signaling error: {}", e);
                    }
                    None => {
                        info!("Signaling channel closed");
                        break;
                    }
                }
            }

            // Legacy input handling (fallback, should not be used with direct sender)
            Some(_input) = input_rx.recv() => {
                // This path is no longer used - DirectInputSender handles input
            }

            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                let should_exit = {
                    let s = state.lock();
                    s.should_exit
                };
                if should_exit {
                    break;
                }
            }
        }
    }

    info!("Streaming ended");
    Ok(())
}
