//! Signaling Client Module
//!
//! Handles WebSocket signaling for GFN streaming sessions.
//! Implements the NVST (NVIDIA Streaming) protocol with full WebRTC support.
//!
//! Protocol based on official GFN browser client:
//! - URL: wss://{server}/nvst/sign_in?peer_id=peer-{random}&version=2
//! - Auth: WebSocket subprotocol x-nv-sessionid.{session_id}
//! - Messages: JSON with ackid, peer_info, peer_msg fields

use anyhow::{Context, Result};
use bytes::Bytes;
#[allow(unused_imports)]
use futures_util::{SinkExt, StreamExt};
use log::{debug, error, info, warn};
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{client::IntoClientRequest, http::header, Message},
};
use webrtc::{
    api::{
        interceptor_registry::register_default_interceptors, media_engine::MediaEngine, APIBuilder,
    },
    ice_transport::{ice_candidate::RTCIceCandidateInit, ice_server::RTCIceServer},
    interceptor::registry::Registry,
    peer_connection::{
        configuration::RTCConfiguration, peer_connection_state::RTCPeerConnectionState,
        sdp::session_description::RTCSessionDescription, RTCPeerConnection,
    },
    rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTPCodecType},
    track::track_remote::TrackRemote,
};

/// Events from the signaling connection
#[derive(Debug)]
pub enum SignalingEvent {
    /// Video RTP packet received (H.264/H.265)
    VideoPacket(Bytes),
    /// Audio RTP packet received (Opus)
    AudioPacket(Bytes),
    /// Input channel is ready
    InputReady,
    /// Connection established
    Connected,
    /// Connection disconnected
    Disconnected(String),
    /// Error occurred
    Error(String),
}

/// GFN Peer Protocol message format
#[derive(Debug, Serialize, Deserialize)]
struct GfnPeerMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    ackid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ack: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_info: Option<GfnPeerInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_msg: Option<GfnPeerMsgContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hb: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GfnPeerInfo {
    browser: String,
    #[serde(rename = "browserVersion")]
    browser_version: String,
    connected: bool,
    id: u32,
    name: String,
    peer_role: u32,
    resolution: String,
    version: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct GfnPeerMsgContent {
    from: u32,
    to: u32,
    msg: String,
}

/// SDP Offer/Answer message
#[derive(Debug, Serialize, Deserialize)]
struct SdpMessage {
    #[serde(rename = "type")]
    msg_type: String,
    sdp: Option<String>,
    #[serde(rename = "nvstSdp", skip_serializing_if = "Option::is_none")]
    nvst_sdp: Option<String>,
}

/// ICE Candidate message
#[derive(Debug, Serialize, Deserialize)]
struct IceCandidateMessage {
    candidate: String,
    #[serde(rename = "sdpMid", skip_serializing_if = "Option::is_none")]
    sdp_mid: Option<String>,
    #[serde(rename = "sdpMLineIndex", skip_serializing_if = "Option::is_none")]
    sdp_mline_index: Option<u32>,
}

/// Channel for sending ICE candidates to the WebSocket task
type IceCandidateSender = mpsc::UnboundedSender<IceCandidateMessage>;

/// Signaling client for GFN streaming
pub struct SignalingClient {
    server: String,
    session_id: String,
    width: u32,
    height: u32,
    fps: u32,
    event_tx: Option<mpsc::Sender<SignalingEvent>>,
    input_tx: Option<mpsc::Sender<Vec<u8>>>,
    peer_connection: Option<Arc<RTCPeerConnection>>,
    ack_id: Arc<ParkingMutex<u32>>,
    /// Input data channel for sending mouse/keyboard events
    input_channel: Arc<ParkingMutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
    /// Input protocol version from handshake (version > 2 needs packet wrapping)
    input_protocol_version: Arc<ParkingMutex<u16>>,
    /// Stream start time for relative timestamps (in milliseconds)
    stream_start_time: Arc<ParkingMutex<Option<std::time::Instant>>>,
}

impl SignalingClient {
    /// Create a new signaling client
    pub fn new(server: String, session_id: String, width: u32, height: u32, fps: u32) -> Self {
        Self {
            server,
            session_id,
            width,
            height,
            fps,
            event_tx: None,
            input_tx: None,
            peer_connection: None,
            ack_id: Arc::new(ParkingMutex::new(0)),
            input_channel: Arc::new(ParkingMutex::new(None)),
            input_protocol_version: Arc::new(ParkingMutex::new(0)),
            stream_start_time: Arc::new(ParkingMutex::new(None)),
        }
    }

    /// Create a new signaling client with shared input channel references
    /// This allows the SDL event loop to send input directly without mpsc channel
    pub fn new_with_refs(
        server: String,
        session_id: String,
        width: u32,
        height: u32,
        fps: u32,
        input_channel: Arc<ParkingMutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
        input_protocol_version: Arc<ParkingMutex<u16>>,
        stream_start_time: Arc<ParkingMutex<Option<std::time::Instant>>>,
    ) -> Self {
        Self {
            server,
            session_id,
            width,
            height,
            fps,
            event_tx: None,
            input_tx: None,
            peer_connection: None,
            ack_id: Arc::new(ParkingMutex::new(0)),
            input_channel,
            input_protocol_version,
            stream_start_time,
        }
    }

    /// Connect to the signaling server and establish WebRTC session
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<SignalingEvent>> {
        // Generate random peer ID (matching GFN browser format)
        let random_peer_id: u64 = rand::random::<u64>() % 10_000_000_000;
        let peer_name = format!("peer-{}", random_peer_id);

        // Build signaling URL
        let ws_url = format!(
            "wss://{}/nvst/sign_in?peer_id={}&version=2",
            self.server, peer_name
        );

        info!("Connecting to GFN signaling: {}", ws_url);

        // Create channels for events and input
        // Use small buffer for lowest latency - drop old frames if behind
        let (event_tx, event_rx) = mpsc::channel(8);
        let (input_tx, input_rx) = mpsc::channel::<Vec<u8>>(16);

        // Channel for ICE candidates from peer connection to WebSocket
        let (ice_tx, mut ice_rx) = mpsc::unbounded_channel::<IceCandidateMessage>();

        self.event_tx = Some(event_tx.clone());
        self.input_tx = Some(input_tx);

        // Build WebSocket request with subprotocol for auth
        let subprotocol = format!("x-nv-sessionid.{}", self.session_id);
        let mut request = ws_url.into_client_request()?;
        request.headers_mut().insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            subprotocol.parse().unwrap(),
        );

        // Connect to WebSocket
        let (ws_stream, response) = connect_async_with_config(request, None, false)
            .await
            .context("Failed to connect to signaling server")?;

        info!(
            "Connected to signaling server, protocol: {:?}",
            response.headers().get(header::SEC_WEBSOCKET_PROTOCOL)
        );

        let (mut write, mut read) = ws_stream.split();

        // Send peer_info immediately after connection
        let peer_info_msg = GfnPeerMessage {
            ackid: Some(self.next_ack_id()),
            ack: None,
            peer_info: Some(GfnPeerInfo {
                browser: "Chrome".to_string(),
                browser_version: "131".to_string(),
                connected: true,
                id: 2, // Client is always peer 2
                name: peer_name.clone(),
                peer_role: 0, // 0 = client
                resolution: format!("{}x{}", self.width, self.height),
                version: 2,
            }),
            peer_msg: None,
            hb: None,
        };

        let peer_info_json = serde_json::to_string(&peer_info_msg)?;
        debug!("Sending peer_info: {}", peer_info_json);
        write.send(Message::Text(peer_info_json)).await?;

        // Create WebRTC peer connection with ICE candidate sender
        let pc = self.create_peer_connection(
            event_tx.clone(),
            ice_tx,
            self.input_channel.clone(),
            self.input_protocol_version.clone(),
            self.stream_start_time.clone(),
        ).await?;
        self.peer_connection = Some(pc.clone());

        // Spawn task to handle WebSocket messages and WebRTC negotiation
        let ack_id = self.ack_id.clone();
        let width = self.width;
        let height = self.height;
        let fps = self.fps;
        let server = self.server.clone();

        tokio::spawn(async move {
            // Start heartbeat
            let heartbeat_write = Arc::new(tokio::sync::Mutex::new(write));
            let hb_write = heartbeat_write.clone();
            let ice_write = heartbeat_write.clone();

            let heartbeat_task = tokio::spawn(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
                loop {
                    interval.tick().await;
                    let msg = serde_json::to_string(&GfnPeerMessage {
                        ackid: None,
                        ack: None,
                        peer_info: None,
                        peer_msg: None,
                        hb: Some(1),
                    })
                    .unwrap();
                    let mut w = hb_write.lock().await;
                    if w.send(Message::Text(msg)).await.is_err() {
                        break;
                    }
                }
            });

            // ICE candidate sender task
            let ice_ack_id = ack_id.clone();
            let ice_task = tokio::spawn(async move {
                while let Some(candidate) = ice_rx.recv().await {
                    let next_id = {
                        let mut id = ice_ack_id.lock();
                        *id += 1;
                        *id
                    };

                    let ice_msg = GfnPeerMessage {
                        ackid: Some(next_id),
                        ack: None,
                        peer_info: None,
                        peer_msg: Some(GfnPeerMsgContent {
                            from: 2,
                            to: 1,
                            msg: serde_json::to_string(&candidate).unwrap(),
                        }),
                        hb: None,
                    };

                    let mut w = ice_write.lock().await;
                    if w.send(Message::Text(serde_json::to_string(&ice_msg).unwrap()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    info!("Sent local ICE candidate to server");
                }
            });

            // Main message loop
            loop {
                tokio::select! {
                    msg = read.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                if let Err(e) = handle_gfn_message(
                                    &text,
                                    &pc,
                                    &heartbeat_write,
                                    &ack_id,
                                    &event_tx,
                                    &server,
                                    width,
                                    height,
                                    fps,
                                ).await {
                                    warn!("Failed to handle message: {}", e);
                                }
                            }
                            Some(Ok(Message::Close(frame))) => {
                                let reason = frame
                                    .map(|f| f.reason.to_string())
                                    .unwrap_or_else(|| "Unknown".to_string());
                                let _ = event_tx.send(SignalingEvent::Disconnected(reason)).await;
                                break;
                            }
                            Some(Ok(Message::Ping(data))) => {
                                let mut w = heartbeat_write.lock().await;
                                let _ = w.send(Message::Pong(data)).await;
                            }
                            Some(Err(e)) => {
                                error!("WebSocket error: {}", e);
                                let _ = event_tx.send(SignalingEvent::Error(e.to_string())).await;
                                break;
                            }
                            None => {
                                let _ = event_tx.send(SignalingEvent::Disconnected("Connection closed".to_string())).await;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }

            heartbeat_task.abort();
            ice_task.abort();
        });

        // Spawn input sender task with protocol version info
        self.spawn_input_sender(
            input_rx,
            self.input_protocol_version.clone(),
            self.stream_start_time.clone(),
        );

        Ok(event_rx)
    }

    /// Create and configure WebRTC peer connection
    async fn create_peer_connection(
        &self,
        event_tx: mpsc::Sender<SignalingEvent>,
        ice_tx: IceCandidateSender,
        input_channel_storage: Arc<ParkingMutex<Option<Arc<webrtc::data_channel::RTCDataChannel>>>>,
        input_protocol_version: Arc<ParkingMutex<u16>>,
        stream_start_time: Arc<ParkingMutex<Option<std::time::Instant>>>,
    ) -> Result<Arc<RTCPeerConnection>> {
        // Create media engine with H.264 and Opus codecs
        let mut media_engine = MediaEngine::default();

        // Register H.264 codec for video (OpenH264 decoder)
        media_engine.register_codec(
            webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecParameters {
                capability: RTCRtpCodecCapability {
                    mime_type: "video/H264".to_string(),
                    clock_rate: 90000,
                    channels: 0,
                    sdp_fmtp_line:
                        "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"
                            .to_string(),
                    rtcp_feedback: vec![],
                },
                payload_type: 96,
                ..Default::default()
            },
            RTPCodecType::Video,
        )?;

        // Register Opus codec for audio
        media_engine.register_codec(
            webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecParameters {
                capability: RTCRtpCodecCapability {
                    mime_type: "audio/opus".to_string(),
                    clock_rate: 48000,
                    channels: 2,
                    sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
                    rtcp_feedback: vec![],
                },
                payload_type: 111,
                ..Default::default()
            },
            RTPCodecType::Audio,
        )?;

        // Create interceptor registry
        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut media_engine)?;

        // Configure SettingEngine for ice-lite server compatibility
        // When connecting to an ice-lite server, we need to be DTLS Client
        // because ice-lite servers are always DTLS Server
        // Per RFC 5763: setup:active (DTLS client) is RECOMMENDED for answerers
        let mut setting_engine = webrtc::api::setting_engine::SettingEngine::default();
        setting_engine
            .set_answering_dtls_role(webrtc::dtls_transport::dtls_role::DTLSRole::Client)
            .map_err(|e| anyhow::anyhow!("Failed to set DTLS role: {}", e))?;

        // Build API with setting engine
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .with_setting_engine(setting_engine)
            .build();

        // Configure ICE servers and policies to match browser behavior
        // Only use NVIDIA's STUN server
        let config = RTCConfiguration {
            ice_servers: vec![
                RTCIceServer {
                    urls: vec!["stun:turn.gamestream.nvidia.com:19302".to_string()],
                    ..Default::default()
                },
            ],
            // Match browser configuration for GFN
            bundle_policy: webrtc::peer_connection::policy::bundle_policy::RTCBundlePolicy::MaxBundle,
            rtcp_mux_policy: webrtc::peer_connection::policy::rtcp_mux_policy::RTCRtcpMuxPolicy::Require,
            // Use 2 candidates in pool like official client
            ice_candidate_pool_size: 2,
            ..Default::default()
        };

        // Create peer connection
        let pc = Arc::new(api.new_peer_connection(config).await?);

        // Set up ICE candidate handler to send candidates to server
        pc.on_ice_candidate(Box::new(move |candidate| {
            let ice_tx = ice_tx.clone();
            Box::pin(async move {
                if let Some(c) = candidate {
                    let candidate_str = c.to_json().map(|j| j.candidate).unwrap_or_default();
                    info!("Local ICE candidate: {}", candidate_str);

                    let msg = IceCandidateMessage {
                        candidate: candidate_str,
                        sdp_mid: c.to_json().ok().and_then(|j| j.sdp_mid),
                        sdp_mline_index: c.to_json().ok().and_then(|j| j.sdp_mline_index.map(|i| i as u32)),
                    };

                    let _ = ice_tx.send(msg);
                } else {
                    info!("ICE gathering complete");
                }
            })
        }));

        // Set up track handler for incoming media
        let event_tx_track = event_tx.clone();
        pc.on_track(Box::new(move |track, _receiver, _transceiver| {
            let event_tx = event_tx_track.clone();
            Box::pin(async move {
                let track = track;
                info!(
                    "Track received: {} ({})",
                    track.kind(),
                    track.codec().capability.mime_type
                );

                // Spawn task to read RTP packets from this track
                tokio::spawn(read_track_packets(track, event_tx));
            })
        }));

        // Set up connection state handler
        let event_tx_state = event_tx.clone();
        pc.on_peer_connection_state_change(Box::new(move |state| {
            let event_tx = event_tx_state.clone();
            Box::pin(async move {
                info!("Peer connection state: {:?}", state);
                match state {
                    RTCPeerConnectionState::Connected => {
                        let _ = event_tx.send(SignalingEvent::Connected).await;
                    }
                    RTCPeerConnectionState::Disconnected => {
                        let _ = event_tx
                            .send(SignalingEvent::Disconnected("Peer disconnected".to_string()))
                            .await;
                    }
                    RTCPeerConnectionState::Failed => {
                        let _ = event_tx
                            .send(SignalingEvent::Error("Connection failed".to_string()))
                            .await;
                    }
                    _ => {}
                }
            })
        }));

        // Set up ICE connection state handler for debugging
        pc.on_ice_connection_state_change(Box::new(move |state| {
            Box::pin(async move {
                info!("ICE connection state: {:?}", state);
            })
        }));

        // Set up ICE gathering state handler
        pc.on_ice_gathering_state_change(Box::new(move |state| {
            Box::pin(async move {
                info!("ICE gathering state: {:?}", state);
            })
        }));

        // Set up data channel handler for server-created channels
        let event_tx_dc = event_tx.clone();
        pc.on_data_channel(Box::new(move |dc| {
            let event_tx = event_tx_dc.clone();
            let label = dc.label().to_string();
            info!("Data channel received from server: {}", label);

            Box::pin(async move {
                // Handle input channel
                if label.contains("input") || label.contains("ri_") {
                    dc.on_open(Box::new(move || {
                        info!("Server input data channel opened");
                        Box::pin(async {})
                    }));

                    let event_tx_msg = event_tx.clone();
                    dc.on_message(Box::new(move |msg| {
                        let event_tx = event_tx_msg.clone();
                        let data = msg.data.to_vec();
                        Box::pin(async move {
                            // Check for handshake message
                            if data.len() >= 2 {
                                let first_word = u16::from_le_bytes([data[0], data[1]]);
                                if first_word == 526 || data.len() == 4 {
                                    info!("Input handshake received from server channel");
                                    let _ = event_tx.send(SignalingEvent::InputReady).await;
                                }
                            }
                        })
                    }));
                }
            })
        }));

        // CRITICAL: Create input data channel BEFORE SDP negotiation
        // Per official GFN browser client, the input channel must be created before
        // setRemoteDescription is called. This ensures the server recognizes the channel.
        // Note: create_data_channel already returns Arc<RTCDataChannel>
        let input_dc = pc.create_data_channel(
            "input_channel_v1",
            Some(webrtc::data_channel::data_channel_init::RTCDataChannelInit {
                ordered: Some(false),           // Unordered for lowest latency
                max_retransmits: Some(0),       // No retransmits - next packet has updated data
                ..Default::default()
            }),
        ).await?;

        // Store reference for sending input events
        let input_channel_for_open = input_channel_storage.clone();
        let input_dc_for_open = input_dc.clone();
        input_dc.on_open(Box::new(move || {
            info!("Client input_channel_v1 opened - storing channel reference");
            // Store the channel for later use
            {
                let mut storage = input_channel_for_open.lock();
                *storage = Some(input_dc_for_open.clone());
            }
            info!("Input channel stored, waiting for server handshake");
            Box::pin(async {})
        }));

        let event_tx_input_msg = event_tx.clone();
        let input_dc_for_msg = input_dc.clone();
        let protocol_version_for_msg = input_protocol_version.clone();
        let stream_start_for_msg = stream_start_time.clone();
        input_dc.on_message(Box::new(move |msg| {
            let event_tx = event_tx_input_msg.clone();
            let input_dc = input_dc_for_msg.clone();
            let protocol_version = protocol_version_for_msg.clone();
            let stream_start = stream_start_for_msg.clone();
            let data = msg.data.to_vec();
            Box::pin(async move {
                info!("Input channel message received: {} bytes", data.len());
                if data.len() >= 2 {
                    let first_word = u16::from_le_bytes([data[0], data[1]]);
                    info!("Input message first word: 0x{:04x} ({})", first_word, first_word);

                    // Handshake: 0x020E (526) = new format, or 4-byte message
                    if first_word == 526 || data.len() == 4 {
                        info!("Input handshake received on client channel!");

                        // Parse protocol version from handshake
                        // Format: [0x0E, 0x02, version_lo, version_hi] where version is bytes 2-3 as u16 LE
                        let version = if data.len() >= 4 {
                            u16::from_le_bytes([data[2], data[3]])
                        } else {
                            // Old format: first word is the version directly
                            first_word
                        };

                        info!("Input protocol version: {} (wrapper needed: {})", version, version > 2);

                        // Store protocol version
                        {
                            let mut pv = protocol_version.lock();
                            *pv = version;
                        }

                        // Start stream timer for relative timestamps
                        {
                            let mut st = stream_start.lock();
                            *st = Some(std::time::Instant::now());
                        }

                        // CRITICAL: Send handshake response back to server
                        // Per GFN protocol, we must echo the handshake bytes back
                        let response = bytes::Bytes::copy_from_slice(&data);
                        match input_dc.send(&response).await {
                            Ok(_) => info!("Input handshake response sent: {:02x?}", &data),
                            Err(e) => error!("Failed to send handshake response: {}", e),
                        }

                        let _ = event_tx.send(SignalingEvent::InputReady).await;
                    }
                }
            })
        }));

        info!("Created input_channel_v1 data channel");

        Ok(pc)
    }

    /// Spawn task to send input data over WebRTC data channel
    fn spawn_input_sender(
        &self,
        mut input_rx: mpsc::Receiver<Vec<u8>>,
        protocol_version: Arc<ParkingMutex<u16>>,
        stream_start_time: Arc<ParkingMutex<Option<std::time::Instant>>>,
    ) {
        let input_channel = self.input_channel.clone();

        tokio::spawn(async move {
            let mut send_count: u64 = 0;
            while let Some(data) = input_rx.recv().await {
                // Get the input channel
                let channel = {
                    let storage = input_channel.lock();
                    storage.clone()
                };

                if let Some(dc) = channel {
                    // Check if channel is open
                    if dc.ready_state() == webrtc::data_channel::data_channel_state::RTCDataChannelState::Open {
                        // Get protocol version
                        let version = {
                            let pv = protocol_version.lock();
                            *pv
                        };

                        // Build the final packet - wrap if protocol version > 2
                        let final_packet = if version > 2 {
                            // v3+ protocol requires 10-byte header wrapper
                            // Format: [0x23 (1 byte)][timestamp μs (8 bytes BE)][0x22 (1 byte)][original packet]
                            let timestamp_us = {
                                let st = stream_start_time.lock();
                                if let Some(start) = *st {
                                    start.elapsed().as_micros() as u64
                                } else {
                                    0
                                }
                            };

                            let mut wrapped = Vec::with_capacity(10 + data.len());
                            wrapped.push(0x23);  // Type marker
                            wrapped.extend_from_slice(&timestamp_us.to_be_bytes());  // Timestamp (BE)
                            wrapped.push(0x22);  // Single event wrapper
                            wrapped.extend_from_slice(&data);  // Original input packet
                            wrapped
                        } else {
                            // No wrapper needed for v2 and below
                            data.clone()
                        };

                        let packet = bytes::Bytes::copy_from_slice(&final_packet);
                        match dc.send(&packet).await {
                            Ok(_) => {
                                send_count += 1;
                                // Log first few sends at INFO level to verify input is working
                                if send_count <= 10 {
                                    info!(
                                        "INPUT SENT #{}: {} bytes (v{} {}wrapped), data: {:02x?}",
                                        send_count,
                                        final_packet.len(),
                                        version,
                                        if version > 2 { "" } else { "un" },
                                        &final_packet[..final_packet.len().min(30)]
                                    );
                                } else if send_count % 500 == 0 {
                                    info!("Input still flowing: {} packets sent (v{} protocol)", send_count, version);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to send input: {}", e);
                            }
                        }
                    } else {
                        debug!("Input channel not open, state: {:?}", dc.ready_state());
                    }
                } else {
                    // Log occasionally that channel isn't ready
                    if send_count == 0 {
                        debug!("Input channel not ready yet, dropping input event");
                    }
                }
            }
            info!("Input sender task ended after {} sends", send_count);
        });
    }

    /// Get next ack ID
    fn next_ack_id(&self) -> u32 {
        let mut id = self.ack_id.lock();
        *id += 1;
        *id
    }

    /// Send input data to the server
    pub async fn send_input(&self, data: &[u8]) -> Result<()> {
        if let Some(ref tx) = self.input_tx {
            tx.send(data.to_vec())
                .await
                .context("Failed to send input")?;
        }
        Ok(())
    }
}

/// Read RTP packets from a track and forward to event channel
async fn read_track_packets(track: Arc<TrackRemote>, event_tx: mpsc::Sender<SignalingEvent>) {
    let kind = track.kind().to_string();
    info!("Starting to read {} track packets", kind);
    
    let mut packet_count: u64 = 0;
    let mut total_bytes: u64 = 0;
    let start_time = std::time::Instant::now();

    loop {
        match track.read_rtp().await {
            Ok((rtp_packet, _attributes)) => {
                // Extract the payload from the RTP packet
                let payload = rtp_packet.payload;
                if payload.is_empty() {
                    continue;
                }
                
                packet_count += 1;
                total_bytes += payload.len() as u64;
                
                // Log first few packets and then periodically
                if packet_count <= 5 || packet_count % 300 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let rate = if elapsed > 0.0 { packet_count as f64 / elapsed } else { 0.0 };
                    
                    if kind == "video" {
                        // Parse NAL type for video packets
                        let nal_type = payload[0] & 0x1F;
                        info!(
                            "Video packet #{}: {} bytes, NAL type={}, seq={}, ts={}, rate={:.1} pkt/s",
                            packet_count,
                            payload.len(),
                            nal_type,
                            rtp_packet.header.sequence_number,
                            rtp_packet.header.timestamp,
                            rate
                        );
                    } else {
                        info!(
                            "Audio packet #{}: {} bytes, seq={}, rate={:.1} pkt/s",
                            packet_count,
                            payload.len(),
                            rtp_packet.header.sequence_number,
                            rate
                        );
                    }
                }

                let packet = payload;

                let event = if kind == "video" {
                    SignalingEvent::VideoPacket(packet)
                } else {
                    SignalingEvent::AudioPacket(packet)
                };

                if event_tx.send(event).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                warn!("Track read error: {}", e);
                break;
            }
        }
    }

    info!("{} track ended after {} packets ({} bytes)", kind, packet_count, total_bytes);
}

/// Handle a GFN protocol message
async fn handle_gfn_message(
    text: &str,
    pc: &Arc<RTCPeerConnection>,
    write: &Arc<
        tokio::sync::Mutex<
            impl SinkExt<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
        >,
    >,
    ack_id: &Arc<ParkingMutex<u32>>,
    _event_tx: &mpsc::Sender<SignalingEvent>,
    server: &str,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<()> {
    debug!("Received: {}", &text[..text.len().min(200)]);

    let message: GfnPeerMessage = serde_json::from_str(text)?;

    // Send ack for messages with ackid (except our own echoes)
    if let Some(ackid) = message.ackid {
        // Don't ack our own peer_info echo
        let is_our_echo = message
            .peer_info
            .as_ref()
            .map(|p| p.id == 2)
            .unwrap_or(false);

        if !is_our_echo {
            let ack_msg = serde_json::to_string(&GfnPeerMessage {
                ackid: None,
                ack: Some(ackid),
                peer_info: None,
                peer_msg: None,
                hb: None,
            })?;
            let mut w = write.lock().await;
            w.send(Message::Text(ack_msg)).await?;
        }
    }

    // Handle heartbeat
    if message.hb.is_some() {
        let hb_msg = serde_json::to_string(&GfnPeerMessage {
            ackid: None,
            ack: None,
            peer_info: None,
            peer_msg: None,
            hb: Some(1),
        })?;
        let mut w = write.lock().await;
        w.send(Message::Text(hb_msg)).await?;
        return Ok(());
    }

    // Handle peer messages (SDP offer, ICE candidates)
    if let Some(peer_msg) = message.peer_msg {
        let inner: serde_json::Value = serde_json::from_str(&peer_msg.msg)?;

        if let Some(msg_type) = inner.get("type").and_then(|t| t.as_str()) {
            match msg_type {
                "offer" => {
                    info!("Received SDP offer");

                    let sdp = inner
                        .get("sdp")
                        .and_then(|s| s.as_str())
                        .context("Missing SDP in offer")?;

                    // Check for ice-lite
                    let is_ice_lite = sdp.contains("a=ice-lite");
                    info!("Server uses ice-lite: {}", is_ice_lite);

                    // Extract server ICE credentials for later use with trickle candidates
                    let server_ufrag = sdp
                        .lines()
                        .find(|l| l.starts_with("a=ice-ufrag:"))
                        .map(|l| l.trim_start_matches("a=ice-ufrag:").to_string())
                        .unwrap_or_default();
                    info!("Server ICE ufrag: {}", server_ufrag);

                    // Set remote description FIRST (before creating answer)
                    let offer = RTCSessionDescription::offer(sdp.to_string())?;
                    pc.set_remote_description(offer).await?;
                    info!("Remote SDP offer set");

                    // Create answer
                    let answer = pc.create_answer(None).await?;
                    pc.set_local_description(answer.clone()).await?;
                    info!("Local SDP answer created");

                    // Wait for first srflx (server-reflexive) ICE candidate OR timeout
                    // Matching TypeScript behavior: don't wait for complete gathering
                    // as that can take too long and cause server timeout
                    info!("Waiting for initial ICE candidates (srflx or 1s timeout)...");
                    let gather_start = std::time::Instant::now();
                    let max_wait = std::time::Duration::from_millis(1000);

                    loop {
                        // Check if we have at least one srflx candidate
                        if let Some(local_desc) = pc.local_description().await {
                            if local_desc.sdp.contains("typ srflx") {
                                info!("Found srflx candidate, proceeding with answer");
                                break;
                            }
                        }

                        // Check for timeout
                        if gather_start.elapsed() >= max_wait {
                            info!("ICE gathering timeout (1s), proceeding with available candidates");
                            break;
                        }

                        // Check if gathering completed
                        let state = pc.ice_gathering_state();
                        if state == webrtc::ice_transport::ice_gathering_state::RTCIceGatheringState::Complete {
                            info!("ICE gathering complete");
                            break;
                        }

                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                    info!("Proceeding with answer after {:?}", gather_start.elapsed());

                    // Get the current local description which may have gathered candidates
                    let local_desc = pc.local_description().await;
                    let answer_sdp = local_desc
                        .as_ref()
                        .map(|d| d.sdp.clone())
                        .unwrap_or_else(|| answer.sdp.clone());

                    // Build nvstSdp with streaming parameters
                    let nvst_sdp = build_nvst_sdp(&answer_sdp, width, height, fps);

                    // Send answer
                    let answer_msg = SdpMessage {
                        msg_type: "answer".to_string(),
                        sdp: Some(answer_sdp),
                        nvst_sdp: Some(nvst_sdp),
                    };

                    let next_id = {
                        let mut id = ack_id.lock();
                        *id += 1;
                        *id
                    };

                    let peer_msg = GfnPeerMessage {
                        ackid: Some(next_id),
                        ack: None,
                        peer_info: None,
                        peer_msg: Some(GfnPeerMsgContent {
                            from: 2,
                            to: 1,
                            msg: serde_json::to_string(&answer_msg)?,
                        }),
                        hb: None,
                    };

                    let mut w = write.lock().await;
                    w.send(Message::Text(serde_json::to_string(&peer_msg)?))
                        .await?;
                    info!("SDP answer sent");

                    // For ice-lite servers, we need to add a manual ICE candidate
                    // constructed from the server IP and SDP port, IN ADDITION to
                    // any trickle ICE candidates that may come later.
                    // This matches the browser client behavior (streaming.ts:1127-1178)
                    if is_ice_lite {
                        info!("ice-lite server detected - adding manual ICE candidate from SDP");
                        
                        // Extract port from SDP (from m=audio or m=video line)
                        let port = sdp
                            .lines()
                            .find(|l| l.starts_with("m=audio") || l.starts_with("m=video"))
                            .and_then(|l| l.split_whitespace().nth(1))
                            .and_then(|p| p.parse::<u16>().ok())
                            .unwrap_or(47998);
                        
                        info!("Extracted port from SDP: {}", port);
                        
                        // Convert server hostname to IP address
                        // Format: 80-250-101-43.cloudmatchbeta.nvidiagrid.net -> 80.250.101.43
                        let server_ip = extract_ip_from_hostname(server);
                        info!("Server IP for ICE candidate: {}", server_ip);
                        
                        // Construct the ICE candidate
                        // Format: candidate:foundation component protocol priority ip port typ type
                        let candidate_string = format!(
                            "candidate:1 1 udp 2130706431 {} {} typ host",
                            server_ip, port
                        );
                        info!("Constructed manual ICE candidate: {}", candidate_string);
                        
                        // Add the candidate for all media lines (0, 1, 2, 3)
                        // The browser tries multiple sdpMid values
                        for mid in ["0", "1", "2", "3"] {
                            let ice_candidate = RTCIceCandidateInit {
                                candidate: candidate_string.clone(),
                                sdp_mid: Some(mid.to_string()),
                                sdp_mline_index: Some(mid.parse().unwrap_or(0)),
                                username_fragment: Some(server_ufrag.clone()),
                            };
                            
                            match pc.add_ice_candidate(ice_candidate).await {
                                Ok(_) => {
                                    info!("Added manual ICE candidate with sdpMid={}", mid);
                                    break; // Success, no need to try other mids
                                }
                                Err(e) => {
                                    debug!("Failed to add ICE candidate with sdpMid={}: {}", mid, e);
                                }
                            }
                        }
                        
                        info!("Manual ICE candidate added, also waiting for trickle ICE candidates");
                    }
                }
                _ => {
                    debug!("Unhandled message type: {}", msg_type);
                }
            }
        }

        // Handle ICE candidates (trickle ICE) - this is the PRIMARY way ice-lite servers send candidates
        if inner.get("candidate").is_some() {
            let candidate_str = inner
                .get("candidate")
                .and_then(|c| c.as_str())
                .unwrap_or("");

            if !candidate_str.is_empty() {
                info!("Received remote ICE candidate: {}", candidate_str);

                // Get sdpMid and sdpMLineIndex from the message
                // If not provided, default to "0" for the first media line (video)
                let sdp_mid = inner
                    .get("sdpMid")
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| Some("0".to_string()));

                let sdp_mline_index = inner
                    .get("sdpMLineIndex")
                    .and_then(|i| i.as_u64())
                    .map(|i| i as u16)
                    .or(Some(0));

                info!(
                    "Adding candidate with sdpMid={:?}, sdpMLineIndex={:?}",
                    sdp_mid, sdp_mline_index
                );

                // For trickle ICE, do NOT pass username_fragment
                // Matching TypeScript browser behavior: the library infers it from remote SDP
                // Passing an incorrect/missing usernameFragment causes "unknown TransactionID"
                let ice_candidate = RTCIceCandidateInit {
                    candidate: candidate_str.to_string(),
                    sdp_mid,
                    sdp_mline_index,
                    username_fragment: None, // Let webrtc-rs infer from remote description
                };

                if let Err(e) = pc.add_ice_candidate(ice_candidate).await {
                    warn!("Failed to add remote ICE candidate: {}", e);
                } else {
                    info!("Added remote ICE candidate successfully");

                    // Log current ICE and peer connection state
                    info!("Current ICE connection state: {:?}", pc.ice_connection_state());
                    info!("Current peer connection state: {:?}", pc.connection_state());
                }
            }
        }
    }

    Ok(())
}

/// Extract IP address from GFN server hostname
/// Format: 80-250-101-43.cloudmatchbeta.nvidiagrid.net -> 80.250.101.43
fn extract_ip_from_hostname(hostname: &str) -> String {
    // Try to extract IP from hostname format like "80-250-101-43.cloudmatchbeta..."
    if let Some(first_part) = hostname.split('.').next() {
        let parts: Vec<&str> = first_part.split('-').collect();
        if parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok()) {
            return parts.join(".");
        }
    }
    
    // If hostname is already an IP address, return as-is
    if hostname.parse::<std::net::Ipv4Addr>().is_ok() {
        return hostname.to_string();
    }
    
    // Fallback: return the hostname and let ICE handle it
    // (This shouldn't happen with GFN servers)
    warn!("Could not extract IP from hostname: {}", hostname);
    hostname.to_string()
}

/// Build nvstSdp string with streaming parameters
fn build_nvst_sdp(answer_sdp: &str, width: u32, height: u32, fps: u32) -> String {
    // Extract ICE credentials from answer SDP
    let ice_ufrag = answer_sdp
        .lines()
        .find(|l| l.starts_with("a=ice-ufrag:"))
        .map(|l| l.trim_start_matches("a=ice-ufrag:"))
        .unwrap_or("");

    let ice_pwd = answer_sdp
        .lines()
        .find(|l| l.starts_with("a=ice-pwd:"))
        .map(|l| l.trim_start_matches("a=ice-pwd:"))
        .unwrap_or("");

    let fingerprint = answer_sdp
        .lines()
        .find(|l| l.starts_with("a=fingerprint:sha-256"))
        .map(|l| l.trim_start_matches("a=fingerprint:sha-256 "))
        .unwrap_or("");

    let is_high_fps = fps >= 120;
    let is_120_fps = fps == 120;

    let mut lines = vec![
        "v=0".to_string(),
        "o=SdpTest test_id_13 14 IN IPv4 127.0.0.1".to_string(),
        "s=-".to_string(),
        "t=0 0".to_string(),
        format!("a=general.icePassword:{}", ice_pwd),
        format!("a=general.iceUserNameFragment:{}", ice_ufrag),
        format!("a=general.dtlsFingerprint:{}", fingerprint),
        "m=video 0 RTP/AVP".to_string(),
        "a=msid:fbc-video-0".to_string(),
        // FEC settings
        "a=vqos.fec.rateDropWindow:10".to_string(),
        "a=vqos.fec.minRequiredFecPackets:2".to_string(),
        "a=vqos.fec.repairMinPercent:5".to_string(),
        "a=vqos.fec.repairPercent:5".to_string(),
        "a=vqos.fec.repairMaxPercent:35".to_string(),
    ];

    // DRC/DFC settings based on FPS
    if is_high_fps {
        lines.extend([
            "a=vqos.drc.enable:0".to_string(),
            "a=vqos.dfc.enable:1".to_string(),
            "a=vqos.dfc.decodeFpsAdjPercent:85".to_string(),
            "a=vqos.dfc.targetDownCooldownMs:250".to_string(),
            "a=vqos.dfc.dfcAlgoVersion:2".to_string(),
            format!(
                "a=vqos.dfc.minTargetFps:{}",
                if is_120_fps { 100 } else { 60 }
            ),
        ]);
    } else {
        lines.push("a=vqos.drc.minRequiredBitrateCheckEnabled:1".to_string());
    }

    // Video encoder settings
    lines.extend([
        "a=video.dx9EnableNv12:1".to_string(),
        "a=video.dx9EnableHdr:1".to_string(),
        "a=vqos.qpg.enable:1".to_string(),
        "a=vqos.resControl.qp.qpg.featureSetting:7".to_string(),
        "a=bwe.useOwdCongestionControl:1".to_string(),
        "a=video.enableRtpNack:1".to_string(),
        "a=vqos.bw.txRxLag.minFeedbackTxDeltaMs:200".to_string(),
        "a=vqos.drc.bitrateIirFilterFactor:18".to_string(),
        "a=video.packetSize:1140".to_string(),
        "a=packetPacing.minNumPacketsPerGroup:15".to_string(),
    ]);

    // High FPS optimizations
    if is_high_fps {
        lines.extend([
            "a=bwe.iirFilterFactor:8".to_string(),
            "a=video.encoderFeatureSetting:47".to_string(),
            "a=video.encoderPreset:6".to_string(),
            "a=vqos.resControl.cpmRtc.badNwSkipFramesCount:600".to_string(),
            "a=vqos.resControl.cpmRtc.decodeTimeThresholdMs:9".to_string(),
            format!(
                "a=video.fbcDynamicFpsGrabTimeoutMs:{}",
                if is_120_fps { 6 } else { 18 }
            ),
            format!(
                "a=vqos.resControl.cpmRtc.serverResolutionUpdateCoolDownCount:{}",
                if is_120_fps { 6000 } else { 12000 }
            ),
        ]);
    }

    // Common settings
    lines.extend([
        "a=vqos.adjustStreamingFpsDuringOutOfFocus:1".to_string(),
        "a=vqos.resControl.cpmRtc.ignoreOutOfFocusWindowState:1".to_string(),
        "a=vqos.resControl.perfHistory.rtcIgnoreOutOfFocusWindowState:1".to_string(),
        "a=vqos.resControl.cpmRtc.featureMask:3".to_string(),
        format!(
            "a=packetPacing.numGroups:{}",
            if is_120_fps { 3 } else { 5 }
        ),
        "a=packetPacing.maxDelayUs:1000".to_string(),
        "a=packetPacing.minNumPacketsFrame:10".to_string(),
        "a=video.rtpNackQueueLength:1024".to_string(),
        "a=video.rtpNackQueueMaxPackets:512".to_string(),
        "a=video.rtpNackMaxPacketCount:25".to_string(),
        "a=vqos.drc.qpMaxResThresholdAdj:4".to_string(),
        "a=vqos.grc.qpMaxResThresholdAdj:4".to_string(),
        "a=vqos.drc.iirFilterFactor:100".to_string(),
        format!("a=video.clientViewportWd:{}", width),
        format!("a=video.clientViewportHt:{}", height),
        format!("a=video.maxFPS:{}", fps),
        "a=video.initialBitrateKbps:50000".to_string(),
        "a=video.initialPeakBitrateKbps:50000".to_string(),
        "a=vqos.bw.maximumBitrateKbps:100000".to_string(),
        "a=vqos.bw.minimumBitrateKbps:10000".to_string(),
        "a=video.maxNumReferenceFrames:4".to_string(),
        "a=video.mapRtpTimestampsToFrames:1".to_string(),
        "a=video.encoderCscMode:3".to_string(),
        "a=video.scalingFeature1:0".to_string(),
        "a=video.prefilterParams.prefilterModel:0".to_string(),
        "m=audio 0 RTP/AVP".to_string(),
        "a=msid:audio".to_string(),
        "m=mic 0 RTP/AVP".to_string(),
        "a=msid:mic".to_string(),
        "m=application 0 RTP/AVP".to_string(),
        "a=msid:input_1".to_string(),
        "a=ri.partialReliableThresholdMs:300".to_string(),
    ]);

    lines.join("\n")
}
