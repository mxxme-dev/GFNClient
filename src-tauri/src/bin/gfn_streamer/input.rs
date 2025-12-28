//! Input Handling Module
//!
//! Handles keyboard and mouse input encoding for the GFN protocol.
//! Uses the binary protocol format discovered from vendor.js analysis.

use bytes::{BytesMut, BufMut};
use winit::keyboard::KeyCode;

/// Input event type constants (from GFN protocol)
/// CRITICAL: These values must match the GFN server expectations exactly!
/// Key down = 3, Key up = 4 (verified from official client TypeScript)
pub const INPUT_HEARTBEAT: u32 = 2;
pub const INPUT_KEY_DOWN: u32 = 3;  // Was incorrectly 4
pub const INPUT_KEY_UP: u32 = 4;    // Was incorrectly 3
pub const INPUT_MOUSE_ABS: u32 = 5;
pub const INPUT_MOUSE_REL: u32 = 7;
pub const INPUT_MOUSE_BUTTON_DOWN: u32 = 8;
pub const INPUT_MOUSE_BUTTON_UP: u32 = 9;
pub const INPUT_MOUSE_WHEEL: u32 = 10;

/// Input events that can be sent to the server
#[derive(Debug, Clone)]
pub enum InputEvent {
    /// Keyboard key pressed
    KeyDown {
        keycode: u16,
        scancode: u16,
        modifiers: u16,
        timestamp_us: u64,
    },
    /// Keyboard key released
    KeyUp {
        keycode: u16,
        scancode: u16,
        modifiers: u16,
        timestamp_us: u64,
    },
    /// Mouse moved (relative)
    MouseMove {
        dx: i16,
        dy: i16,
        timestamp_us: u64,
    },
    /// Mouse button pressed
    MouseButtonDown {
        button: u8,
        timestamp_us: u64,
    },
    /// Mouse button released
    MouseButtonUp {
        button: u8,
        timestamp_us: u64,
    },
    /// Mouse wheel scrolled
    MouseWheel {
        delta_x: i16,
        delta_y: i16,
        timestamp_us: u64,
    },
    /// Heartbeat (keep-alive)
    Heartbeat {
        timestamp_us: u64,
    },
}

/// Encoder for GFN input protocol
pub struct InputEncoder {
    buffer: BytesMut,
}

impl InputEncoder {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(64),
        }
    }

    /// Encode an input event to binary format
    ///
    /// Format from vendor.js analysis:
    /// - Type: 4 bytes LE
    /// - Data: varies by type, Big Endian for multi-byte values
    /// - Timestamp: 8 bytes BE, microseconds
    pub fn encode(&mut self, event: &InputEvent) -> Vec<u8> {
        self.buffer.clear();

        match event {
            InputEvent::KeyDown { keycode, scancode, modifiers, timestamp_us } => {
                // Keyboard: 18 bytes
                // [type 4B LE][keycode 2B BE][modifiers 2B BE][scancode 2B BE][timestamp 8B BE]
                self.buffer.put_u32_le(INPUT_KEY_DOWN);
                self.buffer.put_u16(*keycode);      // BE (default)
                self.buffer.put_u16(*modifiers);    // BE
                self.buffer.put_u16(*scancode);     // BE
                self.buffer.put_u64(*timestamp_us); // BE
            }

            InputEvent::KeyUp { keycode, scancode, modifiers, timestamp_us } => {
                self.buffer.put_u32_le(INPUT_KEY_UP);
                self.buffer.put_u16(*keycode);
                self.buffer.put_u16(*modifiers);
                self.buffer.put_u16(*scancode);
                self.buffer.put_u64(*timestamp_us);
            }

            InputEvent::MouseMove { dx, dy, timestamp_us } => {
                // Mouse Relative: 22 bytes
                // [type 4B LE][dx 2B BE][dy 2B BE][reserved 2B][reserved 4B][timestamp 8B BE]
                self.buffer.put_u32_le(INPUT_MOUSE_REL);
                self.buffer.put_i16(*dx);           // BE
                self.buffer.put_i16(*dy);           // BE
                self.buffer.put_u16(0);             // Reserved
                self.buffer.put_u32(0);             // Reserved
                self.buffer.put_u64(*timestamp_us); // BE
            }

            InputEvent::MouseButtonDown { button, timestamp_us } => {
                // Mouse Button: 18 bytes
                // [type 4B LE][button 1B][pad 1B][reserved 4B][timestamp 8B BE]
                self.buffer.put_u32_le(INPUT_MOUSE_BUTTON_DOWN);
                self.buffer.put_u8(*button);
                self.buffer.put_u8(0);              // Padding
                self.buffer.put_u32(0);             // Reserved
                self.buffer.put_u64(*timestamp_us); // BE
            }

            InputEvent::MouseButtonUp { button, timestamp_us } => {
                self.buffer.put_u32_le(INPUT_MOUSE_BUTTON_UP);
                self.buffer.put_u8(*button);
                self.buffer.put_u8(0);
                self.buffer.put_u32(0);
                self.buffer.put_u64(*timestamp_us);
            }

            InputEvent::MouseWheel { delta_x, delta_y, timestamp_us } => {
                // Mouse Wheel: 22 bytes
                // [type 4B LE][horiz 2B BE][vert 2B BE][reserved 2B BE][reserved 4B][timestamp 8B BE]
                self.buffer.put_u32_le(INPUT_MOUSE_WHEEL);
                self.buffer.put_i16(*delta_x);      // Horizontal
                self.buffer.put_i16(-*delta_y);     // Vertical (negated per vendor.js)
                self.buffer.put_u16(0);             // Reserved
                self.buffer.put_u32(0);             // Reserved
                self.buffer.put_u64(*timestamp_us); // BE
            }

            InputEvent::Heartbeat { timestamp_us: _ } => {
                // Heartbeat: 4 bytes
                self.buffer.put_u32_le(INPUT_HEARTBEAT);
            }
        }

        self.buffer.to_vec()
    }

    /// Encode the protocol handshake response
    pub fn encode_handshake_response(major: u8, minor: u8, flags: u8) -> Vec<u8> {
        vec![0x0e, major, minor, flags]
    }
}

impl Default for InputEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert winit KeyCode to Windows Virtual Key code and scan code
pub fn keycode_to_vk_scan(keycode: KeyCode) -> (u16, u16) {
    match keycode {
        // Letters
        KeyCode::KeyA => (0x41, 0x1E),
        KeyCode::KeyB => (0x42, 0x30),
        KeyCode::KeyC => (0x43, 0x2E),
        KeyCode::KeyD => (0x44, 0x20),
        KeyCode::KeyE => (0x45, 0x12),
        KeyCode::KeyF => (0x46, 0x21),
        KeyCode::KeyG => (0x47, 0x22),
        KeyCode::KeyH => (0x48, 0x23),
        KeyCode::KeyI => (0x49, 0x17),
        KeyCode::KeyJ => (0x4A, 0x24),
        KeyCode::KeyK => (0x4B, 0x25),
        KeyCode::KeyL => (0x4C, 0x26),
        KeyCode::KeyM => (0x4D, 0x32),
        KeyCode::KeyN => (0x4E, 0x31),
        KeyCode::KeyO => (0x4F, 0x18),
        KeyCode::KeyP => (0x50, 0x19),
        KeyCode::KeyQ => (0x51, 0x10),
        KeyCode::KeyR => (0x52, 0x13),
        KeyCode::KeyS => (0x53, 0x1F),
        KeyCode::KeyT => (0x54, 0x14),
        KeyCode::KeyU => (0x55, 0x16),
        KeyCode::KeyV => (0x56, 0x2F),
        KeyCode::KeyW => (0x57, 0x11),
        KeyCode::KeyX => (0x58, 0x2D),
        KeyCode::KeyY => (0x59, 0x15),
        KeyCode::KeyZ => (0x5A, 0x2C),

        // Numbers
        KeyCode::Digit0 => (0x30, 0x0B),
        KeyCode::Digit1 => (0x31, 0x02),
        KeyCode::Digit2 => (0x32, 0x03),
        KeyCode::Digit3 => (0x33, 0x04),
        KeyCode::Digit4 => (0x34, 0x05),
        KeyCode::Digit5 => (0x35, 0x06),
        KeyCode::Digit6 => (0x36, 0x07),
        KeyCode::Digit7 => (0x37, 0x08),
        KeyCode::Digit8 => (0x38, 0x09),
        KeyCode::Digit9 => (0x39, 0x0A),

        // Function keys
        KeyCode::F1 => (0x70, 0x3B),
        KeyCode::F2 => (0x71, 0x3C),
        KeyCode::F3 => (0x72, 0x3D),
        KeyCode::F4 => (0x73, 0x3E),
        KeyCode::F5 => (0x74, 0x3F),
        KeyCode::F6 => (0x75, 0x40),
        KeyCode::F7 => (0x76, 0x41),
        KeyCode::F8 => (0x77, 0x42),
        KeyCode::F9 => (0x78, 0x43),
        KeyCode::F10 => (0x79, 0x44),
        KeyCode::F11 => (0x7A, 0x57),
        KeyCode::F12 => (0x7B, 0x58),

        // Special keys
        KeyCode::Escape => (0x1B, 0x01),
        KeyCode::Tab => (0x09, 0x0F),
        KeyCode::CapsLock => (0x14, 0x3A),
        KeyCode::ShiftLeft => (0x10, 0x2A),
        KeyCode::ShiftRight => (0x10, 0x36),
        KeyCode::ControlLeft => (0x11, 0x1D),
        KeyCode::ControlRight => (0x11, 0x1D),
        KeyCode::AltLeft => (0x12, 0x38),
        KeyCode::AltRight => (0x12, 0x38),
        KeyCode::SuperLeft => (0x5B, 0x5B),
        KeyCode::SuperRight => (0x5C, 0x5C),
        KeyCode::Space => (0x20, 0x39),
        KeyCode::Enter => (0x0D, 0x1C),
        KeyCode::Backspace => (0x08, 0x0E),

        // Navigation
        KeyCode::Insert => (0x2D, 0x52),
        KeyCode::Delete => (0x2E, 0x53),
        KeyCode::Home => (0x24, 0x47),
        KeyCode::End => (0x23, 0x4F),
        KeyCode::PageUp => (0x21, 0x49),
        KeyCode::PageDown => (0x22, 0x51),

        // Arrow keys
        KeyCode::ArrowUp => (0x26, 0x48),
        KeyCode::ArrowDown => (0x28, 0x50),
        KeyCode::ArrowLeft => (0x25, 0x4B),
        KeyCode::ArrowRight => (0x27, 0x4D),

        // Numpad
        KeyCode::NumLock => (0x90, 0x45),
        KeyCode::Numpad0 => (0x60, 0x52),
        KeyCode::Numpad1 => (0x61, 0x4F),
        KeyCode::Numpad2 => (0x62, 0x50),
        KeyCode::Numpad3 => (0x63, 0x51),
        KeyCode::Numpad4 => (0x64, 0x4B),
        KeyCode::Numpad5 => (0x65, 0x4C),
        KeyCode::Numpad6 => (0x66, 0x4D),
        KeyCode::Numpad7 => (0x67, 0x47),
        KeyCode::Numpad8 => (0x68, 0x48),
        KeyCode::Numpad9 => (0x69, 0x49),
        KeyCode::NumpadAdd => (0x6B, 0x4E),
        KeyCode::NumpadSubtract => (0x6D, 0x4A),
        KeyCode::NumpadMultiply => (0x6A, 0x37),
        KeyCode::NumpadDivide => (0x6F, 0x35),
        KeyCode::NumpadDecimal => (0x6E, 0x53),
        KeyCode::NumpadEnter => (0x0D, 0x1C),

        // Punctuation
        KeyCode::Minus => (0xBD, 0x0C),
        KeyCode::Equal => (0xBB, 0x0D),
        KeyCode::BracketLeft => (0xDB, 0x1A),
        KeyCode::BracketRight => (0xDD, 0x1B),
        KeyCode::Backslash => (0xDC, 0x2B),
        KeyCode::Semicolon => (0xBA, 0x27),
        KeyCode::Quote => (0xDE, 0x28),
        KeyCode::Backquote => (0xC0, 0x29),
        KeyCode::Comma => (0xBC, 0x33),
        KeyCode::Period => (0xBE, 0x34),
        KeyCode::Slash => (0xBF, 0x35),

        // Other
        KeyCode::PrintScreen => (0x2C, 0x37),
        KeyCode::ScrollLock => (0x91, 0x46),
        KeyCode::Pause => (0x13, 0x45),

        _ => (0, 0),
    }
}
