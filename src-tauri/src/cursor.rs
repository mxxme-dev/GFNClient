//! Native cursor/mouse capture for macOS and Windows
//! Uses Core Graphics APIs (macOS) or Win32 APIs (Windows) to properly capture mouse input

use tauri::command;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

static CURSOR_CAPTURED: AtomicBool = AtomicBool::new(false);

// High-frequency mouse polling state
static MOUSE_POLLING_ACTIVE: AtomicBool = AtomicBool::new(false);
static ACCUMULATED_DX: AtomicI32 = AtomicI32::new(0);
static ACCUMULATED_DY: AtomicI32 = AtomicI32::new(0);

#[cfg(target_os = "macos")]
mod macos {
    use core_graphics::display::{CGDisplay, CGPoint};
    use core_graphics::event::{CGEvent, CGEventType};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    use std::sync::atomic::{AtomicI32, Ordering};

    // Store center position for delta calculation
    pub static CENTER_X: AtomicI32 = AtomicI32::new(0);
    pub static CENTER_Y: AtomicI32 = AtomicI32::new(0);

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGAssociateMouseAndMouseCursorPosition(connected: bool) -> i32;
        fn CGDisplayHideCursor(display: u32) -> i32;
        fn CGDisplayShowCursor(display: u32) -> i32;
        fn CGWarpMouseCursorPosition(point: CGPoint) -> i32;
    }

    /// Disassociate mouse from cursor position (allows unlimited movement)
    pub fn set_mouse_cursor_association(associated: bool) -> bool {
        unsafe {
            CGAssociateMouseAndMouseCursorPosition(associated) == 0
        }
    }

    /// Hide the cursor on the main display
    pub fn hide_cursor() -> bool {
        unsafe {
            CGDisplayHideCursor(CGDisplay::main().id) == 0
        }
    }

    /// Show the cursor on the main display
    pub fn show_cursor() -> bool {
        unsafe {
            CGDisplayShowCursor(CGDisplay::main().id) == 0
        }
    }

    /// Get display center and store it
    pub fn update_center() -> bool {
        let display = CGDisplay::main();
        let bounds = display.bounds();
        let cx = (bounds.origin.x + bounds.size.width / 2.0) as i32;
        let cy = (bounds.origin.y + bounds.size.height / 2.0) as i32;
        CENTER_X.store(cx, Ordering::SeqCst);
        CENTER_Y.store(cy, Ordering::SeqCst);
        true
    }

    /// Get stored center position
    pub fn get_stored_center() -> (i32, i32) {
        (CENTER_X.load(Ordering::SeqCst), CENTER_Y.load(Ordering::SeqCst))
    }

    /// Warp cursor to center of main display
    pub fn center_cursor() -> bool {
        let display = CGDisplay::main();
        let bounds = display.bounds();
        let center = CGPoint::new(
            bounds.origin.x + bounds.size.width / 2.0,
            bounds.origin.y + bounds.size.height / 2.0,
        );
        unsafe {
            CGWarpMouseCursorPosition(center) == 0
        }
    }

    /// Warp cursor to a specific position
    pub fn warp_cursor(x: f64, y: f64) -> bool {
        let point = CGPoint::new(x, y);
        unsafe {
            CGWarpMouseCursorPosition(point) == 0
        }
    }

    /// Get current mouse position using CGEvent
    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        // Create a null event to query mouse location
        if let Ok(source) = CGEventSource::new(CGEventSourceStateID::HIDSystemState) {
            if let Ok(event) = CGEvent::new(source) {
                let loc = event.location();
                return Some((loc.x as i32, loc.y as i32));
            }
        }
        None
    }

    /// Get mouse delta from center and recenter cursor
    /// macOS version using CGEvent for position query
    pub fn get_delta_and_recenter() -> (i32, i32) {
        let (cx, cy) = get_stored_center();
        if cx == 0 && cy == 0 {
            return (0, 0);
        }

        if let Some((x, y)) = get_cursor_pos() {
            let dx = x - cx;
            let dy = y - cy;

            // Only recenter if there was movement
            if dx != 0 || dy != 0 {
                warp_cursor(cx as f64, cy as f64);
            }

            (dx, dy)
        } else {
            (0, 0)
        }
    }
}

// Linux X11 support for cursor capture
#[cfg(target_os = "linux")]
mod linux {
    use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};
    use std::ptr::null_mut;
    use std::ffi::c_void;

    // Store center position for delta calculation
    pub static CENTER_X: AtomicI32 = AtomicI32::new(0);
    pub static CENTER_Y: AtomicI32 = AtomicI32::new(0);
    // Store X11 display pointer
    static DISPLAY: AtomicPtr<c_void> = AtomicPtr::new(null_mut());
    static ROOT_WINDOW: AtomicI32 = AtomicI32::new(0);

    // X11 types
    type Display = *mut c_void;
    type Window = u64;
    type Bool = i32;

    #[repr(C)]
    struct XEvent {
        _data: [u8; 192], // XEvent is a union, we just need the size
    }

    #[link(name = "X11")]
    extern "C" {
        fn XOpenDisplay(display_name: *const i8) -> Display;
        fn XCloseDisplay(display: Display) -> i32;
        fn XDefaultRootWindow(display: Display) -> Window;
        fn XQueryPointer(
            display: Display,
            w: Window,
            root_return: *mut Window,
            child_return: *mut Window,
            root_x_return: *mut i32,
            root_y_return: *mut i32,
            win_x_return: *mut i32,
            win_y_return: *mut i32,
            mask_return: *mut u32,
        ) -> Bool;
        fn XWarpPointer(
            display: Display,
            src_w: Window,
            dest_w: Window,
            src_x: i32,
            src_y: i32,
            src_width: u32,
            src_height: u32,
            dest_x: i32,
            dest_y: i32,
        ) -> i32;
        fn XFlush(display: Display) -> i32;
        fn XGrabPointer(
            display: Display,
            grab_window: Window,
            owner_events: Bool,
            event_mask: u32,
            pointer_mode: i32,
            keyboard_mode: i32,
            confine_to: Window,
            cursor: u64,
            time: u64,
        ) -> i32;
        fn XUngrabPointer(display: Display, time: u64) -> i32;
        fn XDisplayWidth(display: Display, screen: i32) -> i32;
        fn XDisplayHeight(display: Display, screen: i32) -> i32;
        fn XDefaultScreen(display: Display) -> i32;
    }

    const GrabModeAsync: i32 = 1;
    const ButtonPressMask: u32 = 1 << 2;
    const ButtonReleaseMask: u32 = 1 << 3;
    const PointerMotionMask: u32 = 1 << 6;
    const CurrentTime: u64 = 0;
    const None: u64 = 0;

    /// Initialize X11 display connection
    pub fn init_display() -> bool {
        unsafe {
            if !DISPLAY.load(Ordering::SeqCst).is_null() {
                return true; // Already initialized
            }

            let display = XOpenDisplay(null_mut());
            if display.is_null() {
                log::error!("Failed to open X11 display");
                return false;
            }

            let root = XDefaultRootWindow(display);
            DISPLAY.store(display, Ordering::SeqCst);
            ROOT_WINDOW.store(root as i32, Ordering::SeqCst);
            log::info!("X11 display initialized");
            true
        }
    }

    /// Close X11 display connection
    pub fn close_display() {
        unsafe {
            let display = DISPLAY.swap(null_mut(), Ordering::SeqCst);
            if !display.is_null() {
                XCloseDisplay(display);
            }
        }
    }

    /// Get display pointer
    fn get_display() -> Option<Display> {
        let display = DISPLAY.load(Ordering::SeqCst);
        if display.is_null() {
            None
        } else {
            Some(display)
        }
    }

    /// Get root window
    fn get_root() -> Window {
        ROOT_WINDOW.load(Ordering::SeqCst) as Window
    }

    /// Update center position (screen center)
    pub fn update_center() -> bool {
        unsafe {
            let display = match get_display() {
                Some(d) => d,
                None => {
                    if !init_display() {
                        return false;
                    }
                    get_display().unwrap()
                }
            };

            let screen = XDefaultScreen(display);
            let width = XDisplayWidth(display, screen);
            let height = XDisplayHeight(display, screen);

            let cx = width / 2;
            let cy = height / 2;
            CENTER_X.store(cx, Ordering::SeqCst);
            CENTER_Y.store(cy, Ordering::SeqCst);
            true
        }
    }

    /// Get stored center position
    pub fn get_stored_center() -> (i32, i32) {
        (CENTER_X.load(Ordering::SeqCst), CENTER_Y.load(Ordering::SeqCst))
    }

    /// Hide cursor by grabbing pointer with invisible cursor
    pub fn hide_cursor() -> bool {
        unsafe {
            let display = match get_display() {
                Some(d) => d,
                None => return false,
            };
            let root = get_root();

            // Grab pointer with no cursor (None = invisible)
            let result = XGrabPointer(
                display,
                root,
                1, // owner_events = True
                ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
                GrabModeAsync,
                GrabModeAsync,
                root, // confine to root window
                None, // invisible cursor
                CurrentTime,
            );

            XFlush(display);
            result == 0 // GrabSuccess = 0
        }
    }

    /// Show cursor by ungrabbing pointer
    pub fn show_cursor() -> bool {
        unsafe {
            let display = match get_display() {
                Some(d) => d,
                None => return false,
            };

            XUngrabPointer(display, CurrentTime);
            XFlush(display);
            true
        }
    }

    /// Get current cursor position
    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        unsafe {
            let display = get_display()?;
            let root = get_root();

            let mut root_return: Window = 0;
            let mut child_return: Window = 0;
            let mut root_x: i32 = 0;
            let mut root_y: i32 = 0;
            let mut win_x: i32 = 0;
            let mut win_y: i32 = 0;
            let mut mask: u32 = 0;

            if XQueryPointer(
                display,
                root,
                &mut root_return,
                &mut child_return,
                &mut root_x,
                &mut root_y,
                &mut win_x,
                &mut win_y,
                &mut mask,
            ) != 0
            {
                Some((root_x, root_y))
            } else {
                None
            }
        }
    }

    /// Set cursor position
    pub fn set_cursor_pos(x: i32, y: i32) -> bool {
        unsafe {
            let display = match get_display() {
                Some(d) => d,
                None => return false,
            };
            let root = get_root();

            XWarpPointer(display, 0, root, 0, 0, 0, 0, x, y);
            XFlush(display);
            true
        }
    }

    /// Center cursor on screen
    pub fn center_cursor() -> bool {
        let (cx, cy) = get_stored_center();
        if cx != 0 && cy != 0 {
            set_cursor_pos(cx, cy)
        } else if update_center() {
            let (cx, cy) = get_stored_center();
            set_cursor_pos(cx, cy)
        } else {
            false
        }
    }

    /// Get mouse delta from center and recenter cursor
    pub fn get_delta_and_recenter() -> (i32, i32) {
        let (cx, cy) = get_stored_center();
        if cx == 0 && cy == 0 {
            return (0, 0);
        }

        if let Some((x, y)) = get_cursor_pos() {
            let dx = x - cx;
            let dy = y - cy;

            // Only recenter if there was movement
            if dx != 0 || dy != 0 {
                set_cursor_pos(cx, cy);
            }

            (dx, dy)
        } else {
            (0, 0)
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use std::ptr::null_mut;
    use std::mem::zeroed;
    use std::sync::atomic::{AtomicI32, AtomicIsize, AtomicBool, Ordering};

    // Store window center for recentering
    pub static CENTER_X: AtomicI32 = AtomicI32::new(0);
    pub static CENTER_Y: AtomicI32 = AtomicI32::new(0);
    // Store the original cursor to restore later
    pub static ORIGINAL_CURSOR: AtomicIsize = AtomicIsize::new(0);
    // Store original mouse acceleration settings
    pub static ACCEL_DISABLED: AtomicBool = AtomicBool::new(false);
    static mut ORIGINAL_MOUSE_PARAMS: [i32; 3] = [0, 0, 0];

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct POINT {
        x: i32,
        y: i32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct RECT {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    type HWND = *mut std::ffi::c_void;
    type HCURSOR = *mut std::ffi::c_void;
    type LONG_PTR = isize;

    const GCLP_HCURSOR: i32 = -12;
    const IDC_ARROW: *const u16 = 32512 as *const u16;

    // SystemParametersInfo constants for mouse acceleration
    const SPI_GETMOUSE: u32 = 0x0003;
    const SPI_SETMOUSE: u32 = 0x0004;
    const SPIF_SENDCHANGE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetCursorPos(lpPoint: *mut POINT) -> i32;
        fn SetCursorPos(x: i32, y: i32) -> i32;
        fn ShowCursor(bShow: i32) -> i32;
        fn ClipCursor(lpRect: *const RECT) -> i32;
        fn GetForegroundWindow() -> HWND;
        fn GetWindowRect(hWnd: HWND, lpRect: *mut RECT) -> i32;
        fn SetCursor(hCursor: HCURSOR) -> HCURSOR;
        fn GetClientRect(hWnd: HWND, lpRect: *mut RECT) -> i32;
        fn ClientToScreen(hWnd: HWND, lpPoint: *mut POINT) -> i32;
        fn GetClassLongPtrW(hWnd: HWND, nIndex: i32) -> LONG_PTR;
        fn SetClassLongPtrW(hWnd: HWND, nIndex: i32, dwNewLong: LONG_PTR) -> LONG_PTR;
        fn LoadCursorW(hInstance: *mut std::ffi::c_void, lpCursorName: *const u16) -> HCURSOR;
        fn SystemParametersInfoW(uiAction: u32, uiParam: u32, pvParam: *mut std::ffi::c_void, fWinIni: u32) -> i32;
    }

    /// Disable Windows mouse acceleration (Enhance pointer precision)
    /// Stores original settings to restore later
    pub fn disable_mouse_acceleration() {
        if ACCEL_DISABLED.load(Ordering::SeqCst) {
            return; // Already disabled
        }

        unsafe {
            // Get current mouse parameters [threshold1, threshold2, acceleration]
            let mut params: [i32; 3] = [0, 0, 0];
            if SystemParametersInfoW(SPI_GETMOUSE, 0, params.as_mut_ptr() as *mut _, 0) != 0 {
                // Save original settings
                ORIGINAL_MOUSE_PARAMS = params;

                // Disable acceleration by setting acceleration to 0
                // params[2] is the acceleration flag (0 = disabled, 1 = enabled)
                if params[2] != 0 {
                    let new_params: [i32; 3] = [0, 0, 0]; // Disable acceleration
                    if SystemParametersInfoW(SPI_SETMOUSE, 0, new_params.as_ptr() as *mut _, SPIF_SENDCHANGE) != 0 {
                        ACCEL_DISABLED.store(true, Ordering::SeqCst);
                        log::info!("Mouse acceleration disabled (was: {:?})", ORIGINAL_MOUSE_PARAMS);
                    }
                } else {
                    log::info!("Mouse acceleration already disabled");
                }
            }
        }
    }

    /// Restore original Windows mouse acceleration settings
    pub fn restore_mouse_acceleration() {
        if !ACCEL_DISABLED.load(Ordering::SeqCst) {
            return; // Not disabled by us
        }

        unsafe {
            if SystemParametersInfoW(SPI_SETMOUSE, 0, ORIGINAL_MOUSE_PARAMS.as_ptr() as *mut _, SPIF_SENDCHANGE) != 0 {
                ACCEL_DISABLED.store(false, Ordering::SeqCst);
                log::info!("Mouse acceleration restored to: {:?}", ORIGINAL_MOUSE_PARAMS);
            }
        }
    }

    /// Hide the cursor completely by setting class cursor to NULL
    pub fn hide_cursor() {
        unsafe {
            let hwnd = GetForegroundWindow();
            if !hwnd.is_null() {
                // Save original cursor
                let original = GetClassLongPtrW(hwnd, GCLP_HCURSOR);
                if original != 0 {
                    ORIGINAL_CURSOR.store(original, Ordering::SeqCst);
                }
                // Set class cursor to NULL - this prevents cursor from flickering back
                SetClassLongPtrW(hwnd, GCLP_HCURSOR, 0);
            }
            // Also set current cursor to NULL
            SetCursor(null_mut());
            // Decrement show counter
            let mut count = ShowCursor(0);
            while count >= 0 {
                count = ShowCursor(0);
            }
        }
    }

    /// Show the cursor by restoring the class cursor
    pub fn show_cursor() {
        unsafe {
            let hwnd = GetForegroundWindow();
            if !hwnd.is_null() {
                // Restore original cursor or use arrow
                let original = ORIGINAL_CURSOR.load(Ordering::SeqCst);
                if original != 0 {
                    SetClassLongPtrW(hwnd, GCLP_HCURSOR, original);
                } else {
                    // Load default arrow cursor
                    let arrow = LoadCursorW(null_mut(), IDC_ARROW);
                    SetClassLongPtrW(hwnd, GCLP_HCURSOR, arrow as LONG_PTR);
                }
            }
            // Increment counter until visible
            let mut count = ShowCursor(1);
            while count < 0 {
                count = ShowCursor(1);
            }
        }
    }

    /// Clip cursor to the foreground window
    pub fn clip_cursor_to_window() -> bool {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() {
                return false;
            }
            let mut rect: RECT = zeroed();
            if GetWindowRect(hwnd, &mut rect) == 0 {
                return false;
            }
            ClipCursor(&rect) != 0
        }
    }

    /// Release cursor clipping
    pub fn release_clip() -> bool {
        unsafe {
            ClipCursor(null_mut()) != 0
        }
    }

    /// Get current cursor position
    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        unsafe {
            let mut point: POINT = zeroed();
            if GetCursorPos(&mut point) != 0 {
                Some((point.x, point.y))
            } else {
                None
            }
        }
    }

    /// Set cursor position
    pub fn set_cursor_pos(x: i32, y: i32) -> bool {
        unsafe {
            SetCursorPos(x, y) != 0
        }
    }

    /// Get window client area center (screen coordinates)
    pub fn get_window_center() -> Option<(i32, i32)> {
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() {
                return None;
            }
            let mut client_rect: RECT = zeroed();
            if GetClientRect(hwnd, &mut client_rect) == 0 {
                return None;
            }
            // Get center of client area
            let mut center = POINT {
                x: client_rect.right / 2,
                y: client_rect.bottom / 2,
            };
            // Convert to screen coordinates
            if ClientToScreen(hwnd, &mut center) == 0 {
                return None;
            }
            Some((center.x, center.y))
        }
    }

    /// Update stored center position
    pub fn update_center() -> bool {
        if let Some((x, y)) = get_window_center() {
            CENTER_X.store(x, Ordering::SeqCst);
            CENTER_Y.store(y, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Get stored center position
    pub fn get_stored_center() -> (i32, i32) {
        (CENTER_X.load(Ordering::SeqCst), CENTER_Y.load(Ordering::SeqCst))
    }

    /// Center cursor in window
    pub fn center_cursor() -> bool {
        let (cx, cy) = get_stored_center();
        if cx != 0 && cy != 0 {
            set_cursor_pos(cx, cy)
        } else if let Some((x, y)) = get_window_center() {
            CENTER_X.store(x, Ordering::SeqCst);
            CENTER_Y.store(y, Ordering::SeqCst);
            set_cursor_pos(x, y)
        } else {
            false
        }
    }

    /// Get mouse delta from center and recenter cursor
    /// Returns (dx, dy) - the movement since last center
    pub fn get_delta_and_recenter() -> (i32, i32) {
        let (cx, cy) = get_stored_center();
        if cx == 0 && cy == 0 {
            return (0, 0);
        }

        if let Some((x, y)) = get_cursor_pos() {
            let dx = x - cx;
            let dy = y - cy;

            // Only recenter if there was movement
            if dx != 0 || dy != 0 {
                set_cursor_pos(cx, cy);
                // Hide cursor again after repositioning
                unsafe { SetCursor(null_mut()); }
            }

            (dx, dy)
        } else {
            (0, 0)
        }
    }
}

/// Capture the mouse cursor (hide cursor and allow unlimited movement)
/// Uses native OS APIs: Core Graphics on macOS, Win32 on Windows
#[command]
pub async fn capture_cursor() -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        if CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already captured
        }

        // First, center the cursor
        macos::center_cursor();

        // Hide the cursor
        if !macos::hide_cursor() {
            return Err("Failed to hide cursor".to_string());
        }

        // Disassociate mouse from cursor position (this is the key!)
        // This allows the mouse to move infinitely without hitting screen edges
        if !macos::set_mouse_cursor_association(false) {
            macos::show_cursor(); // Restore cursor on failure
            return Err("Failed to disassociate mouse from cursor".to_string());
        }

        CURSOR_CAPTURED.store(true, Ordering::SeqCst);
        log::info!("Cursor captured (macOS native)");
        Ok(true)
    }

    #[cfg(target_os = "windows")]
    {
        if CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already captured
        }

        // Update and store window center
        if !windows::update_center() {
            return Err("Failed to get window center".to_string());
        }

        // Disable mouse acceleration for 1:1 raw input
        windows::disable_mouse_acceleration();

        // Center the cursor
        windows::center_cursor();

        // Hide the cursor
        windows::hide_cursor();

        // Clip cursor to window to prevent it from going to other monitors
        windows::clip_cursor_to_window();

        CURSOR_CAPTURED.store(true, Ordering::SeqCst);
        log::info!("Cursor captured (Windows native with recentering, acceleration disabled)");
        Ok(true)
    }

    #[cfg(target_os = "linux")]
    {
        if CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already captured
        }

        // Initialize X11 display if needed
        if !linux::init_display() {
            return Err("Failed to initialize X11 display".to_string());
        }

        // Update and store screen center
        if !linux::update_center() {
            return Err("Failed to get screen center".to_string());
        }

        // Center the cursor
        linux::center_cursor();

        // Hide cursor by grabbing pointer
        if !linux::hide_cursor() {
            return Err("Failed to grab pointer".to_string());
        }

        CURSOR_CAPTURED.store(true, Ordering::SeqCst);
        log::info!("Cursor captured (Linux X11 native)");
        Ok(true)
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        // On other platforms, return false to indicate native capture not available
        Ok(false)
    }
}

/// Release the mouse cursor (show cursor and restore normal behavior)
#[command]
pub async fn release_cursor() -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already released
        }

        // Re-associate mouse with cursor position
        macos::set_mouse_cursor_association(true);

        // Show the cursor
        macos::show_cursor();

        // Center cursor so it appears in a reasonable position
        macos::center_cursor();

        CURSOR_CAPTURED.store(false, Ordering::SeqCst);
        log::info!("Cursor released (macOS native)");
        Ok(true)
    }

    #[cfg(target_os = "windows")]
    {
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already released
        }

        // Restore mouse acceleration settings
        windows::restore_mouse_acceleration();

        // Release cursor clipping
        windows::release_clip();

        // Show the cursor
        windows::show_cursor();

        // Center cursor so it appears in a reasonable position
        windows::center_cursor();

        CURSOR_CAPTURED.store(false, Ordering::SeqCst);
        log::info!("Cursor released (Windows native)");
        Ok(true)
    }

    #[cfg(target_os = "linux")]
    {
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return Ok(true); // Already released
        }

        // Show cursor by ungrabbing pointer
        linux::show_cursor();

        // Center cursor so it appears in a reasonable position
        linux::center_cursor();

        CURSOR_CAPTURED.store(false, Ordering::SeqCst);
        log::info!("Cursor released (Linux X11 native)");
        Ok(true)
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        Ok(true)
    }
}

/// Check if cursor is currently captured
#[command]
pub async fn is_cursor_captured() -> Result<bool, String> {
    Ok(CURSOR_CAPTURED.load(Ordering::SeqCst))
}

/// Get mouse delta from center and recenter cursor (Windows, macOS, Linux)
/// Returns (dx, dy) - the movement since cursor was last at center
/// This enables FPS-style infinite mouse movement
#[command]
pub fn get_mouse_delta() -> (i32, i32) {
    if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
        return (0, 0);
    }

    #[cfg(target_os = "windows")]
    {
        windows::get_delta_and_recenter()
    }

    #[cfg(target_os = "macos")]
    {
        macos::get_delta_and_recenter()
    }

    #[cfg(target_os = "linux")]
    {
        linux::get_delta_and_recenter()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        (0, 0)
    }
}

/// Recenter cursor without getting delta (useful after window resize)
#[command]
pub fn recenter_cursor() -> bool {
    if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
        return false;
    }

    #[cfg(target_os = "windows")]
    {
        // Update center position (in case window moved/resized)
        windows::update_center();
        windows::center_cursor()
    }

    #[cfg(target_os = "macos")]
    {
        macos::update_center();
        macos::center_cursor()
    }

    #[cfg(target_os = "linux")]
    {
        linux::update_center();
        linux::center_cursor()
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Start high-frequency mouse polling (Windows, macOS, Linux)
/// Polls at ~1000Hz and accumulates deltas for the frontend to read
#[command]
pub fn start_mouse_polling() -> bool {
    #[cfg(target_os = "windows")]
    {
        if MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) {
            return true; // Already running
        }
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return false; // Need cursor captured first
        }

        MOUSE_POLLING_ACTIVE.store(true, Ordering::SeqCst);
        ACCUMULATED_DX.store(0, Ordering::SeqCst);
        ACCUMULATED_DY.store(0, Ordering::SeqCst);

        // Spawn high-frequency polling thread
        std::thread::spawn(|| {
            use std::time::{Duration, Instant};

            // Poll at ~1000Hz (1ms intervals)
            let poll_interval = Duration::from_micros(1000);

            while MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) &&
                  CURSOR_CAPTURED.load(Ordering::SeqCst) {
                let start = Instant::now();

                // Get delta and recenter
                let (dx, dy) = windows::get_delta_and_recenter();

                // Accumulate deltas
                if dx != 0 {
                    ACCUMULATED_DX.fetch_add(dx, Ordering::SeqCst);
                }
                if dy != 0 {
                    ACCUMULATED_DY.fetch_add(dy, Ordering::SeqCst);
                }

                // Sleep for remaining time in interval
                let elapsed = start.elapsed();
                if elapsed < poll_interval {
                    std::thread::sleep(poll_interval - elapsed);
                }
            }

            MOUSE_POLLING_ACTIVE.store(false, Ordering::SeqCst);
            log::info!("Mouse polling thread stopped");
        });

        log::info!("High-frequency mouse polling started (1000Hz) [Windows]");
        true
    }

    #[cfg(target_os = "macos")]
    {
        if MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) {
            return true; // Already running
        }
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return false; // Need cursor captured first
        }

        // Update center position
        macos::update_center();

        MOUSE_POLLING_ACTIVE.store(true, Ordering::SeqCst);
        ACCUMULATED_DX.store(0, Ordering::SeqCst);
        ACCUMULATED_DY.store(0, Ordering::SeqCst);

        // Spawn high-frequency polling thread for macOS
        std::thread::spawn(|| {
            use std::time::{Duration, Instant};

            // Poll at ~1000Hz (1ms intervals)
            let poll_interval = Duration::from_micros(1000);

            while MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) &&
                  CURSOR_CAPTURED.load(Ordering::SeqCst) {
                let start = Instant::now();

                // Get delta and recenter using macOS APIs
                let (dx, dy) = macos::get_delta_and_recenter();

                // Accumulate deltas
                if dx != 0 {
                    ACCUMULATED_DX.fetch_add(dx, Ordering::SeqCst);
                }
                if dy != 0 {
                    ACCUMULATED_DY.fetch_add(dy, Ordering::SeqCst);
                }

                // Sleep for remaining time in interval
                let elapsed = start.elapsed();
                if elapsed < poll_interval {
                    std::thread::sleep(poll_interval - elapsed);
                }
            }

            MOUSE_POLLING_ACTIVE.store(false, Ordering::SeqCst);
            log::info!("Mouse polling thread stopped");
        });

        log::info!("High-frequency mouse polling started (1000Hz) [macOS]");
        true
    }

    #[cfg(target_os = "linux")]
    {
        if MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) {
            return true; // Already running
        }
        if !CURSOR_CAPTURED.load(Ordering::SeqCst) {
            return false; // Need cursor captured first
        }

        // Update center position
        linux::update_center();

        MOUSE_POLLING_ACTIVE.store(true, Ordering::SeqCst);
        ACCUMULATED_DX.store(0, Ordering::SeqCst);
        ACCUMULATED_DY.store(0, Ordering::SeqCst);

        // Spawn high-frequency polling thread for Linux
        std::thread::spawn(|| {
            use std::time::{Duration, Instant};

            // Poll at ~1000Hz (1ms intervals)
            let poll_interval = Duration::from_micros(1000);

            while MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst) &&
                  CURSOR_CAPTURED.load(Ordering::SeqCst) {
                let start = Instant::now();

                // Get delta and recenter using Linux/X11 APIs
                let (dx, dy) = linux::get_delta_and_recenter();

                // Accumulate deltas
                if dx != 0 {
                    ACCUMULATED_DX.fetch_add(dx, Ordering::SeqCst);
                }
                if dy != 0 {
                    ACCUMULATED_DY.fetch_add(dy, Ordering::SeqCst);
                }

                // Sleep for remaining time in interval
                let elapsed = start.elapsed();
                if elapsed < poll_interval {
                    std::thread::sleep(poll_interval - elapsed);
                }
            }

            MOUSE_POLLING_ACTIVE.store(false, Ordering::SeqCst);
            log::info!("Mouse polling thread stopped");
        });

        log::info!("High-frequency mouse polling started (1000Hz) [Linux]");
        true
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Stop high-frequency mouse polling
#[command]
pub fn stop_mouse_polling() {
    MOUSE_POLLING_ACTIVE.store(false, Ordering::SeqCst);
    ACCUMULATED_DX.store(0, Ordering::SeqCst);
    ACCUMULATED_DY.store(0, Ordering::SeqCst);
}

/// Get accumulated mouse deltas and reset accumulators
/// Returns (dx, dy) accumulated since last call
#[command]
pub fn get_accumulated_mouse_delta() -> (i32, i32) {
    let dx = ACCUMULATED_DX.swap(0, Ordering::SeqCst);
    let dy = ACCUMULATED_DY.swap(0, Ordering::SeqCst);
    (dx, dy)
}

/// Check if mouse polling is active
#[command]
pub fn is_mouse_polling_active() -> bool {
    MOUSE_POLLING_ACTIVE.load(Ordering::SeqCst)
}

/// Check if native input pipeline is available on this platform
#[command]
pub fn is_native_input_available() -> bool {
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
    {
        true
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Get the current platform name for debugging
#[command]
pub fn get_input_platform() -> String {
    #[cfg(target_os = "windows")]
    {
        "windows".to_string()
    }
    #[cfg(target_os = "macos")]
    {
        "macos".to_string()
    }
    #[cfg(target_os = "linux")]
    {
        "linux".to_string()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        "unsupported".to_string()
    }
}
