// CECS490 Senior Project - Hoop IQ API Backend
// Christopher Hong, Gondra Kelly, Matthew Marguiles, Alfonso Mejia Vasquez, Carloz Orozco
// This backend provides REST API endpoints for web and mobile integration

use opencv::{
    core::{self, Mat, Point, Scalar, Size, Vector, BORDER_DEFAULT},
    imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use serde_json::json;
use std::io::prelude::*;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
struct Detection {
    id: u64,
    x: i32,
    y: i32,
    radius: i32,
    timestamp: u64,
    confidence: f32,
}

#[derive(Clone)]
struct GameState {
    frame_count: u64,
    detection_count: u64,
    fps: f32,
    is_running: bool,
    current_detections: Vec<Detection>,
    detection_history: Vec<Detection>, // Last 100 detections
    session_start: u64,
    last_frame_jpeg: Vec<u8>,
}

impl GameState {
    fn new() -> Self {
        Self {
            frame_count: 0,
            detection_count: 0,
            fps: 0.0,
            is_running: true,
            current_detections: Vec::new(),
            detection_history: Vec::new(),
            session_start: get_timestamp(),
            last_frame_jpeg: Vec::new(),
        }
    }

    fn add_detection(&mut self, detection: Detection) {
        self.detection_history.push(detection.clone());
        if self.detection_history.len() > 100 {
            self.detection_history.remove(0);
        }
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏀 HOOP IQ");
    println!("CECS490 Senior Project\n");

    let game_state = Arc::new(Mutex::new(GameState::new()));

    // Start API server in separate thread
    let api_state = Arc::clone(&game_state);
    thread::spawn(move || {
        start_api_server(api_state);
    });

    thread::sleep(Duration::from_millis(500));

    let local_ip = get_local_ip();
    println!("🌐 API Server running at: http://{}:8080", local_ip);
    println!("\n📡 API Endpoints:");
    println!("  GET  /api/status        - Current detection status");
    println!("  GET  /api/detections    - Recent detections");
    println!("  GET  /api/stats         - Session statistics");
    println!("  GET  /api/stream        - MJPEG video stream");
    println!("  GET  /api/snapshot      - Single frame snapshot");
    println!("  POST /api/reset         - Reset statistics");
    println!("\n📱 For mobile app/website integration, use these endpoints!");
    println!("\n📹 Starting camera...\n");

    // Open camera
    let mut cam = match videoio::VideoCapture::new(0, videoio::CAP_V4L2) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("⚠️  V4L2 failed, trying default: {}", e);
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?
        }
    };

    if !videoio::VideoCapture::is_opened(&cam)? {
        eprintln!("❌ Could not open camera");
        return Ok(());
    }

    println!("✅ Camera opened!");

    // Normal Resolution - 30FPS
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
    cam.set(videoio::CAP_PROP_FPS, 60.0)?;
    let actual_fps = cam.get(videoio::CAP_PROP_FPS)?;
    println!(
        "📹 Requested 60 FPS at 640x480, camera reports: {}",
        actual_fps
    );

    // Lower Resolution - 60FPS
    /*cam.set(videoio::CAP_PROP_FRAME_WIDTH, 320.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 240.0)?;
    cam.set(videoio::CAP_PROP_FPS, 60.0)?;*/

    let mut frame = Mat::default();
    let mut hsv = Mat::default();
    let mut mask_orange = Mat::default();
    let mut mask_black = Mat::default();
    let mut combined_mask = Mat::default();
    let mut temp_mask = Mat::default();
    let mut blurred = Mat::default();
    let mut result = Mat::default();

    let mut last_time = Instant::now();
    let mut frame_times = Vec::new();
    let mut detection_id: u64 = 0;

    println!("🎯 Detection loop started!\n");

    loop {
        // Calculate FPS
        let now = Instant::now();
        let frame_time = now.duration_since(last_time).as_secs_f32();
        frame_times.push(frame_time);
        if frame_times.len() > 30 {
            frame_times.remove(0);
        }
        let avg_fps = if frame_times.len() > 0 {
            frame_times.len() as f32 / frame_times.iter().sum::<f32>()
        } else {
            0.0
        };
        last_time = now;

        if let Err(_) = cam.read(&mut frame) {
            continue;
        }

        if frame.empty() {
            continue;
        }

        // Convert to HSV
        imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        // Basketball color ranges
        let lower_orange = Scalar::new(5.0, 100.0, 100.0, 0.0);
        let upper_orange = Scalar::new(25.0, 255.0, 255.0, 0.0);
        let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
        let upper_black = Scalar::new(180.0, 255.0, 60.0, 0.0);

        // Create and combine masks
        core::in_range(&hsv, &lower_orange, &upper_orange, &mut mask_orange)?;
        core::in_range(&hsv, &lower_black, &upper_black, &mut mask_black)?;
        core::bitwise_or(
            &mask_orange,
            &mask_black,
            &mut combined_mask,
            &core::no_array(),
        )?;

        // Morphological operations
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            Size::new(5, 5),
            Point::new(-1, -1),
        )?;

        imgproc::morphology_ex(
            &combined_mask,
            &mut temp_mask,
            imgproc::MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            2,
            BORDER_DEFAULT,
            core::Scalar::default(),
        )?;

        imgproc::morphology_ex(
            &temp_mask,
            &mut combined_mask,
            imgproc::MORPH_OPEN,
            &kernel,
            Point::new(-1, -1),
            2,
            BORDER_DEFAULT,
            core::Scalar::default(),
        )?;

        // Blur for better detection
        imgproc::gaussian_blur(
            &combined_mask,
            &mut blurred,
            Size::new(9, 9),
            2.0,
            2.0,
            BORDER_DEFAULT,
        )?;

        // Detect circles
        let mut circles = Vector::<core::Vec3f>::new();
        imgproc::hough_circles(
            &blurred,
            &mut circles,
            imgproc::HOUGH_GRADIENT,
            1.0,
            50.0,
            100.0,
            30.0,
            20,
            200,
        )?;

        // Draw results
        frame.copy_to(&mut result)?;
        let num_detections = circles.len();

        let mut current_detections = Vec::new();
        let timestamp = get_timestamp();

        for i in 0..num_detections {
            let circle = circles.get(i)?;
            let center = Point::new(circle[0] as i32, circle[1] as i32);
            let radius = circle[2] as i32;

            detection_id += 1;
            let detection = Detection {
                id: detection_id,
                x: center.x,
                y: center.y,
                radius,
                timestamp,
                confidence: 0.85, // Could be calculated based on circle quality
            };
            current_detections.push(detection);

            // Draw on frame
            imgproc::circle(
                &mut result,
                center,
                radius,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                3,
                imgproc::LINE_AA,
                0,
            )?;

            imgproc::circle(
                &mut result,
                center,
                5,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;

            let text = format!("Ball #{}", detection_id);
            imgproc::put_text(
                &mut result,
                &text,
                Point::new(center.x - 40, center.y - radius - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        // Add overlay info
        let info_text = format!("FPS: {:.1} | Detections: {}", avg_fps, num_detections);
        imgproc::put_text(
            &mut result,
            &info_text,
            Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        imgproc::put_text(
            &mut result,
            "HOOP IQ API",
            Point::new(10, 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(100.0, 200.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        // Encode frame as JPEG
        let mut buf = Vector::new();
        let params = Vector::new();
        imgcodecs::imencode(".jpg", &result, &mut buf, &params)?;
        let jpeg_data = buf.to_vec();

        // Update game state
        let mut state = game_state.lock().unwrap();
        state.frame_count += 1;
        if num_detections > 0 {
            state.detection_count += 1;
            for det in &current_detections {
                state.add_detection(det.clone());
            }
        }
        state.fps = avg_fps;
        state.current_detections = current_detections;
        state.last_frame_jpeg = jpeg_data;

        // Log periodically
        if state.frame_count % 100 == 0 {
            println!(
                "📊 Frame {} | Detections: {} | FPS: {:.1}",
                state.frame_count, num_detections, avg_fps
            );
        }
        drop(state);

        thread::sleep(Duration::from_millis(1));
    }
}

fn start_api_server(state: Arc<Mutex<GameState>>) {
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind to port 8080");
    println!("🌐 API server listening on port 8080");

    for stream in listener.incoming() {
        if let Ok(stream) = stream {
            let state_clone = Arc::clone(&state);
            thread::spawn(move || {
                handle_api_request(stream, state_clone);
            });
        }
    }
}

fn handle_api_request(mut stream: TcpStream, state: Arc<Mutex<GameState>>) {
    let mut buffer = [0; 2048];
    if stream.read(&mut buffer).is_err() {
        return;
    }

    let request = String::from_utf8_lossy(&buffer[..]);
    let request_line = request.lines().next().unwrap_or("");

    // CORS headers for web integration
    let cors_headers = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n";

    if request_line.starts_with("OPTIONS") {
        let response = format!("HTTP/1.1 200 OK\r\n{}\r\n", cors_headers);
        let _ = stream.write(response.as_bytes());
        return;
    }

    if request_line.starts_with("GET /api/status") {
        send_status(&mut stream, state, cors_headers);
    } else if request_line.starts_with("GET /api/detections") {
        send_detections(&mut stream, state, cors_headers);
    } else if request_line.starts_with("GET /api/stats") {
        send_stats(&mut stream, state, cors_headers);
    } else if request_line.starts_with("GET /api/stream") {
        send_mjpeg_stream(&mut stream, state);
    } else if request_line.starts_with("GET /api/snapshot") {
        send_snapshot(&mut stream, state, cors_headers);
    } else if request_line.starts_with("POST /api/reset") {
        reset_stats(&mut stream, state, cors_headers);
    } else if request_line.starts_with("GET /") {
        send_api_docs(&mut stream);
    } else {
        let response = format!("HTTP/1.1 404 NOT FOUND\r\n{}\r\n", cors_headers);
        let _ = stream.write(response.as_bytes());
    }
}

fn send_status(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    let state_guard = state.lock().unwrap();

    let json = json!({
        "status": "running",
        "frame_count": state_guard.frame_count,
        "fps": format!("{:.1}", state_guard.fps),
        "current_detections": state_guard.current_detections.len(),
        "total_detections": state_guard.detection_count,
        "session_duration": get_timestamp() - state_guard.session_start,
        "detections": state_guard.current_detections.iter().map(|d| {
            json!({
                "id": d.id,
                "x": d.x,
                "y": d.y,
                "radius": d.radius,
                "timestamp": d.timestamp,
                "confidence": d.confidence
            })
        }).collect::<Vec<_>>()
    });

    let body = json.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        cors,
        body.len(),
        body
    );
    let _ = stream.write(response.as_bytes());
}

fn send_detections(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    let state_guard = state.lock().unwrap();

    let json = json!({
        "total": state_guard.detection_history.len(),
        "detections": state_guard.detection_history.iter().map(|d| {
            json!({
                "id": d.id,
                "x": d.x,
                "y": d.y,
                "radius": d.radius,
                "timestamp": d.timestamp,
                "confidence": d.confidence
            })
        }).collect::<Vec<_>>()
    });

    let body = json.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        cors,
        body.len(),
        body
    );
    let _ = stream.write(response.as_bytes());
}

fn send_stats(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    let state_guard = state.lock().unwrap();

    let detection_rate = if state_guard.frame_count > 0 {
        (state_guard.detection_count as f64 / state_guard.frame_count as f64) * 100.0
    } else {
        0.0
    };

    let json = json!({
        "session_start": state_guard.session_start,
        "session_duration": get_timestamp() - state_guard.session_start,
        "frame_count": state_guard.frame_count,
        "detection_count": state_guard.detection_count,
        "detection_rate": format!("{:.2}", detection_rate),
        "fps": format!("{:.1}", state_guard.fps),
        "is_running": state_guard.is_running
    });

    let body = json.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        cors,
        body.len(),
        body
    );
    let _ = stream.write(response.as_bytes());
}

fn send_snapshot(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    let jpeg_data = {
        let state_guard = state.lock().unwrap();
        state_guard.last_frame_jpeg.clone()
    };

    if !jpeg_data.is_empty() {
        let response = format!(
            "HTTP/1.1 200 OK\r\n{}Content-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            cors,
            jpeg_data.len()
        );
        let _ = stream.write(response.as_bytes());
        let _ = stream.write(&jpeg_data);
    } else {
        let response = format!("HTTP/1.1 503 SERVICE UNAVAILABLE\r\n{}\r\n", cors);
        let _ = stream.write(response.as_bytes());
    }
}

fn send_mjpeg_stream(stream: &mut TcpStream, state: Arc<Mutex<GameState>>) {
    let header =
        "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    let _ = stream.write(header.as_bytes());

    loop {
        let jpeg_data = {
            let state_guard = state.lock().unwrap();
            state_guard.last_frame_jpeg.clone()
        };

        if !jpeg_data.is_empty() {
            let part = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg_data.len()
            );

            if stream.write(part.as_bytes()).is_err() {
                break;
            }
            if stream.write(&jpeg_data).is_err() {
                break;
            }
            if stream.write(b"\r\n").is_err() {
                break;
            }
        }

        thread::sleep(Duration::from_millis(33)); // ~30 FPS
    }
}

fn reset_stats(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    let mut state_guard = state.lock().unwrap();
    state_guard.frame_count = 0;
    state_guard.detection_count = 0;
    state_guard.detection_history.clear();
    state_guard.session_start = get_timestamp();

    let json = json!({"status": "reset", "message": "Statistics reset successfully"});
    let body = json.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n{}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        cors,
        body.len(),
        body
    );
    let _ = stream.write(response.as_bytes());
}

fn send_api_docs(stream: &mut TcpStream) {
    let html = r#"<!DOCTYPE html>
<html><head><title>HOOP IQ API</title><style>
body{font-family:Arial;max-width:800px;margin:50px auto;padding:20px;background:#f5f5f5;}
h1{color:#333;border-bottom:3px solid #4CAF50;padding-bottom:10px;}
.endpoint{background:white;padding:15px;margin:10px 0;border-left:4px solid #4CAF50;border-radius:4px;}
.method{display:inline-block;padding:4px 8px;border-radius:3px;font-weight:bold;margin-right:10px;}
.get{background:#61affe;color:white;}.post{background:#49cc90;color:white;}
code{background:#eee;padding:2px 6px;border-radius:3px;}
</style></head><body>
<h1>🏀 HOOP IQ API Documentation</h1>
<p>Real-time Basketball Detection API for Web & Mobile Integration</p>
<div class="endpoint">
<span class="method get">GET</span><code>/api/status</code>
<p>Get current detection status and active detections</p>
</div>
<div class="endpoint">
<span class="method get">GET</span><code>/api/detections</code>
<p>Get recent detection history (last 100)</p>
</div>
<div class="endpoint">
<span class="method get">GET</span><code>/api/stats</code>
<p>Get session statistics</p>
</div>
<div class="endpoint">
<span class="method get">GET</span><code>/api/stream</code>
<p>MJPEG video stream with detections</p>
</div>
<div class="endpoint">
<span class="method get">GET</span><code>/api/snapshot</code>
<p>Get single frame snapshot (JPEG)</p>
</div>
<div class="endpoint">
<span class="method post">POST</span><code>/api/reset</code>
<p>Reset statistics counters</p>
</div>
<p style="margin-top:30px;color:#666;">CECS490 Senior Project | Team: Hong, Kelly, Marguiles, Mejia Vasquez, Orozco</p>
</body></html>"#;

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
        html.len(),
        html
    );
    let _ = stream.write(response.as_bytes());
}

fn get_local_ip() -> String {
    use std::process::Command;
    if let Ok(output) = Command::new("hostname").arg("-I").output() {
        if let Ok(ip) = String::from_utf8(output.stdout) {
            if let Some(first_ip) = ip.split_whitespace().next() {
                return first_ip.to_string();
            }
        }
    }
    "localhost".to_string()
}
