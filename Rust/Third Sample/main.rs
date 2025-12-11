// CECS490 Senior Project - Hoop IQ
// Team 2
// Christopher Hong, Gondra Kelly, Matthew Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// Optimized Single Camera Version

use opencv::{
    core::{self, Mat, Point, Scalar, Size, Vector, BORDER_DEFAULT},
    imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{prelude::*, BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Detection {
    id: u64,
    camera_id: u8,
    x: i32,
    y: i32,
    radius: i32,
    timestamp: u64,
    confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShotData {
    id: u32,
    timestamp: u64,
    result: String,
    backboard_hits: u32,
    rim_hits: u32,
    shot_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BasketballMake {
    id: u64,
    timestamp: u64,
    confidence: f32,
    x: i32,
    y: i32,
}

struct GameState {
    frame_count: u64,
    fps: f32,
    current_detections: Vec<Detection>,
    detection_history: Vec<Detection>,
    basketball_makes: Vec<BasketballMake>,
    last_frame: Arc<RwLock<Vec<u8>>>,
    make_count: u64,
    shot_data: Vec<ShotData>,
    makes_count: u64,
    misses_count: u64,
}

impl GameState {
    fn new() -> Self {
        Self {
            frame_count: 0,
            fps: 0.0,
            current_detections: Vec::new(),
            detection_history: Vec::new(),
            basketball_makes: Vec::new(),
            last_frame: Arc::new(RwLock::new(Vec::new())),
            make_count: 0,
            shot_data: Vec::new(),
            makes_count: 0,
            misses_count: 0,
        }
    }

    fn add_detection(&mut self, detection: Detection) {
        self.detection_history.push(detection);
        if self.detection_history.len() > 50 {
            self.detection_history.remove(0);
        }
    }

    fn add_basketball_make(&mut self, make: BasketballMake) {
        println!("🎯 MAKE #{} at ({}, {})", make.id, make.x, make.y);
        self.basketball_makes.push(make.clone());
        if self.basketball_makes.len() > 50 {
            self.basketball_makes.remove(0);
        }
        self.make_count += 1;
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏀 HOOP IQ - C922x @ 60 FPS");
    println!("CECS490 Senior Project\n");

    configure_c922x_60fps();

    let game_state = Arc::new(Mutex::new(GameState::new()));
    let frame_lock = Arc::clone(&game_state.lock().unwrap().last_frame);

    let esp32_state = Arc::clone(&game_state);
    thread::spawn(move || {
        listen_to_esp32(esp32_state);
    });

    let api_state = Arc::clone(&game_state);
    let api_frame = Arc::clone(&frame_lock);
    thread::spawn(move || {
        start_api_server(api_state, api_frame);
    });

    let analyzer_state = Arc::clone(&game_state);
    thread::spawn(move || {
        analyze_basketball_makes(analyzer_state);
    });

    thread::sleep(Duration::from_millis(1000));

    let local_ip = get_local_ip();
    println!("🌐 API Server: http://{}:8080", local_ip);
    println!("📡 Stream: http://{}:8080/api/stream\n", local_ip);

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;

    if !videoio::VideoCapture::is_opened(&cam)? {
        eprintln!("❌ Could not open camera");
        return Ok(());
    }

    println!("✅ Camera opened!");

    // ⭐ CRITICAL: Set format BEFORE checking - OpenCV needs this
    let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    cam.set(videoio::CAP_PROP_FOURCC, fourcc as f64)?;
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cam.set(videoio::CAP_PROP_FPS, 60.0)?;
    cam.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    // Small delay to let camera apply settings
    thread::sleep(Duration::from_millis(500));
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cam.set(videoio::CAP_PROP_FPS, 60.0)?;
    cam.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    // Small delay to let camera apply settings
    thread::sleep(Duration::from_millis(500));

    let actual_fps = cam.get(videoio::CAP_PROP_FPS)?;
    let actual_width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let actual_height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?;

    println!(
        "📹 Camera: {}x{} @ {:.1} FPS",
        actual_width, actual_height, actual_fps
    );

    if actual_fps >= 55.0 && actual_width >= 1280.0 {
        println!("🚀 720p @ 60 FPS achieved!\n");
    } else if actual_fps >= 55.0 {
        println!("✅ 60 FPS achieved (but lower resolution)\n");
    } else if actual_fps >= 25.0 {
        println!("⚠️  Running at 30 FPS - check lighting or USB connection");
        println!("💡 Tip: Increase room lighting for 60 FPS\n");
    } else {
        println!("\n⚠️  FPS is low - camera configuration issue");
        println!("💡 Run: v4l2-ctl -d /dev/video0 --list-formats-ext\n");
    }

    let state_cam = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_camera(cam, 1, state_cam) {
            eprintln!("Camera error: {}", e);
        }
    });

    println!("🎯 Detection active!\n");

    loop {
        thread::sleep(Duration::from_secs(5));
        if let Ok(state) = game_state.lock() {
            println!(
                "📊 Frames: {} | Makes: {} | Misses: {} | FPS: {:.1}",
                state.frame_count, state.makes_count, state.misses_count, state.fps
            );
        }
    }
}

fn configure_c922x_60fps() {
    // Renamed function
    println!("⚙️  Configuring C922x for 720p @ 60 FPS...");

    // C922x works best with these settings for 60 FPS
    let output = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-fmt-video=width=1280,height=720,pixelformat=MJPG",
        ])
        .output();

    if let Ok(out) = output {
        if !out.status.success() {
            println!("⚠️  Camera format configuration issue");
        }
    }

    // Set framerate separately (more reliable for C922x)
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-parm=60"])
        .output();

    thread::sleep(Duration::from_millis(500));

    // Disable autofocus (C922x has this)
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=focus_auto=0"])
        .output();

    // Set manual focus to infinity (good for court view)
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=focus_absolute=0"])
        .output();

    // Manual exposure for consistent frame timing
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=exposure_auto=1"])
        .output();

    // Exposure value optimized for 60 FPS (lower = faster shutter)
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=exposure_absolute=156"])
        .output();

    // Disable auto white balance for consistent colors
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-ctrl=white_balance_temperature_auto=0",
        ])
        .output();

    // Set white balance for indoor lighting (adjust if needed)
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-ctrl=white_balance_temperature=4000",
        ])
        .output();

    println!("✅ C922x configured for 720p @ 60 FPS");
}

fn process_camera(
    mut cam: videoio::VideoCapture,
    camera_id: u8,
    state: Arc<Mutex<GameState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame = Mat::default();
    let mut hsv = Mat::default();
    let mut mask_orange = Mat::default();
    let mut combined_mask = Mat::default();
    let mut blurred = Mat::default();

    let mut last_time = Instant::now();
    let mut frame_times = Vec::with_capacity(60);
    let mut detection_id: u64 = camera_id as u64 * 100000;

    // Optimize JPEG encoding for speed (60 FPS needs fast encoding)
    let mut encode_params = Vector::<i32>::new();
    encode_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    encode_params.push(65);
    encode_params.push(imgcodecs::IMWRITE_JPEG_OPTIMIZE);
    encode_params.push(0);

    let mut current_detections = Vec::with_capacity(5);

    // Detect every 2 frames for 60 FPS (30 detections per second)
    let mut frame_counter = 0u64;

    loop {
        let now = Instant::now();
        let frame_time = now.duration_since(last_time).as_secs_f32();
        frame_times.push(frame_time);
        if frame_times.len() > 60 {
            frame_times.remove(0);
        }
        let avg_fps = if !frame_times.is_empty() {
            frame_times.len() as f32 / frame_times.iter().sum::<f32>()
        } else {
            0.0
        };
        last_time = now;

        if cam.read(&mut frame).is_err() || frame.empty() {
            continue;
        }

        frame_counter += 1;
        let timestamp = get_timestamp();
        let should_detect = frame_counter % 2 == 0;

        if should_detect {
            current_detections.clear();

            imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

            let lower_orange = Scalar::new(5.0, 120.0, 120.0, 0.0);
            let upper_orange = Scalar::new(20.0, 255.0, 255.0, 0.0);

            core::in_range(&hsv, &lower_orange, &upper_orange, &mut mask_orange)?;

            let kernel = imgproc::get_structuring_element(
                imgproc::MORPH_ELLIPSE,
                Size::new(3, 3),
                Point::new(-1, -1),
            )?;

            imgproc::morphology_ex(
                &mask_orange,
                &mut combined_mask,
                imgproc::MORPH_CLOSE,
                &kernel,
                Point::new(-1, -1),
                1,
                BORDER_DEFAULT,
                core::Scalar::default(),
            )?;

            imgproc::gaussian_blur(
                &combined_mask,
                &mut blurred,
                Size::new(5, 5),
                1.5,
                1.5,
                BORDER_DEFAULT,
            )?;

            let mut circles = Vector::<core::Vec3f>::new();
            imgproc::hough_circles(
                &blurred,
                &mut circles,
                imgproc::HOUGH_GRADIENT,
                1.0,
                50.0,
                80.0,
                25.0,
                10,
                100,
            )?;

            for i in 0..circles.len() {
                let circle = circles.get(i)?;
                let center = Point::new(circle[0] as i32, circle[1] as i32);
                let radius = circle[2] as i32;

                detection_id += 1;
                current_detections.push(Detection {
                    id: detection_id,
                    camera_id,
                    x: center.x,
                    y: center.y,
                    radius,
                    timestamp,
                    confidence: 0.90,
                });
            }
        }

        // Draw detections
        for det in &current_detections {
            imgproc::circle(
                &mut frame,
                Point::new(det.x, det.y),
                det.radius,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Get camera's actual FPS
        let camera_fps = cam.get(videoio::CAP_PROP_FPS).unwrap_or(60.0);

        // FPS display - show both camera and processing FPS
        imgproc::put_text(
            &mut frame,
            &format!("CAM: {:.0} | PROC: {:.0}", camera_fps, avg_fps),
            Point::new(10, 25),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        // Encode with optimized settings
        let mut buf = Vector::new();
        imgcodecs::imencode(".jpg", &frame, &mut buf, &encode_params)?;
        let jpeg_data = buf.to_vec();

        // Update state (non-blocking)
        if let Ok(mut state_guard) = state.try_lock() {
            state_guard.frame_count += 1;
            state_guard.fps = avg_fps;

            if should_detect {
                state_guard.current_detections = current_detections.clone();
                for det in &current_detections {
                    state_guard.add_detection(det.clone());
                }
            }

            if let Ok(mut frame_guard) = state_guard.last_frame.try_write() {
                *frame_guard = jpeg_data;
            }
        }
    }
}

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    let serial_ports = vec!["/dev/ttyUSB0", "/dev/ttyACM0"];

    for port in &serial_ports {
        if let Ok(file) = std::fs::OpenOptions::new().read(true).open(port) {
            println!("📡 Listening to ESP32 on {}", port);
            let reader = BufReader::new(file);

            for line in reader.lines() {
                if let Ok(data) = line {
                    let data = data.trim();

                    if data.starts_with("RESULT:") {
                        parse_shot_result(&data, &state);
                    } else if data.starts_with("SWISH") {
                        if let Ok(mut state_guard) = state.lock() {
                            state_guard.makes_count += 1;
                        }
                    }
                }
            }
            return;
        }
    }
}

fn parse_shot_result(data: &str, state: &Arc<Mutex<GameState>>) {
    let mut result = String::new();
    let mut shot_id = 0u32;
    let mut timestamp = 0u64;
    let mut backboard_hits = 0u32;
    let mut rim_hits = 0u32;

    for part in data.split(',') {
        let part = part.trim();
        if part.starts_with("RESULT:") {
            result = part.replace("RESULT:", "");
        } else if part.starts_with("ID:") {
            shot_id = part.replace("ID:", "").parse().unwrap_or(0);
        } else if part.starts_with("TIME:") {
            timestamp = part.replace("TIME:", "").parse().unwrap_or(0);
        } else if part.starts_with("BACK:") {
            backboard_hits = part.replace("BACK:", "").parse().unwrap_or(0);
        } else if part.starts_with("RIM:") {
            rim_hits = part.replace("RIM:", "").parse().unwrap_or(0);
        }
    }

    let shot_type = if backboard_hits > 0 && rim_hits > 0 {
        "BOTH"
    } else if backboard_hits > 0 {
        "BACKBOARD"
    } else if rim_hits > 0 {
        "RIM"
    } else {
        "SWISH"
    };

    if let Ok(mut state_guard) = state.lock() {
        state_guard.shot_data.push(ShotData {
            id: shot_id,
            timestamp,
            result: result.clone(),
            backboard_hits,
            rim_hits,
            shot_type: shot_type.to_string(),
        });

        if result == "MADE" {
            state_guard.makes_count += 1;
        } else if result == "MISSED" {
            state_guard.misses_count += 1;
        }
    }
}

fn analyze_basketball_makes(state: Arc<Mutex<GameState>>) {
    let mut make_id = 0u64;
    let mut last_make_time = 0u64;
    let mut last_pos: Option<(i32, i32)> = None;

    loop {
        thread::sleep(Duration::from_millis(100));

        if let Ok(mut state_guard) = state.try_lock() {
            let now = get_timestamp();

            if let Some(det) = state_guard.detection_history.last() {
                if now - det.timestamp < 500 {
                    let pos = (det.x, det.y);

                    if let Some(last) = last_pos {
                        if pos.1 - last.1 > 30 && (now - last_make_time > 2000) {
                            make_id += 1;
                            state_guard.add_basketball_make(BasketballMake {
                                id: make_id,
                                timestamp: now,
                                confidence: 0.90,
                                x: pos.0,
                                y: pos.1,
                            });
                            last_make_time = now;
                        }
                    }
                    last_pos = Some(pos);
                } else {
                    last_pos = None;
                }
            }
        }
    }
}

fn start_api_server(state: Arc<Mutex<GameState>>, frame_lock: Arc<RwLock<Vec<u8>>>) {
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind");
    println!("✅ API Server started");

    for stream in listener.incoming() {
        if let Ok(stream) = stream {
            let state_clone = Arc::clone(&state);
            let frame_clone = Arc::clone(&frame_lock);
            thread::spawn(move || {
                handle_api_request(stream, state_clone, frame_clone);
            });
        }
    }
}

fn handle_api_request(
    mut stream: TcpStream,
    state: Arc<Mutex<GameState>>,
    frame_lock: Arc<RwLock<Vec<u8>>>,
) {
    let mut buffer = [0; 2048];
    if stream.read(&mut buffer).is_err() {
        return;
    }

    let request = String::from_utf8_lossy(&buffer[..]);
    let request_line = request.lines().next().unwrap_or("");
    let cors = "Access-Control-Allow-Origin: *\r\n";

    if request_line.starts_with("OPTIONS") {
        let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\n{}\r\n", cors).as_bytes());
        return;
    }

    if request_line.starts_with("GET /api/stream") {
        send_camera_stream(&mut stream, frame_lock);
    } else if request_line.starts_with("GET /api/status") {
        send_status(&mut stream, state, cors);
    } else if request_line.starts_with("GET /api/shots") {
        send_shots(&mut stream, state, cors);
    } else {
        let _ = stream.write_all(format!("HTTP/1.1 404\r\n{}\r\n", cors).as_bytes());
    }
}

fn send_camera_stream(stream: &mut TcpStream, frame_lock: Arc<RwLock<Vec<u8>>>) {
    // Optimize TCP settings for low latency
    let _ = stream.set_nodelay(true); // Disable Nagle's algorithm
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let header = "HTTP/1.1 200 OK\r\n\
                  Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\
                  Cache-Control: no-cache, no-store, must-revalidate\r\n\
                  Pragma: no-cache\r\n\
                  Expires: 0\r\n\
                  Connection: close\r\n\r\n";

    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }

    loop {
        // Non-blocking read - get latest frame immediately
        if let Ok(frame_guard) = frame_lock.try_read() {
            if !frame_guard.is_empty() {
                let part = format!(
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                    frame_guard.len()
                );

                // Send frame as fast as possible
                if stream.write_all(part.as_bytes()).is_err()
                    || stream.write_all(&frame_guard).is_err()
                    || stream.write_all(b"\r\n").is_err()
                {
                    break;
                }

                // Flush immediately for low latency
                let _ = stream.flush();
            }
        }

        // Minimal sleep - just yield to other threads
        thread::sleep(Duration::from_millis(8)); // ~120 FPS max send rate
    }
}

fn send_status(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let json = json!({
            "status": "running",
            "fps": format!("{:.1}", state_guard.fps),
            "makes": state_guard.makes_count,
            "misses": state_guard.misses_count
        });
        send_json_response(stream, &json, cors);
    }
}

fn send_shots(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let total = state_guard.makes_count + state_guard.misses_count;
        let json = json!({
            "total_shots": total,
            "makes": state_guard.makes_count,
            "misses": state_guard.misses_count,
            "accuracy": if total > 0 {
                format!("{:.1}%", (state_guard.makes_count as f32 / total as f32) * 100.0)
            } else { "0.0%".to_string() }
        });
        send_json_response(stream, &json, cors);
    }
}

fn send_json_response(stream: &mut TcpStream, json: &serde_json::Value, cors: &str) {
    let body = json.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\n{}Content-Type: application/json\r\n\r\n{}",
        cors, body
    );
    let _ = stream.write_all(response.as_bytes());
}

fn get_local_ip() -> String {
    if let Ok(output) = Command::new("hostname").arg("-I").output() {
        if let Ok(ip) = String::from_utf8(output.stdout) {
            if let Some(first_ip) = ip.split_whitespace().next() {
                return first_ip.to_string();
            }
        }
    }
    "192.168.100.1".to_string()
}
