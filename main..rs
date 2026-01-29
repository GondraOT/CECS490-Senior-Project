// CECS490 Senior Project - Hoop IQ
// Team 2 - Player Heatmap System
// Christopher Hong, Gondra Kelly, Matthew Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// IP Camera Player Detection & Heatmap Generation

use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io::{prelude::*, BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PlayerDetection {
    id: u64,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    timestamp: u64,
    confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HeatmapPoint {
    x: i32,
    y: i32,
    intensity: u32,
    timestamp: u64,
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

struct GameState {
    frame_count: u64,
    fps: f32,
    current_players: Vec<PlayerDetection>,
    player_history: Vec<PlayerDetection>,
    heatmap_data: HashMap<(i32, i32), u32>,
    last_frame: Arc<RwLock<Vec<u8>>>,
    shot_data: Vec<ShotData>,
    makes_count: u64,
    misses_count: u64,
    total_players_detected: u64,
}

impl GameState {
    fn new() -> Self {
        Self {
            frame_count: 0,
            fps: 0.0,
            current_players: Vec::new(),
            player_history: Vec::new(),
            heatmap_data: HashMap::new(),
            last_frame: Arc::new(RwLock::new(Vec::new())),
            shot_data: Vec::new(),
            makes_count: 0,
            misses_count: 0,
            total_players_detected: 0,
        }
    }

    fn add_player_detection(&mut self, detection: PlayerDetection) {
        self.player_history.push(detection.clone());
        if self.player_history.len() > 100 {
            self.player_history.remove(0);
        }

        let grid_x = (detection.x / 10) * 10;
        let grid_y = (detection.y / 10) * 10;
        let grid_pos = (grid_x, grid_y);

        *self.heatmap_data.entry(grid_pos).or_insert(0) += 1;
        self.total_players_detected += 1;
    }

    fn get_heatmap_points(&self) -> Vec<HeatmapPoint> {
        self.heatmap_data
            .iter()
            .map(|((x, y), intensity)| HeatmapPoint {
                x: *x,
                y: *y,
                intensity: *intensity,
                timestamp: get_timestamp(),
            })
            .collect()
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HOOP IQ - Player Heatmap System");
    println!("CECS490 Senior Project - Team 2\n");

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

    thread::sleep(Duration::from_millis(1000));

    let local_ip = get_local_ip();
    println!("API Server: http://{}:8080", local_ip);
    println!("Stream: http://{}:8080/api/stream", local_ip);
    println!("Heatmap: http://{}:8080/api/heatmap\n", local_ip);

    // Update with your camera password
    let rtsp_url = "rtsp://admin:CHANGEME@192.168.4.2:554/cam/realmonitor?channel=1&subtype=0";

    println!("Connecting to IP camera...");
    let cam = videoio::VideoCapture::from_file(rtsp_url, videoio::CAP_FFMPEG)?;

    if !videoio::VideoCapture::is_opened(&cam)? {
        eprintln!("ERROR: Could not connect to IP camera");
        eprintln!("Check: Camera IP, password, RTSP port, WiFi connection");
        return Ok(());
    }

    println!("Camera connected");

    let actual_width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let actual_height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    let actual_fps = cam.get(videoio::CAP_PROP_FPS)?;

    println!(
        "Stream: {}x{} @ {:.1} FPS\n",
        actual_width, actual_height, actual_fps
    );

    let state_cam = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_player_detection(cam, state_cam) {
            eprintln!("Camera processing error: {}", e);
        }
    });

    println!("Player detection active\n");

    loop {
        thread::sleep(Duration::from_secs(5));
        if let Ok(state) = game_state.lock() {
            println!(
                "Frames: {} | Players: {} | Heatmap Points: {} | FPS: {:.1}",
                state.frame_count,
                state.total_players_detected,
                state.heatmap_data.len(),
                state.fps
            );
        }
    }
}

fn process_player_detection(
    mut cam: videoio::VideoCapture,
    state: Arc<Mutex<GameState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame = Mat::default();
    //let mut blob = Mat::default();
    let mut blob;

    let mut last_time = Instant::now();
    let mut frame_times = Vec::with_capacity(30);
    let mut detection_id: u64 = 0;

    let config = "yolov4-tiny.cfg";
    let weights = "yolov4-tiny.weights";

    println!("Loading YOLO model...");
    let mut net = match dnn::read_net_from_darknet(config, weights) {
        Ok(n) => {
            println!("YOLO model loaded");
            n
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load YOLO model: {}", e);
            eprintln!("Download: wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights");
            eprintln!("Download: wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg");
            return Err(Box::new(e));
        }
    };

    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    let output_layers = net.get_unconnected_out_layers_names()?;

    let mut encode_params = Vector::<i32>::new();
    encode_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    encode_params.push(70);

    let mut current_players = Vec::with_capacity(10);
    let mut frame_counter = 0u64;

    loop {
        let now = Instant::now();
        let frame_time = now.duration_since(last_time).as_secs_f32();
        frame_times.push(frame_time);
        if frame_times.len() > 30 {
            frame_times.remove(0);
        }
        let avg_fps = if !frame_times.is_empty() {
            frame_times.len() as f32 / frame_times.iter().sum::<f32>()
        } else {
            0.0
        };
        last_time = now;

        if cam.read(&mut frame).is_err() || frame.empty() {
            thread::sleep(Duration::from_millis(100));
            continue;
        }

        frame_counter += 1;
        let timestamp = get_timestamp();
        let should_detect = frame_counter % 3 == 0;

        if should_detect {
            current_players.clear();

            let blob_size = Size::new(416, 416);
            let scale = 1.0 / 255.0;
            let mean = Scalar::new(0.0, 0.0, 0.0, 0.0);
            let swap_rb = true;
            let crop = false;

            blob =
                dnn::blob_from_image(&frame, scale, blob_size, mean, swap_rb, crop, core::CV_32F)?;
            net.set_input(&blob, "", 1.0, Scalar::default())?;

            let mut detections = Vector::<Mat>::new();
            net.forward(&mut detections, &output_layers)?;

            let frame_height = frame.rows();
            let frame_width = frame.cols();

            for detection_mat in detections.iter() {
                for i in 0..detection_mat.rows() {
                    let scores = detection_mat.row(i)?;
                    let scores_data = scores.data_typed::<f32>()?;

                    let mut max_score = 0.0f32;
                    let mut class_id = 0;
                    for j in 5..scores.cols() {
                        let score = scores_data[j as usize];
                        if score > max_score {
                            max_score = score;
                            class_id = j - 5;
                        }
                    }

                    if class_id == 0 && max_score > 0.5 {
                        let center_x = (scores_data[0] * frame_width as f32) as i32;
                        let center_y = (scores_data[1] * frame_height as f32) as i32;
                        let width = (scores_data[2] * frame_width as f32) as i32;
                        let height = (scores_data[3] * frame_height as f32) as i32;

                        detection_id += 1;
                        current_players.push(PlayerDetection {
                            id: detection_id,
                            x: center_x,
                            y: center_y,
                            width,
                            height,
                            timestamp,
                            confidence: max_score,
                        });
                    }
                }
            }
        }

        for player in &current_players {
            let top_left = Point::new(player.x - player.width / 2, player.y - player.height / 2);

            imgproc::rectangle(
                &mut frame,
                Rect::new(top_left.x, top_left.y, player.width, player.height),
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;

            imgproc::circle(
                &mut frame,
                Point::new(player.x, player.y),
                5,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        imgproc::put_text(
            &mut frame,
            &format!("FPS: {:.1} | Players: {}", avg_fps, current_players.len()),
            Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        let mut buf = Vector::new();
        imgcodecs::imencode(".jpg", &frame, &mut buf, &encode_params)?;
        let jpeg_data = buf.to_vec();

        if let Ok(mut state_guard) = state.try_lock() {
            state_guard.frame_count += 1;
            state_guard.fps = avg_fps;

            if should_detect {
                state_guard.current_players = current_players.clone();
                for player in &current_players {
                    state_guard.add_player_detection(player.clone());
                }
            }

            if let Ok(mut frame_guard) = state_guard.last_frame.try_write() {
                *frame_guard = jpeg_data;
            }
        }
    }
}

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    // USB Serial connection - tries both common serial ports
    let serial_ports = vec!["/dev/ttyUSB0", "/dev/ttyACM0"];

    for port in &serial_ports {
        if let Ok(file) = std::fs::OpenOptions::new().read(true).open(port) {
            println!("ESP32 connected on {}", port);
            let reader = BufReader::new(file);

            for line in reader.lines() {
                if let Ok(data) = line {
                    let data = data.trim();
                    if data.starts_with("RESULT:") {
                        parse_shot_result(&data, &state);
                    }
                }
            }
            return;
        }
    }
    println!("WARNING: No ESP32 found - shot data integration disabled");
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

        println!("Shot {} - {} ({})", shot_id, result, shot_type);
    }
}

fn start_api_server(state: Arc<Mutex<GameState>>, frame_lock: Arc<RwLock<Vec<u8>>>) {
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind");
    println!("API Server started");

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
    } else if request_line.starts_with("GET /api/heatmap") {
        send_heatmap(&mut stream, state, cors);
    } else if request_line.starts_with("GET /api/shots") {
        send_shots(&mut stream, state, cors);
    } else {
        let _ = stream.write_all(format!("HTTP/1.1 404\r\n{}\r\n", cors).as_bytes());
    }
}

fn send_camera_stream(stream: &mut TcpStream, frame_lock: Arc<RwLock<Vec<u8>>>) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let header = "HTTP/1.1 200 OK\r\n\
                  Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\
                  Cache-Control: no-cache\r\n\
                  Connection: close\r\n\r\n";

    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }

    loop {
        if let Ok(frame_guard) = frame_lock.try_read() {
            if !frame_guard.is_empty() {
                let part = format!(
                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                    frame_guard.len()
                );

                if stream.write_all(part.as_bytes()).is_err()
                    || stream.write_all(&frame_guard).is_err()
                    || stream.write_all(b"\r\n").is_err()
                {
                    break;
                }
                let _ = stream.flush();
            }
        }
        thread::sleep(Duration::from_millis(33));
    }
}

fn send_status(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let json = json!({
            "status": "running",
            "fps": format!("{:.1}", state_guard.fps),
            "players_detected": state_guard.total_players_detected,
            "heatmap_points": state_guard.heatmap_data.len(),
            "makes": state_guard.makes_count,
            "misses": state_guard.misses_count
        });
        send_json_response(stream, &json, cors);
    }
}

fn send_heatmap(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let heatmap_points = state_guard.get_heatmap_points();
        let json = json!({
            "heatmap": heatmap_points,
            "total_points": heatmap_points.len(),
            "timestamp": get_timestamp()
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
            } else { "0.0%".to_string() },
            "recent_shots": state_guard.shot_data.iter().rev().take(10).collect::<Vec<_>>()
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
    if let Ok(output) = std::process::Command::new("hostname").arg("-I").output() {
        if let Ok(ip) = String::from_utf8(output.stdout) {
            if let Some(first_ip) = ip.split_whitespace().next() {
                return first_ip.to_string();
            }
        }
    }
    "192.168.4.1".to_string()
}
