// CECS490 Senior Project - Hoop IQ
// Team 2
// Christopher Hong, Gondra Kelly, Matthew "god" Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// C922x: single camera → two streams via v4l2loopback
//   heatmap  → Rust OpenCV detection overlay → RTSP → MediaMTX
//   basketball → raw BRIO frames via loopback → FFmpeg → RTSP → MediaMTX

use base64::{engine::general_purpose, Engine as _};
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector, BORDER_DEFAULT},
    imgcodecs, imgproc,
    prelude::*,
    video, videoio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Tuning constants ──────────────────────────────────────────────────
const HEATMAP_DETECT_EVERY_N_FRAMES: u64 = 3;
const JPEG_QUALITY_HEATMAP: i32 = 70;
const BG_HISTORY_FRAMES: i32 = 500;
const BG_VARIANCE_THRESHOLD: f64 = 16.0;
const MIN_CONTOUR_AREA: f64 = 50.0;
const MAX_CONTOUR_AREA: f64 = 15000.0;
const MIN_ASPECT_RATIO: f32 = 0.8;
const MAX_ASPECT_RATIO: f32 = 3.5;
const HEATMAP_GRID_SIZE: i32 = 30;
const HEATMAP_CIRCLE_RADIUS: i32 = 45;
const HEATMAP_OPACITY: f64 = 0.5;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PlayerDetection {
    id: u64,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShotEntry {
    x: f32,
    y: f32,
    made: bool,
    zone: String,
    shot_type: String,
    timestamp: u64,
}

struct FrameStore {
    heatmap: RwLock<Vec<u8>>,
}

impl FrameStore {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            heatmap: RwLock::new(Vec::new()),
        })
    }
}

struct GameState {
    heatmap_frame_count: u64,
    heatmap_fps: f32,
    current_players: Vec<PlayerDetection>,
    heatmap_data: HashMap<(i32, i32), u32>,
    make_count: u64,
    backboard_count: u32,
    rim_count: u32,
    swish_count: u32,
    backboard_make_count: u32,
    backboard_miss_count: u32,
    total_players_detected: u64,
    shot_chart: Vec<ShotEntry>,
    last_shot_type: String,
}

impl GameState {
    fn new() -> Self {
        Self {
            heatmap_frame_count: 0,
            heatmap_fps: 0.0,
            current_players: Vec::new(),
            heatmap_data: HashMap::new(),
            make_count: 0,
            backboard_count: 0,
            rim_count: 0,
            swish_count: 0,
            backboard_make_count: 0,
            backboard_miss_count: 0,
            total_players_detected: 0,
            shot_chart: Vec::new(),
            last_shot_type: "—".to_string(),
        }
    }

    fn record_shot(
        &mut self,
        players: &[PlayerDetection],
        made: bool,
        shot_type: &str,
        frame_width: i32,
        frame_height: i32,
    ) {
        if let Some(player) = players.first() {
            let x = (player.x as f32 / frame_width as f32 * 100.0).clamp(0.0, 100.0);
            let y = (player.y as f32 / frame_height as f32 * 100.0).clamp(0.0, 100.0);
            // 5-zone court classification (x/y are 0-100% of frame)
            let zone = if x < 20.0 && y > 40.0 {
                "Left Corner 3"
            } else if x > 80.0 && y > 40.0 {
                "Right Corner 3"
            } else if y < 30.0 || x < 15.0 || x > 85.0 {
                "Above Break 3"
            } else if y > 60.0 && x > 30.0 && x < 70.0 {
                "Paint"
            } else {
                "Mid-Range"
            }
            .to_string();
            self.shot_chart.push(ShotEntry {
                x,
                y,
                made,
                zone,
                shot_type: shot_type.to_string(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            });
            if self.shot_chart.len() > 500 {
                self.shot_chart.remove(0);
            }
        }
        self.last_shot_type = shot_type.to_string();
    }

    fn update_heatmap(&mut self, players: &[PlayerDetection]) {
        for player in players {
            let key = (
                (player.x / HEATMAP_GRID_SIZE) * HEATMAP_GRID_SIZE,
                (player.y / HEATMAP_GRID_SIZE) * HEATMAP_GRID_SIZE,
            );
            *self.heatmap_data.entry(key).or_insert(0) += 1;
        }
        if self.heatmap_data.len() > 1000 {
            self.heatmap_data.retain(|_, &mut v| v >= 5);
        }
    }

    fn attempts(&self) -> u64 {
        self.make_count + self.backboard_miss_count as u64 + self.rim_count as u64
    }

    fn fg_percent(&self) -> Option<f32> {
        let a = self.attempts();
        if a == 0 {
            None
        } else {
            Some(self.make_count as f32 / a as f32 * 100.0)
        }
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn detect_players(
    frame: &Mat,
    bg_sub: &mut opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
) -> Result<Vec<PlayerDetection>, opencv::Error> {
    let mut players = Vec::new();
    let timestamp = get_timestamp();

    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    let mut fg_mask = Mat::default();
    opencv::prelude::BackgroundSubtractorTrait::apply(&mut *bg_sub, &gray, &mut fg_mask, -1.0)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut clean_mask = Mat::default();
    imgproc::morphology_ex(
        &fg_mask,
        &mut clean_mask,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &clean_mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        if area > MIN_CONTOUR_AREA && area < MAX_CONTOUR_AREA {
            let rect = imgproc::bounding_rect(&contour)?;
            let aspect_ratio = rect.height as f32 / rect.width as f32;
            if aspect_ratio > MIN_ASPECT_RATIO && aspect_ratio < MAX_ASPECT_RATIO {
                players.push(PlayerDetection {
                    id: timestamp + i as u64,
                    x: rect.x + rect.width / 2,
                    y: rect.y + rect.height / 2,
                    width: rect.width,
                    height: rect.height,
                    timestamp,
                });
            }
        }
    }
    Ok(players)
}

fn draw_heatmap_overlay(
    frame: &mut Mat,
    heatmap_data: &HashMap<(i32, i32), u32>,
) -> Result<(), opencv::Error> {
    if heatmap_data.is_empty() {
        return Ok(());
    }

    let mut heatmap_mat =
        Mat::zeros(frame.rows(), frame.cols(), opencv::core::CV_8UC3)?.to_mat()?;
    let max_intensity = heatmap_data.values().max().copied().unwrap_or(1);

    for (&(x, y), &intensity) in heatmap_data {
        let normalized = (intensity as f64 / max_intensity as f64).min(1.0);
        let hue = ((1.0 - normalized) * 120.0) as i32;
        imgproc::circle(
            &mut heatmap_mat,
            Point::new(x, y),
            HEATMAP_CIRCLE_RADIUS,
            Scalar::new(hue as f64, 255.0, 255.0, 255.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    let mut heatmap_bgr = Mat::default();
    imgproc::cvt_color(&heatmap_mat, &mut heatmap_bgr, imgproc::COLOR_HSV2BGR, 0)?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &heatmap_bgr,
        &mut blurred,
        Size::new(25, 25),
        0.0,
        0.0,
        BORDER_DEFAULT,
    )?;

    let mut output = Mat::default();
    core::add_weighted(
        frame,
        1.0 - HEATMAP_OPACITY,
        &blurred,
        HEATMAP_OPACITY,
        0.0,
        &mut output,
        -1,
    )?;
    output.copy_to(frame)?;
    Ok(())
}

fn draw_player_overlay(frame: &mut Mat, players: &[PlayerDetection]) -> Result<(), opencv::Error> {
    for player in players {
        imgproc::rectangle(
            frame,
            Rect::new(
                player.x - player.width / 2,
                player.y - player.height / 2,
                player.width,
                player.height,
            ),
            Scalar::new(0.0, 200.0, 80.0, 255.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;
    }

    let cyan = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let texts: &[&str] = &["PLAYER HEATMAP", &format!("Players: {}", players.len())];
    for (i, text) in texts.iter().enumerate() {
        imgproc::put_text(
            frame,
            text,
            Point::new(10, 28 + i as i32 * 28),
            imgproc::FONT_HERSHEY_DUPLEX,
            0.55,
            cyan,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

fn push_frame_to_rtsp(frame_data: &[u8], ffmpeg_stdin: &mut std::process::ChildStdin) -> bool {
    ffmpeg_stdin.write_all(frame_data).is_ok()
}

fn process_heatmap_camera(
    mut camera: videoio::VideoCapture,
    state: Arc<Mutex<GameState>>,
    frames: Arc<FrameStore>,
) -> Result<(), opencv::Error> {
    let mut frame = Mat::default();
    let mut frame_times: VecDeque<Instant> = VecDeque::new();
    let mut local_count: u64 = 0;

    let mut bg_sub =
        video::create_background_subtractor_mog2(BG_HISTORY_FRAMES, BG_VARIANCE_THRESHOLD, false)?;

    // ── FFmpeg 1: heatmap stream (OpenCV overlay) → RTSP ─────────────
    let mut ffmpeg_heatmap = std::process::Command::new("ffmpeg")
        .args([
            "-f",
            "mjpeg",
            "-framerate",
            "60",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-threads",
            "4",
            "-b:v",
            "4M",
            "-maxrate",
            "6M",
            "-bufsize",
            "2M",
            "-g",
            "60",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            "rtsp://localhost:8554/heatmap",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to spawn FFmpeg heatmap");

    let mut ffmpeg_heatmap_stdin = ffmpeg_heatmap
        .stdin
        .take()
        .expect("Failed to get heatmap stdin");

    // ── FFmpeg 2: raw MJPEG → v4l2loopback → basketball RTSP ─────────
    let mut ffmpeg_loop = std::process::Command::new("ffmpeg")
        .args([
            "-f",
            "mjpeg",
            "-framerate",
            "60",
            "-i",
            "pipe:0",
            "-vf",
            "format=yuv420p",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "v4l2",
            "/dev/video10",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to spawn FFmpeg loopback writer");

    let mut ffmpeg_loop_stdin = ffmpeg_loop
        .stdin
        .take()
        .expect("Failed to get loopback stdin");

    println!("C922x dual-stream started: heatmap + basketball @ 720p60");

    loop {
        camera.read(&mut frame)?;
        if frame.empty() {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        local_count += 1;

        // ── Push raw frame to loopback BEFORE drawing overlays ────────
        let mut raw_buf = Vector::<u8>::new();
        let raw_params = Vector::<i32>::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, 85]);
        if imgcodecs::imencode(".jpg", &frame, &mut raw_buf, &raw_params).is_ok() {
            let _ = ffmpeg_loop_stdin.write_all(&raw_buf.to_vec());
        }

        // ── Player detection every N frames ───────────────────────────
        let players = if local_count % HEATMAP_DETECT_EVERY_N_FRAMES == 0 {
            detect_players(&frame, &mut bg_sub).unwrap_or_default()
        } else {
            state.lock().unwrap().current_players.clone()
        };

        let heatmap_snapshot = {
            let mut s = state.lock().unwrap();
            s.heatmap_frame_count = local_count;
            s.total_players_detected += players.len() as u64;
            s.update_heatmap(&players);
            s.current_players = players.clone();

            frame_times.push_back(Instant::now());
            if frame_times.len() > 30 {
                frame_times.pop_front();
            }
            if frame_times.len() >= 2 {
                let dur = frame_times
                    .back()
                    .unwrap()
                    .duration_since(*frame_times.front().unwrap());
                s.heatmap_fps = (frame_times.len() - 1) as f32 / dur.as_secs_f32();
            }
            s.heatmap_data.clone()
        };

        // ── Draw overlays ─────────────────────────────────────────────
        if let Err(e) = draw_heatmap_overlay(&mut frame, &heatmap_snapshot) {
            eprintln!("Heatmap overlay error: {}", e);
        }
        if let Err(e) = draw_player_overlay(&mut frame, &players) {
            eprintln!("Player overlay error: {}", e);
        }

        // ── Encode overlaid frame → heatmap RTSP ─────────────────────
        let mut buf = Vector::<u8>::new();
        let params =
            Vector::<i32>::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, JPEG_QUALITY_HEATMAP]);
        if imgcodecs::imencode(".jpg", &frame, &mut buf, &params).is_ok() {
            let jpeg = buf.to_vec();
            if let Ok(mut f) = frames.heatmap.write() {
                *f = jpeg.clone();
            }
            if !push_frame_to_rtsp(&jpeg, &mut ffmpeg_heatmap_stdin) {
                eprintln!("Heatmap FFmpeg pipe broken, restarting...");
                break;
            }
        }
    }

    Ok(())
}

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    thread::spawn(move || {
        let ports = ["/dev/ttyUSB0", "/dev/ttyACM0"];
        loop {
            for port in &ports {
                println!("Attempting to connect to ESP32 on {}", port);
                if let Ok(mut file) = std::fs::OpenOptions::new().read(true).write(true).open(port) {
                    println!("Connected to ESP32 on {}", port);
                    thread::sleep(Duration::from_millis(2000));
                    let _ = file.write_all(b"ENABLE\n");
                    println!("Sent ENABLE to ESP32");
                    for line in BufReader::new(file).lines() {
                        if let Ok(data) = line {
                            let data = data.trim().to_string();
                            if data.is_empty() {
                                continue;
                            }
                            let mut s = state.lock().unwrap();
                            let players = s.current_players.clone();
                            if data.starts_with("MAKE:") {
                                // IR beam broken with prior piezo = backboard make
                                s.make_count += 1;
                                s.backboard_make_count += 1;
                                s.record_shot(&players, true, "Backboard Make", 1280, 720);
                                println!("Basketball make (backboard): total {}", s.make_count);
                            } else if data.starts_with("SWISH:") {
                                // IR beam broken with NO prior piezo = swish
                                s.make_count += 1;
                                s.swish_count += 1;
                                s.record_shot(&players, true, "Swish", 1280, 720);
                                println!("Swish! total makes: {}", s.make_count);
                            } else if data.starts_with("BACK:") {
                                // Piezo hit, no IR = backboard miss
                                s.backboard_count += 1;
                                s.backboard_miss_count += 1;
                                s.record_shot(&players, false, "Backboard Miss", 1280, 720);
                                println!("Backboard miss: total {}", s.backboard_count);
                            } else if data.starts_with("RIM:") {
                                // Rim hit only
                                s.rim_count += 1;
                                s.record_shot(&players, false, "Rim Hit", 1280, 720);
                                println!("Rim hit: total {}", s.rim_count);
                            }
                        }
                    }
                    println!("ESP32 disconnected on {}", port);
                }
            }
            thread::sleep(Duration::from_secs(3));
        }
    });
}

fn send_to_cloud_api(state: Arc<Mutex<GameState>>, frames: Arc<FrameStore>, api_url: String) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::new();
        println!("Cloud API updater started");

        loop {
            thread::sleep(Duration::from_secs(1));

            let h_frame = frames.heatmap.read().unwrap().clone();

            let payload = {
                let s = state.lock().unwrap();
                let fg = s
                    .fg_percent()
                    .map(|v| format!("{:.1}", v))
                    .unwrap_or_else(|| "0.0".to_string());
                json!({
                    "basketball_fps": 0,
                    "makes": s.make_count,
                    "attempts": s.attempts(),
                    "swishes": s.swish_count,
                    "backboard_makes": s.backboard_make_count,
                    "backboard_misses": s.backboard_miss_count,
                    "backboard_hits": s.backboard_count,
                    "rim_hits": s.rim_count,
                    "fg_percent": fg,
                    "last_shot_type": s.last_shot_type,
                    "trajectories": 0,
                    "heatmap_fps": s.heatmap_fps,
                    "current_players": s.current_players.len() as u64,
                    "total_players_detected": s.total_players_detected,
                    "heatmap_points": s.heatmap_data.len() as u64,
                    "shot_chart": s.shot_chart,
                    "basketball_frame": "",
                    "heatmap_frame": general_purpose::STANDARD.encode(&h_frame),
                    "timestamp": get_timestamp()
                })
            };

            match client
                .post(format!("{}/update", api_url))
                .json(&payload)
                .timeout(Duration::from_secs(3))
                .send()
            {
                Ok(_) => {}
                Err(e) => eprintln!("Cloud API error: {}", e),
            }
        }
    });
}

fn main() -> Result<(), opencv::Error> {
    println!("========================================");
    println!("HOOP IQ - C922x Dual Stream 720p@60fps");
    println!("CECS490 Senior Project - Team 2");
    println!("========================================\n");

    let game_state = Arc::new(Mutex::new(GameState::new()));
    let frame_store = FrameStore::new();

    println!("Opening C922x camera at 720p@60fps...");
    let mut cam = None;
    for attempt in 1..=5 {
        match videoio::VideoCapture::new(0, videoio::CAP_V4L2) {
            Ok(mut c) if videoio::VideoCapture::is_opened(&c).unwrap_or(false) => {
                c.set(
                    videoio::CAP_PROP_FOURCC,
                    videoio::VideoWriter::fourcc('M', 'J', 'P', 'G').unwrap() as f64,
                )
                .ok();
                c.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0).ok();
                c.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0).ok();
                c.set(videoio::CAP_PROP_FPS, 60.0).ok();
                c.set(videoio::CAP_PROP_BUFFERSIZE, 1.0).ok();
                let w = c.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(0.0);
                let h = c.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(0.0);
                let fps = c.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);
                println!(
                    "C922x negotiated: {}x{} @ {}fps on attempt {}",
                    w, h, fps, attempt
                );
                cam = Some(c);
                break;
            }
            Ok(_) | Err(_) => {
                if attempt < 5 {
                    println!("C922x attempt {} failed, retrying in 2s...", attempt);
                    thread::sleep(Duration::from_secs(2));
                }
            }
        }
    }

    let cam = match cam {
        Some(c) => c,
        None => {
            eprintln!("Could not open C922x after 5 attempts");
            return Ok(());
        }
    };

    listen_to_esp32(Arc::clone(&game_state));

    let cloud_api_url = "https://cecs490-senior-project.onrender.com".to_string();
    send_to_cloud_api(
        Arc::clone(&game_state),
        Arc::clone(&frame_store),
        cloud_api_url,
    );

    println!("Starting C922x dual-stream thread...");
    let state_h = Arc::clone(&game_state);
    let frames_h = Arc::clone(&frame_store);
    thread::spawn(move || {
        if let Err(e) = process_heatmap_camera(cam, state_h, frames_h) {
            eprintln!("C922x camera error: {}", e);
        }
    });

    println!("All systems running!");
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}





