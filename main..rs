// CECS490 Senior Project - Hoop IQ
// Team 2
// Christopher Hong, Gondra Kelly, Matthew "god" Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// Logitech C922x (Player Heatmap) and C920 (Basketball Tracker) Dual Camera Integration

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
struct PlayerDetection {
    id: u64,
    x: i32,
    y: i32,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BasketballMake {
    id: u64,
    timestamp: u64,
    confidence: f32,
    x: i32,
    y: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShotTrajectory {
    id: u64,
    start_time: u64,
    start_position: (i32, i32),
    trajectory_points: Vec<(i32, i32, u64)>, // (x, y, timestamp)
    peak_height: i32,
    arc_quality: f32, // 0.0-1.0 score
    shot_angle: f32,
    approaching_rim: bool,
}

#[allow(dead_code)]
impl ShotTrajectory {
    fn new(id: u64, x: i32, y: i32, timestamp: u64) -> Self {
        Self {
            id,
            start_time: timestamp,
            start_position: (x, y),
            trajectory_points: vec![(x, y, timestamp)],
            peak_height: y,
            arc_quality: 0.0,
            shot_angle: 0.0,
            approaching_rim: false,
        }
    }

    fn add_point(&mut self, x: i32, y: i32, timestamp: u64) {
        self.trajectory_points.push((x, y, timestamp));

        // Track peak height (lowest y value = highest point on screen)
        if y < self.peak_height {
            self.peak_height = y;
        }

        // Check if ball is approaching rim (moving downward in lower portion of frame)
        if let Some(last) = self.trajectory_points.iter().rev().nth(1) {
            let moving_down = y > last.1;
            let in_lower_half = y > 360; // Lower half of 720p frame
            self.approaching_rim = moving_down && in_lower_half;
        }

        // Calculate arc quality
        self.calculate_arc_quality();
    }

    fn calculate_arc_quality(&mut self) {
        if self.trajectory_points.len() < 3 {
            return;
        }

        // Simple arc quality: measure how much the ball went up
        let start_y = self.start_position.1;
        let arc_height = start_y - self.peak_height;

        // Good arc: 100-300 pixels of rise (adjust based on your setup)
        self.arc_quality = (arc_height as f32 / 200.0).min(1.0).max(0.0);

        // Calculate shot angle
        if let Some(first) = self.trajectory_points.first() {
            if let Some(second) = self.trajectory_points.get(5) {
                let dx = (second.0 - first.0) as f32;
                let dy = (second.1 - first.1) as f32;
                self.shot_angle = dy.atan2(dx).to_degrees();
            }
        }
    }

    fn duration_ms(&self) -> u64 {
        if let Some(last) = self.trajectory_points.last() {
            return last.2 - self.start_time;
        }
        0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletedShot {
    shot_id: u64,
    timestamp: u64,

    // Camera data
    start_position: (i32, i32),
    arc_quality: f32,
    shot_angle: f32,
    trajectory_points: Vec<(i32, i32, u64)>,

    // ESP32 data (filled in when received)
    result: Option<String>, // "MADE" or "MISSED"
    backboard_hits: u32,
    rim_hits: u32,
    shot_type: Option<String>, // "SWISH", "RIM", "BACKBOARD", "BOTH"
}

struct GameState {
    frame_count: u64,
    fps_basketball: f32,
    fps_heatmap: f32,
    current_detections: Vec<Detection>,
    detection_history: Vec<Detection>,
    basketball_makes: Vec<BasketballMake>,
    basketball_frame: Arc<RwLock<Vec<u8>>>,
    heatmap_frame: Arc<RwLock<Vec<u8>>>,
    make_count: u64,
    shot_data: Vec<ShotData>,
    makes_count: u64,
    misses_count: u64,

    active_trajectories: Vec<ShotTrajectory>,
    completed_shots: Vec<CompletedShot>,
    waiting_for_esp32: Option<u64>, // Shot ID waiting for ESP32 result

    // Player heatmap data
    player_positions: Vec<PlayerDetection>,
    heatmap_grid: Vec<Vec<u32>>, // Grid for accumulating player positions
}

impl GameState {
    fn new() -> Self {
        // Initialize 20x20 grid for heatmap (adjust based on your court dimensions)
        let heatmap_grid = vec![vec![0u32; 20]; 20];

        Self {
            frame_count: 0,
            fps_basketball: 0.0,
            fps_heatmap: 0.0,
            current_detections: Vec::new(),
            detection_history: Vec::new(),
            basketball_makes: Vec::new(),
            basketball_frame: Arc::new(RwLock::new(Vec::new())),
            heatmap_frame: Arc::new(RwLock::new(Vec::new())),
            make_count: 0,
            shot_data: Vec::new(),
            makes_count: 0,
            misses_count: 0,
            active_trajectories: Vec::new(),
            completed_shots: Vec::new(),
            waiting_for_esp32: None,
            player_positions: Vec::new(),
            heatmap_grid,
        }
    }

    fn add_detection(&mut self, detection: Detection) {
        self.detection_history.push(detection);
        if self.detection_history.len() > 50 {
            self.detection_history.remove(0);
        }
    }

    #[allow(dead_code)]
    fn add_basketball_make(&mut self, make: BasketballMake) {
        println!("MAKE #{} at ({}, {})", make.id, make.x, make.y);
        self.basketball_makes.push(make.clone());
        if self.basketball_makes.len() > 50 {
            self.basketball_makes.remove(0);
        }
        self.make_count += 1;
    }

    fn add_player_position(&mut self, player: PlayerDetection) {
        // Add to position history
        self.player_positions.push(player.clone());
        if self.player_positions.len() > 500 {
            self.player_positions.remove(0);
        }

        // Update heatmap grid
        let grid_x = ((player.x as f32 / 1280.0) * 20.0).floor() as usize;
        let grid_y = ((player.y as f32 / 720.0) * 20.0).floor() as usize;

        if grid_x < 20 && grid_y < 20 {
            self.heatmap_grid[grid_y][grid_x] += 1;
        }
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("HOOP IQ - Dual Camera System");
    println!("C922x: Player Heatmap Tracking");
    println!("C920: Basketball Detection");
    println!("CECS490 Senior Project\n");

    // Configure C922x (Camera 0) for player tracking at 60 FPS
    configure_c922x_heatmap();

    // Configure C920 (Camera 1) for basketball tracking at 30 FPS
    configure_c920_basketball();

    let game_state = Arc::new(Mutex::new(GameState::new()));
    let basketball_frame_lock = Arc::clone(&game_state.lock().unwrap().basketball_frame);
    let heatmap_frame_lock = Arc::clone(&game_state.lock().unwrap().heatmap_frame);

    let esp32_state = Arc::clone(&game_state);
    thread::spawn(move || {
        listen_to_esp32(esp32_state);
    });

    let api_state = Arc::clone(&game_state);
    let api_basketball_frame = Arc::clone(&basketball_frame_lock);
    let api_heatmap_frame = Arc::clone(&heatmap_frame_lock);
    thread::spawn(move || {
        start_api_server(api_state, api_basketball_frame, api_heatmap_frame);
    });

    // Shot tracker (uncomment when ESP32 is connected)
    /*let analyzer_state = Arc::clone(&game_state);
    thread::spawn(move || {
        analyze_basketball_makes(analyzer_state);
    });*/

    thread::sleep(Duration::from_millis(1000));

    let local_ip = get_local_ip();
    println!("API Server: http://{}:8080", local_ip);
    println!(
        "Basketball Stream: http://{}:8080/api/stream/basketball",
        local_ip
    );
    println!(
        "Heatmap Stream: http://{}:8080/api/stream/heatmap\n",
        local_ip
    );

    // Start C920 basketball tracking camera (Camera 2)
    let mut cam_basketball = videoio::VideoCapture::from_file("/dev/video2", videoio::CAP_V4L2)?;
    if !videoio::VideoCapture::is_opened(&cam_basketball)? {
        eprintln!("Could not open C920 basketball camera");
        return Ok(());
    }

    println!("C920 basketball camera opened");

    // Configure C920 for basketball tracking
    let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    cam_basketball.set(videoio::CAP_PROP_FOURCC, fourcc as f64)?;
    cam_basketball.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cam_basketball.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cam_basketball.set(videoio::CAP_PROP_FPS, 30.0)?;
    cam_basketball.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    thread::sleep(Duration::from_millis(500));

    let basketball_fps = cam_basketball.get(videoio::CAP_PROP_FPS)?;
    let basketball_width = cam_basketball.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let basketball_height = cam_basketball.get(videoio::CAP_PROP_FRAME_HEIGHT)?;

    println!(
        "C920 Basketball: {}x{} @ {:.1} FPS",
        basketball_width, basketball_height, basketball_fps
    );

    // Start C922x heatmap camera (Camera 0)
    let mut cam_heatmap = videoio::VideoCapture::new(0, videoio::CAP_V4L2)?;
    if !videoio::VideoCapture::is_opened(&cam_heatmap)? {
        eprintln!("Could not open C922x heatmap camera");
        return Ok(());
    }

    println!("C922x heatmap camera opened");

    // Configure C922x for heatmap tracking
    cam_heatmap.set(videoio::CAP_PROP_FOURCC, fourcc as f64)?;
    cam_heatmap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cam_heatmap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cam_heatmap.set(videoio::CAP_PROP_FPS, 60.0)?;
    cam_heatmap.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    thread::sleep(Duration::from_millis(500));

    let heatmap_fps = cam_heatmap.get(videoio::CAP_PROP_FPS)?;
    let heatmap_width = cam_heatmap.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let heatmap_height = cam_heatmap.get(videoio::CAP_PROP_FRAME_HEIGHT)?;

    println!(
        "C922x Heatmap: {}x{} @ {:.1} FPS\n",
        heatmap_width, heatmap_height, heatmap_fps
    );

    // Start basketball tracking thread (C920, Camera 2)
    let state_basketball = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_basketball_camera(cam_basketball, 2, state_basketball) {
            eprintln!("Basketball camera error: {}", e);
        }
    });

    // Start heatmap tracking thread (C922x, Camera 0)
    let state_heatmap = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_heatmap_camera(cam_heatmap, 0, state_heatmap) {
            eprintln!("Heatmap camera error: {}", e);
        }
    });

    println!("Detection active!\n");

    loop {
        thread::sleep(Duration::from_secs(5));
        if let Ok(state) = game_state.lock() {
            println!(
                "Frames: {} | Makes: {} | Misses: {} | Basketball FPS: {:.1} | Heatmap FPS: {:.1}",
                state.frame_count,
                state.makes_count,
                state.misses_count,
                state.fps_basketball,
                state.fps_heatmap
            );
        }
    }
}

fn configure_c922x_heatmap() {
    println!("Configuring C922x (Camera 0) for player heatmap tracking...");

    // C922x settings for 720p @ 60 FPS
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-fmt-video=width=1280,height=720,pixelformat=MJPG",
        ])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-parm=60"])
        .output();

    thread::sleep(Duration::from_millis(500));

    // Disable autofocus
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=focus_auto=0"])
        .output();

    // Set manual focus to infinity for court overview
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=focus_absolute=0"])
        .output();

    // Manual exposure for consistent frame timing
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=exposure_auto=1"])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video0", "--set-ctrl=exposure_absolute=156"])
        .output();

    // Disable auto white balance
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-ctrl=white_balance_temperature_auto=0",
        ])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video0",
            "--set-ctrl=white_balance_temperature=4000",
        ])
        .output();

    println!("C922x configured for heatmap tracking");
}

fn configure_c920_basketball() {
    println!("Configuring C920 (Camera 1) for basketball tracking...");

    // C920 settings for 720p @ 30 FPS (more stable for basketball detection)
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video2",
            "--set-fmt-video=width=1280,height=720,pixelformat=MJPG",
        ])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video2", "--set-parm=30"])
        .output();

    thread::sleep(Duration::from_millis(500));

    // Disable autofocus
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video2", "--set-ctrl=focus_auto=0"])
        .output();

    // Focus on hoop area
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video2", "--set-ctrl=focus_absolute=20"])
        .output();

    // Manual exposure
    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video2", "--set-ctrl=exposure_auto=1"])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&["-d", "/dev/video2", "--set-ctrl=exposure_absolute=200"])
        .output();

    // Disable auto white balance
    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video2",
            "--set-ctrl=white_balance_temperature_auto=0",
        ])
        .output();

    let _ = Command::new("v4l2-ctl")
        .args(&[
            "-d",
            "/dev/video2",
            "--set-ctrl=white_balance_temperature=4000",
        ])
        .output();

    println!("C920 configured for basketball tracking");
}

fn process_basketball_camera(
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
    let mut frame_times = Vec::with_capacity(30);
    let mut detection_id: u64 = camera_id as u64 * 100000;

    // Optimize JPEG encoding for basketball tracking
    let mut encode_params = Vector::<i32>::new();
    encode_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    encode_params.push(70);
    encode_params.push(imgcodecs::IMWRITE_JPEG_OPTIMIZE);
    encode_params.push(0);

    let mut current_detections = Vec::with_capacity(5);
    let mut _frame_counter = 0u64;

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
            continue;
        }

        _frame_counter += 1;
        let timestamp = get_timestamp();

        current_detections.clear();

        // Basketball detection using orange color
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

        // Draw basketball detections
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

        // FPS display
        imgproc::put_text(
            &mut frame,
            &format!("Basketball Tracker - FPS: {:.0}", avg_fps),
            Point::new(10, 25),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        // Encode frame
        let mut buf = Vector::new();
        imgcodecs::imencode(".jpg", &frame, &mut buf, &encode_params)?;
        let jpeg_data = buf.to_vec();

        // Update state
        if let Ok(mut state_guard) = state.try_lock() {
            state_guard.frame_count += 1;
            state_guard.fps_basketball = avg_fps;

            state_guard.current_detections = current_detections.clone();
            for det in &current_detections {
                state_guard.add_detection(det.clone());
            }

            if let Ok(mut frame_guard) = state_guard.basketball_frame.try_write() {
                *frame_guard = jpeg_data;
            }
        }
    }
}

fn process_heatmap_camera(
    mut cam: videoio::VideoCapture,
    _camera_id: u8,
    state: Arc<Mutex<GameState>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame = Mat::default();
    let mut _gray = Mat::default();
    let mut fg_mask = Mat::default();

    let mut last_time = Instant::now();
    let mut frame_times = Vec::with_capacity(60);
    let mut player_id: u64 = 0;

    // Background subtractor for motion detection
    let mut bg_subtractor = opencv::video::create_background_subtractor_mog2(
        500,  // history
        16.0, // varThreshold
        true, // detectShadows
    )?;

    // Optimize JPEG encoding for heatmap
    let mut encode_params = Vector::<i32>::new();
    encode_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    encode_params.push(65);
    encode_params.push(imgcodecs::IMWRITE_JPEG_OPTIMIZE);
    encode_params.push(0);

    let mut _frame_counter = 0u64;

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

        _frame_counter += 1;
        let timestamp = get_timestamp();

        // Apply background subtraction
        opencv::video::BackgroundSubtractorTrait::apply(
            &mut bg_subtractor,
            &frame,
            &mut fg_mask,
            -1.0,
        )?;

        // Find contours for player detection
        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(
            &fg_mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut detected_players = Vec::new();

        // Process contours to find players
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;

            // Filter by area (adjust thresholds for your setup)
            // Players typically have area between 1000-10000 pixels
            if area > 1000.0 && area < 15000.0 {
                let moments = imgproc::moments(&contour, false)?;
                if moments.m00 != 0.0 {
                    let cx = (moments.m10 / moments.m00) as i32;
                    let cy = (moments.m01 / moments.m00) as i32;

                    player_id += 1;
                    detected_players.push(PlayerDetection {
                        id: player_id,
                        x: cx,
                        y: cy,
                        timestamp,
                    });

                    // Draw player detection
                    imgproc::circle(
                        &mut frame,
                        Point::new(cx, cy),
                        8,
                        Scalar::new(255.0, 0.0, 0.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            }
        }

        // Overlay heatmap visualization
        if let Ok(state_guard) = state.try_lock() {
            let heatmap = &state_guard.heatmap_grid;
            let max_heat = heatmap
                .iter()
                .flat_map(|row| row.iter())
                .max()
                .unwrap_or(&1);

            for y in 0..20 {
                for x in 0..20 {
                    let heat_value = heatmap[y][x];
                    if heat_value > 0 {
                        let intensity = (heat_value as f32 / *max_heat as f32 * 255.0) as f64;
                        let rect_x = (x * 64) as i32;
                        let rect_y = (y * 36) as i32;

                        imgproc::rectangle(
                            &mut frame,
                            opencv::core::Rect::new(rect_x, rect_y, 64, 36),
                            Scalar::new(0.0, 0.0, intensity, 0.0),
                            -1,
                            imgproc::LINE_8,
                            0,
                        )?;
                    }
                }
            }
        }

        // FPS display
        imgproc::put_text(
            &mut frame,
            &format!(
                "Heatmap Tracker - FPS: {:.0} | Players: {}",
                avg_fps,
                detected_players.len()
            ),
            Point::new(10, 25),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        // Encode frame
        let mut buf = Vector::new();
        imgcodecs::imencode(".jpg", &frame, &mut buf, &encode_params)?;
        let jpeg_data = buf.to_vec();

        // Update state
        if let Ok(mut state_guard) = state.try_lock() {
            state_guard.fps_heatmap = avg_fps;

            for player in detected_players {
                state_guard.add_player_position(player);
            }

            if let Ok(mut frame_guard) = state_guard.heatmap_frame.try_write() {
                *frame_guard = jpeg_data;
            }
        }
    }
}

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    let serial_ports = vec!["/dev/ttyUSB0", "/dev/ttyACM0"];

    for port in &serial_ports {
        if let Ok(file) = std::fs::OpenOptions::new().read(true).open(port) {
            println!("Listening to ESP32 on {}", port);
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

    // Parse ESP32 data
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
        // Store raw ESP32 data
        state_guard.shot_data.push(ShotData {
            id: shot_id,
            timestamp,
            result: result.clone(),
            backboard_hits,
            rim_hits,
            shot_type: shot_type.to_string(),
        });

        // Link ESP32 result with camera trajectory
        if let Some(waiting_shot_id) = state_guard.waiting_for_esp32 {
            if let Some(shot) = state_guard
                .completed_shots
                .iter_mut()
                .find(|s| s.shot_id == waiting_shot_id && s.result.is_none())
            {
                shot.result = Some(result.clone());
                shot.backboard_hits = backboard_hits;
                shot.rim_hits = rim_hits;
                shot.shot_type = Some(shot_type.to_string());

                println!("\n=======================================");
                println!("SHOT #{} COMPLETE", shot.shot_id);
                println!("=======================================");
                println!(
                    "Origin: ({}, {})",
                    shot.start_position.0, shot.start_position.1
                );
                println!("Arc Quality: {:.1}%", shot.arc_quality * 100.0);
                println!("Shot Angle: {:.1} degrees", shot.shot_angle);
                println!(
                    "Flight Time: {}ms",
                    shot.trajectory_points.last().unwrap().2 - shot.timestamp
                );
                println!("Result: {}", result);
                println!("Type: {}", shot_type);
                if backboard_hits > 0 {
                    println!("Backboard Hits: {}", backboard_hits);
                }
                if rim_hits > 0 {
                    println!("Rim Hits: {}", rim_hits);
                }
                println!("=======================================\n");

                state_guard.waiting_for_esp32 = None;
            }
        }

        // Update counts
        if result == "MADE" {
            state_guard.makes_count += 1;
        } else if result == "MISSED" {
            state_guard.misses_count += 1;
        }
    }
}

#[allow(dead_code)]
fn analyze_basketball_makes(state: Arc<Mutex<GameState>>) {
    let mut shot_id_counter = 0u64;
    let mut last_detection_time = 0u64;

    loop {
        thread::sleep(Duration::from_millis(50));

        if let Ok(mut state_guard) = state.try_lock() {
            let now = get_timestamp();

            // Get recent detections
            let recent_detections: Vec<Detection> = state_guard
                .detection_history
                .iter()
                .filter(|d| now - d.timestamp < 1000)
                .cloned()
                .collect();

            // Start new trajectory if we detect ball after a gap
            if !recent_detections.is_empty() {
                let latest = recent_detections.last().unwrap();

                if now - last_detection_time > 500 && state_guard.active_trajectories.is_empty() {
                    shot_id_counter += 1;
                    let new_trajectory =
                        ShotTrajectory::new(shot_id_counter, latest.x, latest.y, latest.timestamp);
                    state_guard.active_trajectories.push(new_trajectory);
                    println!(
                        "Shot #{} started at ({}, {})",
                        shot_id_counter, latest.x, latest.y
                    );
                }

                // Update active trajectories
                for trajectory in &mut state_guard.active_trajectories {
                    for det in &recent_detections {
                        if det.timestamp > trajectory.trajectory_points.last().unwrap().2 {
                            trajectory.add_point(det.x, det.y, det.timestamp);
                        }
                    }
                }

                last_detection_time = now;
            }

            // Check if any trajectory is approaching rim
            let mut completed_indices = Vec::new();
            let mut completed_shots_to_add = Vec::new();

            for (idx, trajectory) in state_guard.active_trajectories.iter().enumerate() {
                if trajectory.approaching_rim && (now - last_detection_time > 300) {
                    println!(
                        "Shot #{} approaching rim - waiting for ESP32...",
                        trajectory.id
                    );

                    let completed_shot = CompletedShot {
                        shot_id: trajectory.id,
                        timestamp: trajectory.start_time,
                        start_position: trajectory.start_position,
                        arc_quality: trajectory.arc_quality,
                        shot_angle: trajectory.shot_angle,
                        trajectory_points: trajectory.trajectory_points.clone(),
                        result: None,
                        backboard_hits: 0,
                        rim_hits: 0,
                        shot_type: None,
                    };

                    completed_shots_to_add.push((trajectory.id, completed_shot));
                    completed_indices.push(idx);
                }

                // Timeout old trajectories
                if now - trajectory.trajectory_points.last().unwrap().2 > 2000 {
                    println!("Shot #{} timed out (lost tracking)", trajectory.id);
                    completed_indices.push(idx);
                }
            }

            // Add completed shots to state
            for (shot_id, completed_shot) in completed_shots_to_add {
                state_guard.completed_shots.push(completed_shot);
                state_guard.waiting_for_esp32 = Some(shot_id);
            }

            // Remove completed trajectories
            for idx in completed_indices.iter().rev() {
                state_guard.active_trajectories.remove(*idx);
            }
        }
    }
}

fn start_api_server(
    state: Arc<Mutex<GameState>>,
    basketball_frame_lock: Arc<RwLock<Vec<u8>>>,
    heatmap_frame_lock: Arc<RwLock<Vec<u8>>>,
) {
    let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind");
    println!("API Server started");

    for stream in listener.incoming() {
        if let Ok(stream) = stream {
            let state_clone = Arc::clone(&state);
            let basketball_frame_clone = Arc::clone(&basketball_frame_lock);
            let heatmap_frame_clone = Arc::clone(&heatmap_frame_lock);
            thread::spawn(move || {
                handle_api_request(
                    stream,
                    state_clone,
                    basketball_frame_clone,
                    heatmap_frame_clone,
                );
            });
        }
    }
}

fn handle_api_request(
    mut stream: TcpStream,
    state: Arc<Mutex<GameState>>,
    basketball_frame_lock: Arc<RwLock<Vec<u8>>>,
    heatmap_frame_lock: Arc<RwLock<Vec<u8>>>,
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

    if request_line.starts_with("GET /api/stream/basketball") {
        send_camera_stream(&mut stream, basketball_frame_lock);
    } else if request_line.starts_with("GET /api/stream/heatmap") {
        send_camera_stream(&mut stream, heatmap_frame_lock);
    } else if request_line.starts_with("GET /api/status") {
        send_status(&mut stream, state, cors);
    } else if request_line.starts_with("GET /api/shots") {
        send_shots(&mut stream, state, cors);
    } else if request_line.starts_with("GET /api/analytics") {
        send_analytics(&mut stream, state, cors);
    } else if request_line.starts_with("GET /api/heatmap") {
        send_heatmap_data(&mut stream, state, cors);
    } else {
        let _ = stream.write_all(format!("HTTP/1.1 404\r\n{}\r\n", cors).as_bytes());
    }
}

fn send_heatmap_data(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let json = json!({
            "heatmap_grid": state_guard.heatmap_grid,
            "total_positions": state_guard.player_positions.len(),
            "grid_size": [20, 20]
        });
        send_json_response(stream, &json, cors);
    }
}

fn send_analytics(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let recent_shots: Vec<&CompletedShot> =
            state_guard.completed_shots.iter().rev().take(10).collect();

        let json = json!({
            "total_shots": state_guard.makes_count + state_guard.misses_count,
            "makes": state_guard.makes_count,
            "misses": state_guard.misses_count,
            "accuracy": if state_guard.makes_count + state_guard.misses_count > 0 {
                format!("{:.1}%", (state_guard.makes_count as f32 / (state_guard.makes_count + state_guard.misses_count) as f32) * 100.0)
            } else { "0.0%".to_string() },
            "recent_shots": recent_shots.iter().map(|shot| {
                json!({
                    "id": shot.shot_id,
                    "start_x": shot.start_position.0,
                    "start_y": shot.start_position.1,
                    "arc_quality": format!("{:.1}%", shot.arc_quality * 100.0),
                    "angle": format!("{:.1} degrees", shot.shot_angle),
                    "result": shot.result,
                    "shot_type": shot.shot_type,
                    "backboard_hits": shot.backboard_hits,
                    "rim_hits": shot.rim_hits,
                })
            }).collect::<Vec<_>>()
        });
        send_json_response(stream, &json, cors);
    }
}

fn send_camera_stream(stream: &mut TcpStream, frame_lock: Arc<RwLock<Vec<u8>>>) {
    let _ = stream.set_nodelay(true);
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

        thread::sleep(Duration::from_millis(16)); // ~60 FPS max send rate
    }
}

fn send_status(stream: &mut TcpStream, state: Arc<Mutex<GameState>>, cors: &str) {
    if let Ok(state_guard) = state.lock() {
        let json = json!({
            "status": "running",
            "fps_basketball": format!("{:.1}", state_guard.fps_basketball),
            "fps_heatmap": format!("{:.1}", state_guard.fps_heatmap),
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
