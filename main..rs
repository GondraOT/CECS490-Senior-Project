// CECS490 Senior Project - Hoop IQ
// Team 2
// Christopher Hong, Gondra Kelly, Matthew "god" Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// Logitech C922x (Player Heatmap) and C920 (Basketball Tracker) Dual Camera Integration

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
use std::io::{prelude::*, BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// Performance Configuration
const BASKETBALL_DETECT_EVERY_N_FRAMES: u64 = 2;
const HEATMAP_DETECT_EVERY_N_FRAMES: u64 = 1;
const JPEG_QUALITY_BASKETBALL: i32 = 65;
const JPEG_QUALITY_HEATMAP: i32 = 55;
const MAX_TRAJECTORY_POINTS: usize = 50;
const TRAJECTORY_FADE_TIME_MS: u64 = 3000;

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
struct TrajectoryPoint {
    x: i32,
    y: i32,
    timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShotTrajectory {
    id: u64,
    points: VecDeque<TrajectoryPoint>,
    start_time: u64,
    active: bool,
}

impl ShotTrajectory {
    fn new(id: u64, x: i32, y: i32, timestamp: u64) -> Self {
        let mut points = VecDeque::new();
        points.push_back(TrajectoryPoint { x, y, timestamp });

        Self {
            id,
            points,
            start_time: timestamp,
            active: true,
        }
    }

    fn add_point(&mut self, x: i32, y: i32, timestamp: u64) {
        self.points.push_back(TrajectoryPoint { x, y, timestamp });

        if self.points.len() > MAX_TRAJECTORY_POINTS {
            self.points.pop_front();
        }
    }

    fn should_expire(&self, current_time: u64) -> bool {
        current_time - self.start_time > TRAJECTORY_FADE_TIME_MS
    }

    fn get_color_for_age(&self, point_timestamp: u64, current_time: u64) -> Scalar {
        let age = current_time.saturating_sub(point_timestamp) as f64;
        let max_age = TRAJECTORY_FADE_TIME_MS as f64;
        let alpha = 1.0 - (age / max_age).min(1.0);

        let blue = 255.0;
        let green = (255.0 * alpha) as f64;
        let red = 0.0;

        Scalar::new(blue, green, red, alpha * 255.0)
    }
}

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
struct HeatmapPoint {
    x: i32,
    y: i32,
    intensity: u32,
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
struct ShotData {
    id: u32,
    timestamp: u64,
    result: String,
    backboard_hits: u32,
    rim_hits: u32,
}

struct GameState {
    // Basketball tracking (C920)
    basketball_frame_count: u64,
    basketball_fps: f32,
    current_detection: Option<Detection>,
    active_trajectories: Vec<ShotTrajectory>,
    basketball_frame: Arc<RwLock<Vec<u8>>>,

    // Player heatmap (C922x)
    heatmap_frame_count: u64,
    heatmap_fps: f32,
    current_players: Vec<PlayerDetection>,
    heatmap_data: HashMap<(i32, i32), u32>,
    heatmap_frame: Arc<RwLock<Vec<u8>>>,

    // Shared data
    basketball_makes: Vec<BasketballMake>,
    shot_data: Vec<ShotData>,
    session_start: u64,
    make_count: u64,
    backboard_count: u32,
    rim_count: u32,
    total_players_detected: u64,
}

impl GameState {
    fn new() -> Self {
        Self {
            basketball_frame_count: 0,
            basketball_fps: 0.0,
            current_detection: None,
            active_trajectories: Vec::new(),
            basketball_frame: Arc::new(RwLock::new(Vec::new())),

            heatmap_frame_count: 0,
            heatmap_fps: 0.0,
            current_players: Vec::new(),
            heatmap_data: HashMap::new(),
            heatmap_frame: Arc::new(RwLock::new(Vec::new())),

            basketball_makes: Vec::new(),
            shot_data: Vec::new(),
            session_start: get_timestamp(),
            make_count: 0,
            backboard_count: 0,
            rim_count: 0,
            total_players_detected: 0,
        }
    }

    fn update_trajectory(&mut self, detection: &Detection) {
        let current_time = detection.timestamp;

        self.active_trajectories
            .retain(|traj| !traj.should_expire(current_time));

        if let Some(traj) = self.active_trajectories.last_mut() {
            if traj.active {
                traj.add_point(detection.x, detection.y, detection.timestamp);
            } else {
                let new_id = self.active_trajectories.len() as u64;
                let new_traj =
                    ShotTrajectory::new(new_id, detection.x, detection.y, detection.timestamp);
                self.active_trajectories.push(new_traj);
            }
        } else {
            let new_traj = ShotTrajectory::new(0, detection.x, detection.y, detection.timestamp);
            self.active_trajectories.push(new_traj);
        }
    }

    fn mark_trajectory_complete(&mut self) {
        if let Some(traj) = self.active_trajectories.last_mut() {
            traj.active = false;
        }
    }

    fn update_heatmap(&mut self, players: &[PlayerDetection]) {
        for player in players {
            let grid_x = (player.x / 20) * 20;
            let grid_y = (player.y / 20) * 20;

            let key = (grid_x, grid_y);
            *self.heatmap_data.entry(key).or_insert(0) += 1;
        }

        if self.heatmap_data.len() > 1000 {
            let min_intensity = 5;
            self.heatmap_data.retain(|_, &mut v| v >= min_intensity);
        }
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn send_to_cloud_api(state: Arc<Mutex<GameState>>, api_url: String) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::new();
        println!("Cloud API updater started");
        println!("Sending to: {}/update", api_url);

        loop {
            thread::sleep(Duration::from_secs(1));

            let (
                basketball_fps,
                makes,
                trajectories,
                backboard_hits,
                rim_hits,
                heatmap_fps,
                current_players,
                total_detected,
                heatmap_points,
                basketball_frame_data,
                heatmap_frame_data,
                timestamp,
            ) = {
                let state_lock = state.lock().unwrap();

                let b_frame = state_lock.basketball_frame.read().unwrap().clone();
                let h_frame = state_lock.heatmap_frame.read().unwrap().clone();

                (
                    state_lock.basketball_fps,
                    state_lock.make_count,
                    state_lock.active_trajectories.len() as u64,
                    state_lock.backboard_count,
                    state_lock.rim_count,
                    state_lock.heatmap_fps,
                    state_lock.current_players.len() as u64,
                    state_lock.total_players_detected,
                    state_lock.heatmap_data.len() as u64,
                    b_frame,
                    h_frame,
                    get_timestamp(),
                )
            };

            if basketball_frame_data.is_empty() || heatmap_frame_data.is_empty() {
                continue;
            }

            let basketball_b64 = general_purpose::STANDARD.encode(&basketball_frame_data);
            let heatmap_b64 = general_purpose::STANDARD.encode(&heatmap_frame_data);

            let payload = json!({
                "basketball_fps": basketball_fps,
                "makes": makes,
                "trajectories": trajectories,
                "backboard_hits": backboard_hits,
                "rim_hits": rim_hits,
                "heatmap_fps": heatmap_fps,
                "current_players": current_players,
                "total_players_detected": total_detected,
                "heatmap_points": heatmap_points,
                "basketball_frame": basketball_b64,
                "heatmap_frame": heatmap_b64,
                "timestamp": timestamp
            });

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

fn detect_basketball(frame: &Mat) -> Result<Vec<Detection>, opencv::Error> {
    let mut detections = Vec::new();
    let timestamp = get_timestamp();

    let mut hsv = Mat::default();
    imgproc::cvt_color(frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    let lower_orange = core::Scalar::new(5.0, 100.0, 100.0, 0.0);
    let upper_orange = core::Scalar::new(20.0, 255.0, 255.0, 0.0);

    let mut mask = Mat::default();
    core::in_range(&hsv, &lower_orange, &upper_orange, &mut mask)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut temp_mask = Mat::default();
    imgproc::morphology_ex(
        &mask,
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
        &mut mask,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        2,
        BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &mask,
        &mut blurred,
        Size::new(9, 9),
        2.0,
        2.0,
        BORDER_DEFAULT,
    )?;

    let mut circles = Mat::default();
    imgproc::hough_circles(
        &blurred,
        &mut circles,
        imgproc::HOUGH_GRADIENT,
        1.0,
        50.0,
        100.0,
        30.0,
        20,
        150,
    )?;

    for i in 0..circles.cols() {
        let circle_data = circles.at_2d::<core::Vec3f>(0, i)?;
        let x = circle_data[0] as i32;
        let y = circle_data[1] as i32;
        let radius = circle_data[2] as i32;

        let confidence = (radius as f32 / 100.0).min(1.0);

        detections.push(Detection {
            id: timestamp,
            camera_id: 1,
            x,
            y,
            radius,
            timestamp,
            confidence,
        });
    }

    Ok(detections)
}

fn detect_players(frame: &Mat) -> Result<Vec<PlayerDetection>, opencv::Error> {
    let mut players = Vec::new();
    let timestamp = get_timestamp();

    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Use MOG2 background subtractor from video module
    let mut background_subtractor = video::create_background_subtractor_mog2(500, 16.0, false)?;
    let mut fg_mask = Mat::default();
    video::BackgroundSubtractorTrait::apply(&mut background_subtractor, &gray, &mut fg_mask, -1.0)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut temp_fg_mask = Mat::default();
    imgproc::morphology_ex(
        &fg_mask,
        &mut temp_fg_mask,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &temp_fg_mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;

        if area > 500.0 && area < 15000.0 {
            let rect = imgproc::bounding_rect(&contour)?;

            let aspect_ratio = rect.height as f32 / rect.width as f32;
            if aspect_ratio > 1.2 && aspect_ratio < 4.0 {
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

fn draw_trajectory_overlay(frame: &mut Mat, state: &GameState) -> Result<(), opencv::Error> {
    let current_time = get_timestamp();

    for trajectory in &state.active_trajectories {
        let points = &trajectory.points;

        if points.len() < 2 {
            continue;
        }

        for i in 0..points.len() {
            let point = &points[i];
            let color = trajectory.get_color_for_age(point.timestamp, current_time);

            let age = current_time.saturating_sub(point.timestamp) as f64;
            let max_age = TRAJECTORY_FADE_TIME_MS as f64;
            let size_factor = 1.0 - (age / max_age).min(1.0);
            let radius = (3.0 + size_factor * 5.0) as i32;

            imgproc::circle(
                frame,
                Point::new(point.x, point.y),
                radius,
                color,
                -1,
                imgproc::LINE_AA,
                0,
            )?;

            if i > 0 {
                let prev_point = &points[i - 1];
                imgproc::line(
                    frame,
                    Point::new(prev_point.x, prev_point.y),
                    Point::new(point.x, point.y),
                    color,
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;
            }
        }

        if let Some(first_point) = points.front() {
            let text = format!("Trajectory {}", trajectory.id);
            imgproc::put_text(
                frame,
                &text,
                Point::new(first_point.x + 10, first_point.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 255.0, 255.0, 255.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
    }

    Ok(())
}

fn draw_basketball_overlay(
    frame: &mut Mat,
    detection: &Detection,
    state: &GameState,
) -> Result<(), opencv::Error> {
    imgproc::circle(
        frame,
        Point::new(detection.x, detection.y),
        detection.radius,
        Scalar::new(0.0, 255.0, 0.0, 255.0),
        3,
        imgproc::LINE_AA,
        0,
    )?;

    imgproc::circle(
        frame,
        Point::new(detection.x, detection.y),
        5,
        Scalar::new(0.0, 0.0, 255.0, 255.0),
        -1,
        imgproc::LINE_AA,
        0,
    )?;

    let info_text = format!(
        "Ball: ({}, {}) r={}",
        detection.x, detection.y, detection.radius
    );
    imgproc::put_text(
        frame,
        &info_text,
        Point::new(detection.x + detection.radius + 10, detection.y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(0.0, 255.0, 0.0, 255.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    let stats_y_start = 30;
    let stats_x = 10;

    imgproc::put_text(
        frame,
        &format!("C920 Basketball Tracker"),
        Point::new(stats_x, stats_y_start),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("FPS: {:.1}", state.basketball_fps),
        Point::new(stats_x, stats_y_start + 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Makes: {}", state.make_count),
        Point::new(stats_x, stats_y_start + 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(0.0, 255.0, 0.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Trajectories: {}", state.active_trajectories.len()),
        Point::new(stats_x, stats_y_start + 90),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(0.0, 255.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

fn draw_heatmap_overlay(frame: &mut Mat, state: &GameState) -> Result<(), opencv::Error> {
    let mut heatmap_mat =
        Mat::zeros(frame.rows(), frame.cols(), opencv::core::CV_8UC3)?.to_mat()?;

    let max_intensity = state.heatmap_data.values().max().copied().unwrap_or(1);

    for (&(x, y), &intensity) in &state.heatmap_data {
        let normalized = (intensity as f64 / max_intensity as f64).min(1.0);

        let hue = ((1.0 - normalized) * 120.0) as i32;
        let color = Scalar::new(hue as f64, 255.0, 255.0, 255.0);

        imgproc::circle(
            &mut heatmap_mat,
            Point::new(x, y),
            15,
            color,
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    let mut heatmap_bgr = Mat::default();
    imgproc::cvt_color(&heatmap_mat, &mut heatmap_bgr, imgproc::COLOR_HSV2BGR, 0)?;

    let mut blurred_heatmap = Mat::default();
    imgproc::gaussian_blur(
        &heatmap_bgr,
        &mut blurred_heatmap,
        Size::new(25, 25),
        0.0,
        0.0,
        BORDER_DEFAULT,
    )?;

    let mut output_frame = Mat::default();
    core::add_weighted(
        frame,
        0.7,
        &blurred_heatmap,
        0.3,
        0.0,
        &mut output_frame,
        -1,
    )?;
    output_frame.copy_to(frame)?;

    Ok(())
}

fn draw_player_overlay(
    frame: &mut Mat,
    players: &[PlayerDetection],
    state: &GameState,
) -> Result<(), opencv::Error> {
    for player in players {
        imgproc::rectangle(
            frame,
            Rect::new(
                player.x - player.width / 2,
                player.y - player.height / 2,
                player.width,
                player.height,
            ),
            Scalar::new(255.0, 0.0, 255.0, 255.0),
            2,
            imgproc::LINE_AA,
            0,
        )?;

        imgproc::circle(
            frame,
            Point::new(player.x, player.y),
            5,
            Scalar::new(255.0, 0.0, 255.0, 255.0),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }

    let stats_y_start = 30;
    let stats_x = 10;

    imgproc::put_text(
        frame,
        &format!("C922x Player Heatmap"),
        Point::new(stats_x, stats_y_start),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("FPS: {:.1}", state.heatmap_fps),
        Point::new(stats_x, stats_y_start + 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Players: {}", players.len()),
        Point::new(stats_x, stats_y_start + 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 0.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Total Detected: {}", state.total_players_detected),
        Point::new(stats_x, stats_y_start + 90),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 0.0, 255.0, 255.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

fn process_basketball_camera(
    mut camera: videoio::VideoCapture,
    state: Arc<Mutex<GameState>>,
) -> Result<(), opencv::Error> {
    let mut frame = Mat::default();
    let mut frame_times = VecDeque::new();

    println!("C920 basketball tracking started");

    loop {
        camera.read(&mut frame)?;
        if frame.empty() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let mut state_lock = state.lock().unwrap();
        state_lock.basketball_frame_count += 1;

        if state_lock.basketball_frame_count % BASKETBALL_DETECT_EVERY_N_FRAMES == 0 {
            match detect_basketball(&frame) {
                Ok(detections) => {
                    if let Some(detection) = detections.first() {
                        state_lock.current_detection = Some(detection.clone());
                        state_lock.update_trajectory(detection);
                    } else {
                        state_lock.current_detection = None;
                    }
                }
                Err(e) => eprintln!("Basketball detection error: {}", e),
            }
        }

        if let Err(e) = draw_trajectory_overlay(&mut frame, &state_lock) {
            eprintln!("Trajectory overlay error: {}", e);
        }

        if let Some(ref detection) = state_lock.current_detection {
            if let Err(e) = draw_basketball_overlay(&mut frame, detection, &state_lock) {
                eprintln!("Basketball overlay error: {}", e);
            }
        }

        frame_times.push_back(Instant::now());
        if frame_times.len() > 30 {
            frame_times.pop_front();
        }
        if frame_times.len() >= 2 {
            let duration = frame_times
                .back()
                .unwrap()
                .duration_since(*frame_times.front().unwrap());
            state_lock.basketball_fps = (frame_times.len() - 1) as f32 / duration.as_secs_f32();
        }

        // Clone Arc before dropping lock - encode JPEG outside mutex
        let basketball_frame_arc = state_lock.basketball_frame.clone();
        drop(state_lock);

        let mut buf = Vector::<u8>::new();
        let params =
            Vector::<i32>::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, JPEG_QUALITY_BASKETBALL]);
        if imgcodecs::imencode(".jpg", &frame, &mut buf, &params).is_ok() {
            if let Ok(mut f) = basketball_frame_arc.write() {
                *f = buf.to_vec();
            }
        }

        thread::sleep(Duration::from_millis(33));
    }
}

fn process_heatmap_camera(
    mut camera: videoio::VideoCapture,
    state: Arc<Mutex<GameState>>,
) -> Result<(), opencv::Error> {
    let mut frame = Mat::default();
    let mut frame_times = VecDeque::new();

    println!("C922x player heatmap tracking started");

    loop {
        camera.read(&mut frame)?;
        if frame.empty() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let mut state_lock = state.lock().unwrap();
        state_lock.heatmap_frame_count += 1;

        if state_lock.heatmap_frame_count % HEATMAP_DETECT_EVERY_N_FRAMES == 0 {
            match detect_players(&frame) {
                Ok(players) => {
                    state_lock.total_players_detected += players.len() as u64;
                    state_lock.update_heatmap(&players);
                    state_lock.current_players = players.clone();
                }
                Err(e) => eprintln!("Player detection error: {}", e),
            }
        }

        if let Err(e) = draw_heatmap_overlay(&mut frame, &state_lock) {
            eprintln!("Heatmap overlay error: {}", e);
        }

        if let Err(e) = draw_player_overlay(&mut frame, &state_lock.current_players, &state_lock) {
            eprintln!("Player overlay error: {}", e);
        }

        frame_times.push_back(Instant::now());
        if frame_times.len() > 30 {
            frame_times.pop_front();
        }
        if frame_times.len() >= 2 {
            let duration = frame_times
                .back()
                .unwrap()
                .duration_since(*frame_times.front().unwrap());
            state_lock.heatmap_fps = (frame_times.len() - 1) as f32 / duration.as_secs_f32();
        }

        // Clone Arc before dropping lock - encode JPEG outside mutex
        let heatmap_frame_arc = state_lock.heatmap_frame.clone();
        drop(state_lock);

        let mut buf = Vector::<u8>::new();
        let params =
            Vector::<i32>::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, JPEG_QUALITY_HEATMAP]);
        if imgcodecs::imencode(".jpg", &frame, &mut buf, &params).is_ok() {
            if let Ok(mut f) = heatmap_frame_arc.write() {
                *f = buf.to_vec();
            }
        }

        thread::sleep(Duration::from_millis(16));
    }
}

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    thread::spawn(move || {
        let serial_ports = vec!["/dev/ttyUSB0", "/dev/ttyACM0"];

        for port in &serial_ports {
            println!("Attempting to connect to ESP32 on {}", port);

            if let Ok(file) = std::fs::OpenOptions::new().read(true).open(port) {
                println!("Connected to ESP32 on {}", port);
                println!("Listening for sensor data...");

                let reader = BufReader::new(file);

                for line in reader.lines() {
                    if let Ok(data) = line {
                        let data = data.trim();

                        if data.is_empty() {
                            continue;
                        }

                        let mut state_lock = state.lock().unwrap();

                        if data.starts_with("MAKE:") {
                            let id = data
                                .split(':')
                                .nth(1)
                                .and_then(|s| s.parse::<u64>().ok())
                                .unwrap_or(0);
                            println!("Basketball make detected: ID {}", id);

                            state_lock.make_count += 1;
                            state_lock.mark_trajectory_complete();

                            let make = BasketballMake {
                                id,
                                timestamp: get_timestamp(),
                                confidence: 1.0,
                                x: 0,
                                y: 0,
                            };
                            state_lock.basketball_makes.push(make);
                        } else if data.starts_with("BACK:") {
                            state_lock.backboard_count += 1;
                            println!(
                                "Backboard hit detected: Total {}",
                                state_lock.backboard_count
                            );
                        } else if data.starts_with("RIM:") {
                            state_lock.rim_count += 1;
                            println!("Rim hit detected: Total {}", state_lock.rim_count);
                        }

                        drop(state_lock);
                    }
                }

                println!("ESP32 disconnected on {}", port);
            }
        }

        eprintln!("ESP32 not found on any port");
    });
}

fn stream_mjpeg(mut stream: TcpStream, state: Arc<Mutex<GameState>>, is_basketball: bool) {
    // Disable Nagle's algorithm - send frames immediately without buffering
    let _ = stream.set_nodelay(true);
    // Drop frozen/disconnected clients after 5 seconds
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }

    let label = if is_basketball {
        "basketball"
    } else {
        "heatmap"
    };

    loop {
        let frame_data = {
            let state_lock = state.lock().unwrap();
            let data = if is_basketball {
                state_lock.basketball_frame.read().unwrap().clone()
            } else {
                state_lock.heatmap_frame.read().unwrap().clone()
            };
            drop(state_lock);
            data
        };

        if !frame_data.is_empty() {
            let frame_header = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                frame_data.len()
            );
            if stream.write_all(frame_header.as_bytes()).is_err()
                || stream.write_all(&frame_data).is_err()
                || stream.write_all(b"\r\n").is_err()
                || stream.flush().is_err()
            {
                println!("{} stream client disconnected", label);
                return;
            }
        }

        // 50ms = ~20 FPS - more stable over Tailscale than 33ms
        thread::sleep(Duration::from_millis(50));
    }
}

fn handle_http_request(mut stream: TcpStream, state: Arc<Mutex<GameState>>) -> std::io::Result<()> {
    let mut buffer = [0u8; 1024];
    stream.read(&mut buffer)?;

    let request = String::from_utf8_lossy(&buffer);
    let path = request
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();

    if path == "/api/stream/basketball" {
        stream_mjpeg(stream, state, true);
        return Ok(());
    } else if path == "/api/stream/heatmap" {
        stream_mjpeg(stream, state, false);
        return Ok(());
    } else if path == "/api/stats" {
        let state_lock = state.lock().unwrap();
        let stats = json!({
            "basketball": {
                "fps": state_lock.basketball_fps,
                "makes": state_lock.make_count,
                "trajectories": state_lock.active_trajectories.len(),
            },
            "heatmap": {
                "fps": state_lock.heatmap_fps,
                "current_players": state_lock.current_players.len(),
                "total_detected": state_lock.total_players_detected,
                "heatmap_points": state_lock.heatmap_data.len(),
            },
            "sensors": {
                "backboard_hits": state_lock.backboard_count,
                "rim_hits": state_lock.rim_count,
            }
        });

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            stats
        );
        stream.write_all(response.as_bytes())?;
    } else {
        let response = "HTTP/1.1 404 NOT FOUND\r\n\r\n";
        stream.write_all(response.as_bytes())?;
    }

    Ok(())
}

fn start_api_server(state: Arc<Mutex<GameState>>) {
    thread::spawn(move || {
        let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind to port 8080");
        println!("API Server started");
        println!("Basketball Stream: http://0.0.0.0:8080/api/stream/basketball");
        println!("Heatmap Stream: http://0.0.0.0:8080/api/stream/heatmap");
        println!("Stats: http://0.0.0.0:8080/api/stats");

        for stream in listener.incoming() {
            if let Ok(stream) = stream {
                let state_clone = Arc::clone(&state);
                thread::spawn(move || {
                    let _ = handle_http_request(stream, state_clone);
                });
            }
        }
    });
}

fn main() -> Result<(), opencv::Error> {
    println!("========================================");
    println!("HOOP IQ - Dual Camera System");
    println!("C920: Basketball Detection + Trajectory");
    println!("C922x: Player Heatmap Tracking");
    println!("CECS490 Senior Project - Team 2");
    println!("========================================");
    println!("");

    let game_state = Arc::new(Mutex::new(GameState::new()));

    // Open C920 for basketball tracking with retry logic
    println!("Opening C920 basketball camera...");
    let mut cam_basketball = None;
    for attempt in 1..=5 {
        match videoio::VideoCapture::from_file("/dev/video2", videoio::CAP_V4L2) {
            Ok(cam) => {
                if videoio::VideoCapture::is_opened(&cam).unwrap_or(false) {
                    cam_basketball = Some(cam);
                    println!("C920 initialized successfully on attempt {}", attempt);
                    break;
                }
            }
            Err(e) => {
                eprintln!("C920 attempt {} error: {}", attempt, e);
            }
        }
        if attempt < 5 {
            println!("C920 attempt {} failed, retrying in 2 seconds...", attempt);
            thread::sleep(Duration::from_secs(2));
        }
    }

    let cam_basketball = match cam_basketball {
        Some(cam) => cam,
        None => {
            eprintln!("Could not open C920 basketball camera on /dev/video2 after 5 attempts");
            eprintln!("Try: ls /dev/video* to find correct camera");
            return Ok(());
        }
    };

    // Open C922x for player heatmap with retry logic
    println!("Opening C922x heatmap camera...");
    let mut cam_heatmap = None;
    for attempt in 1..=5 {
        match videoio::VideoCapture::from_file("/dev/video0", videoio::CAP_V4L2) {
            Ok(cam) => {
                if videoio::VideoCapture::is_opened(&cam).unwrap_or(false) {
                    cam_heatmap = Some(cam);
                    println!("C922x initialized successfully on attempt {}", attempt);
                    break;
                }
            }
            Err(e) => {
                eprintln!("C922x attempt {} error: {}", attempt, e);
            }
        }
        if attempt < 5 {
            println!("C922x attempt {} failed, retrying in 2 seconds...", attempt);
            thread::sleep(Duration::from_secs(2));
        }
    }

    let cam_heatmap = match cam_heatmap {
        Some(cam) => cam,
        None => {
            eprintln!("Could not open C922x heatmap camera on /dev/video0 after 5 attempts");
            eprintln!("Try: ls /dev/video* to find correct camera");
            return Ok(());
        }
    };

    println!("");

    // Start ESP32 listener
    listen_to_esp32(Arc::clone(&game_state));

    // Start API server
    start_api_server(Arc::clone(&game_state));

    println!("Starting camera processing threads...");

    let cloud_api_url = "https://cecs490-senior-project.onrender.com".to_string();
    //let cloud_api_url = "https://api.hoopiq.shop".to_string();
    send_to_cloud_api(Arc::clone(&game_state), cloud_api_url);
    println!("Cloud API updater started");

    // Start basketball camera thread
    let state_basketball = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_basketball_camera(cam_basketball, state_basketball) {
            eprintln!("Basketball camera error: {}", e);
        }
    });

    // Start heatmap camera thread
    let state_heatmap = Arc::clone(&game_state);
    thread::spawn(move || {
        if let Err(e) = process_heatmap_camera(cam_heatmap, state_heatmap) {
            eprintln!("Heatmap camera error: {}", e);
        }
    });

    println!("All systems running!");
    println!("Press Ctrl+C to stop");

    // Keep main thread alive
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

}

