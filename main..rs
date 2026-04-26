// CECS490 Senior Project - Hoop IQ
// Team 2
// Christopher Hong, Gondra Kelly, Matthew "god" Marguiles, Alfonso Mejia Vasquez, Carlos Orozco
// C922x: single camera → two streams via v4l2loopback
//   heatmap  → Rust OpenCV ball detection overlay → RTSP → MediaMTX
//   basketball → raw frames via loopback → FFmpeg → RTSP → MediaMTX

use base64::{engine::general_purpose, Engine as _};
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector, BORDER_DEFAULT},
    imgcodecs, imgproc,
    prelude::*,
    video, videoio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Camera / frame tuning ─────────────────────────────────────────────
const HEATMAP_DETECT_EVERY_N_FRAMES: u64 = 2;
const JPEG_QUALITY_HEATMAP: i32 = 65;

// ── Ball HSV color range (tune at your venue) ─────────────────────────
const BALL_H_MIN: f64 = 8.0;
const BALL_H_MAX: f64 = 25.0;
const BALL_S_MIN: f64 = 90.0;
const BALL_S_MAX: f64 = 255.0;
const BALL_V_MIN: f64 = 50.0;
const BALL_V_MAX: f64 = 230.0;
const BALL_MIN_RADIUS_PX: i32 = 8;
const BALL_MAX_RADIUS_PX: i32 = 250;
const BALL_LOST_TIMEOUT_MS: u64 = 3000;

// ── Ball tracker reliability ──────────────────────────────────────────
const BALL_LOCK_CONFIRM_FRAMES: u32 = 1;
const BALL_ROI_RADIUS_PX: i32 = 400;
const BALL_MAX_SIZE_CHANGE_PCT: f32 = 0.70;
const BALL_MIN_CIRCULARITY: f64 = 0.40;
const BALL_MAX_ASPECT_DEVIATION: f32 = 0.50;

// ── Monocular depth estimation ────────────────────────────────────────
// A basketball has a fixed real-world diameter of 9.4 inches.
// We use this known size to estimate distance from the camera.
//
// Formula: distance = (real_diameter × focal_length) / pixel_diameter
//
// ONE-TIME CALIBRATION:
//   1. Hold ball exactly CALIB_DISTANCE_INCHES (36in = 3ft) from camera lens
//   2. Run: ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 /tmp/calib.jpg
//   3. Open calib.jpg, measure ball diameter in pixels (left edge to right edge)
//   4. Update CALIB_PIXEL_DIAMETER with that measurement
//   5. Rebuild — works in any environment from then on
const BALL_REAL_DIAMETER_INCHES: f64 = 9.4;
const CALIB_DISTANCE_INCHES: f64 = 36.0; // 3 feet — hold ball here during calib
const CALIB_PIXEL_DIAMETER: f64 = 124.0; // ← UPDATE AFTER CALIBRATION

// ── Shot zone threshold ───────────────────────────────────────────────
// 3PT line is 23.75ft = 285 inches from basket in NBA.
// Camera is placed near basket, so distance from camera ≈ distance from basket.
const THREE_PT_DISTANCE_INCHES: f64 = 333.0;

// ── Airball / shot release detection ─────────────────────────────────
const SHOT_RELEASE_VEL_Y: f32 = -220.0; // px/s upward to count as release
const SHOT_RELEASE_VEL_X_MAX: f32 = 300.0; // cap lateral speed (not a pass)
const AIRBALL_WAIT_MS: u64 = 5000; // 5 second window for ESP32 response
const AIRBALL_COOLDOWN_MS: u64 = 4000; // min gap between airball calls

// ── Structs ───────────────────────────────────────────────────────────

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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ShotDot {
    px: i32,
    py: i32,
    dist_ft: f64,
    is_three: bool,
    made: bool,
    shot_type: String,
    timestamp: u64,
}

#[derive(Clone, Debug)]
struct BallPosition {
    px: i32,
    py: i32,
    radius: i32,
    last_seen_ms: u64,
    distance_inches: f64,
    distance_ft: f64,
    is_three: bool,
    vel_x: f32,        // pixels/second, positive = right
    vel_y: f32,        // pixels/second, negative = moving up in frame
    smooth_vel_y: f32, // ← ADD: averaged over last N frames
}

#[derive(Clone, Debug, PartialEq)]
enum TrackerState {
    Searching,  // never had a lock — accept first good detection
    Locked,     // actively tracking the ball
    Recovering, // had lock, ball left frame — wait for it to return, reject everything else
}

impl Default for TrackerState {
    fn default() -> Self {
        TrackerState::Searching
    }
}

#[derive(Clone, Debug, Default)]
struct BallCandidate {
    px: i32,
    py: i32,
    radius: i32,
    confirm_count: u32,
    vel_y_history: Vec<f32>,
    state: TrackerState,
    lost_at_ms: u64,         // when lock was lost
    last_locked_radius: i32, // radius when last locked — used for re-acquisition
    lock_frame_count: u32,   // total frames locked — if high, we're confident
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

// ── GameState ─────────────────────────────────────────────────────────

struct GameState {
    heatmap_frame_count: u64,
    heatmap_fps: f32,
    current_players: Vec<PlayerDetection>,
    ball_position: Option<BallPosition>,
    shot_dots: Vec<ShotDot>,
    make_count: u64,
    backboard_count: u32,
    rim_count: u32,
    swish_count: u32,
    two_pt_makes: u32,
    two_pt_attempts: u32,
    three_pt_makes: u32,
    three_pt_attempts: u32,
    backboard_make_count: u32,
    backboard_miss_count: u32,
    total_players_detected: u64,
    shot_chart: Vec<ShotEntry>,
    last_shot_type: String,
    pending_zone: Option<bool>,
    airball_count: u32,
    last_serial_event_ms: u64,
}

impl GameState {
    fn new() -> Self {
        Self {
            heatmap_frame_count: 0,
            heatmap_fps: 0.0,
            current_players: Vec::new(),
            ball_position: None,
            shot_dots: Vec::new(),
            make_count: 0,
            backboard_count: 0,
            rim_count: 0,
            swish_count: 0,
            backboard_make_count: 0,
            backboard_miss_count: 0,
            two_pt_makes: 0,
            two_pt_attempts: 0,
            three_pt_makes: 0,
            three_pt_attempts: 0,
            total_players_detected: 0,
            shot_chart: Vec::new(),
            last_shot_type: "—".to_string(),
            pending_zone: None,
            airball_count: 0,
            last_serial_event_ms: 0,
        }
    }

    fn record_shot(&mut self, made: bool, shot_type: &str, frame_width: i32, frame_height: i32) {
        self.record_shot_with_zone(made, shot_type, frame_width, frame_height, None);
    }

    fn record_shot_with_zone(
        &mut self,
        made: bool,
        shot_type: &str,
        frame_width: i32,
        frame_height: i32,
        zone_override: Option<bool>,
    ) {
        let (px, py, distance_ft, is_three) = if let Some(ref ball) = self.ball_position {
            (
                ball.px,
                ball.py,
                ball.distance_ft,
                zone_override.unwrap_or(ball.is_three),
            )
        } else {
            (frame_width / 2, frame_height / 2, 15.0, false)
        };

        if is_three {
            self.three_pt_attempts += 1;
            if made {
                self.three_pt_makes += 1;
            }
        } else {
            self.two_pt_attempts += 1;
            if made {
                self.two_pt_makes += 1;
            }
        }

        self.shot_dots.push(ShotDot {
            px,
            py,
            dist_ft: distance_ft,
            is_three,
            made,
            shot_type: shot_type.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        });
        if self.shot_dots.len() > 500 {
            self.shot_dots.remove(0);
        }

        let zone = if is_three { "3PT" } else { "2PT" }.to_string();
        let x = (px as f32 / frame_width as f32 * 100.0).clamp(0.0, 100.0);
        let y = (py as f32 / frame_height as f32 * 100.0).clamp(0.0, 100.0);

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

        self.last_shot_type = shot_type.to_string();
    }

    fn attempts(&self) -> u64 {
        self.make_count + self.backboard_miss_count as u64 + self.airball_count as u64
    }

    fn fg_percent(&self) -> Option<f32> {
        let a = self.attempts();
        if a == 0 {
            None
        } else {
            Some(self.make_count as f32 / a as f32 * 100.0)
        }
    }
} // ← closes impl GameState

// ── Distance estimation ───────────────────────────────────────────────

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn focal_length() -> f64 {
    (CALIB_PIXEL_DIAMETER * CALIB_DISTANCE_INCHES) / BALL_REAL_DIAMETER_INCHES
}

fn estimate_distance_inches(pixel_diameter: f64) -> f64 {
    if pixel_diameter < 1.0 {
        return 9999.0;
    }
    (BALL_REAL_DIAMETER_INCHES * focal_length()) / pixel_diameter
}

fn distance_to_zone(distance_inches: f64) -> bool {
    // true = 3PT, false = 2PT
    distance_inches >= THREE_PT_DISTANCE_INCHES
}

// ── Ball detection ────────────────────────────────────────────────────

fn detect_ball(
    frame: &Mat,
    last_ball: &Option<BallPosition>,
    candidate: &mut BallCandidate,
) -> Result<Option<BallPosition>, opencv::Error> {
    let now_ms = get_timestamp();

    // ── HSV threshold ─────────────────────────────────────────────────
    let mut hsv = Mat::default();
    imgproc::cvt_color(frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    let lower = Scalar::new(BALL_H_MIN, BALL_S_MIN, BALL_V_MIN, 0.0);
    let upper = Scalar::new(BALL_H_MAX, BALL_S_MAX, BALL_V_MAX, 0.0);
    let mut mask = Mat::default();
    core::in_range(&hsv, &lower, &upper, &mut mask)?;

    // ── Morphology ────────────────────────────────────────────────────
    let k_small = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let k_large = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(15, 15),
        Point::new(-1, -1),
    )?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut opened,
        imgproc::MORPH_OPEN,
        &k_small,
        Point::new(-1, -1),
        1,
        BORDER_DEFAULT,
        core::Scalar::default(),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &opened,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &k_large,
        Point::new(-1, -1),
        3,
        BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    // ── Find contours ─────────────────────────────────────────────────
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &closed,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    // ── Score all candidates ──────────────────────────────────────────
    let mut best: Option<(f64, i32, i32, i32)> = None;

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = imgproc::contour_area(&contour, false)?;
        if area < 200.0 {
            continue;
        }

        let rect = imgproc::bounding_rect(&contour)?;
        let radius = ((rect.width + rect.height) / 4).max(1);
        if radius < BALL_MIN_RADIUS_PX || radius > BALL_MAX_RADIUS_PX {
            continue;
        }

        let perimeter = imgproc::arc_length(&contour, true)?;
        let circularity = if perimeter > 1.0 {
            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
        } else {
            0.1
        };

        let aspect = rect.width as f64 / rect.height.max(1) as f64;
        let aspect_score = 1.0 - (aspect - 1.0).abs().min(1.0);

        // ── RECOVERING: only accept blobs similar in size to last lock ─
        if candidate.state == TrackerState::Recovering {
            let expected = candidate.last_locked_radius as f64;
            let size_diff = (radius as f64 - expected).abs() / expected;
            // Must be within 40% of last known ball size to re-acquire
            if size_diff > 0.40 {
                continue;
            }
            // Must be reasonably circular to re-acquire
            if circularity < 0.35 {
                continue;
            }
        }

        // ── SEARCHING: stricter initial lock requirements ──────────────
        if candidate.state == TrackerState::Searching {
            // Require decent circularity before first lock
            if circularity < 0.30 {
                continue;
            }
        }

        let score = area * circularity * aspect_score;

        if best.is_none() || score > best.unwrap().0 {
            let cx = rect.x + rect.width / 2;
            let cy = rect.y + rect.height / 2;
            best = Some((score, cx, cy, radius));
        }
    }

    // ── State machine ─────────────────────────────────────────────────
    match best {
        Some((_, cx, cy, radius)) => {
            let dist_from_last = {
                let dx = cx - candidate.px;
                let dy = cy - candidate.py;
                ((dx * dx + dy * dy) as f32).sqrt()
            };

            // If locked and new detection is far away — suspect it's
            // a different object, not the ball jumping across the frame.
            // Only allow large jumps if ball was moving fast (high velocity).
            let max_jump = match candidate.state {
                TrackerState::Locked => {
                    let speed = last_ball
                        .as_ref()
                        .map(|b| (b.vel_x.abs() + b.vel_y.abs()))
                        .unwrap_or(0.0);
                    // Allow larger jump if ball was already moving fast
                    (150.0 + speed * 0.1).min(400.0) as f32
                }
                TrackerState::Recovering => 250.0, // wider re-acquisition window
                TrackerState::Searching => 400.0,  // wide initial search
            };

            if dist_from_last < max_jump || candidate.confirm_count == 0 {
                candidate.px = cx;
                candidate.py = cy;
                candidate.radius = radius;
                candidate.confirm_count += 1;
            } else {
                // Too far from last position — if locked, ignore this blob.
                // If searching/recovering, reset to this new position.
                match candidate.state {
                    TrackerState::Locked => {
                        // Refuse the jump — hold last known position
                        if let Some(ref last) = last_ball {
                            if now_ms.saturating_sub(last.last_seen_ms) < BALL_LOST_TIMEOUT_MS {
                                return Ok(Some(last.clone()));
                            }
                        }
                        // Timed out — go to recovering
                        candidate.state = TrackerState::Recovering;
                        candidate.lost_at_ms = now_ms;
                        candidate.confirm_count = 0;
                        return Ok(None);
                    }
                    _ => {
                        candidate.px = cx;
                        candidate.py = cy;
                        candidate.radius = radius;
                        candidate.confirm_count = 1;
                        candidate.vel_y_history.clear();
                    }
                }
            }

            let frames_needed = match candidate.state {
                TrackerState::Searching => 3,  // strict initial lock — 3 frames
                TrackerState::Recovering => 2, // faster re-acquisition
                TrackerState::Locked => 1,     // already locked, stay locked
            };

            if candidate.confirm_count >= frames_needed {
                // Transition to locked
                candidate.state = TrackerState::Locked;
                candidate.lock_frame_count += 1;
                candidate.last_locked_radius = radius;

                let (vel_x, vel_y) = if let Some(ref last) = last_ball {
                    let dt = (now_ms.saturating_sub(last.last_seen_ms)).max(1) as f32 / 1000.0;
                    ((cx - last.px) as f32 / dt, (cy - last.py) as f32 / dt)
                } else {
                    (0.0, 0.0)
                };

                candidate.vel_y_history.push(vel_y);
                if candidate.vel_y_history.len() > 2 {
                    candidate.vel_y_history.remove(0);
                }
                let smooth_vel_y = if candidate.vel_y_history.is_empty() {
                    vel_y
                } else {
                    candidate.vel_y_history.iter().sum::<f32>()
                        / candidate.vel_y_history.len() as f32
                };

                let pixel_diameter = (radius * 2) as f64;
                let distance_inches = estimate_distance_inches(pixel_diameter);
                let distance_ft = distance_inches / 12.0;
                let is_three = distance_to_zone(distance_inches);

                Ok(Some(BallPosition {
                    px: cx,
                    py: cy,
                    radius,
                    last_seen_ms: now_ms,
                    distance_inches,
                    distance_ft,
                    is_three,
                    vel_x,
                    vel_y,
                    smooth_vel_y,
                }))
            } else {
                // Still confirming — hold last if recent
                if let Some(ref last) = last_ball {
                    if now_ms.saturating_sub(last.last_seen_ms) < BALL_LOST_TIMEOUT_MS {
                        return Ok(Some(last.clone()));
                    }
                }
                Ok(None)
            }
        }

        None => {
            // No detection this frame
            match candidate.state {
                TrackerState::Locked => {
                    // Was locked — start recovering
                    candidate.state = TrackerState::Recovering;
                    candidate.lost_at_ms = now_ms;
                    candidate.confirm_count = 0;
                    // Hold last position for the timeout window
                    if let Some(ref last) = last_ball {
                        if now_ms.saturating_sub(last.last_seen_ms) < BALL_LOST_TIMEOUT_MS {
                            return Ok(Some(last.clone()));
                        }
                    }
                    Ok(None)
                }
                TrackerState::Recovering => {
                    // Still waiting for ball to return — hold nothing,
                    // refuse to lock onto anything new (handled above by size check)
                    candidate.vel_y_history.clear();
                    Ok(None)
                }
                TrackerState::Searching => {
                    candidate.confirm_count = 0;
                    candidate.vel_y_history.clear();
                    Ok(None)
                }
            }
        }
    }
}

// ── Draw overlays ─────────────────────────────────────────────────────

fn draw_ball_overlay(
    frame: &mut Mat,
    ball: &Option<BallPosition>,
    tracker_state: &TrackerState,
) -> Result<(), opencv::Error> {
    match ball {
        Some(ref b) => {
            // ── Thick circle around ball — hard to miss ───────────────
            imgproc::circle(
                frame,
                Point::new(b.px, b.py),
                b.radius + 6,
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                3,
                imgproc::LINE_AA,
                0,
            )?;
            // Inner circle
            imgproc::circle(
                frame,
                Point::new(b.px, b.py),
                b.radius + 2,
                Scalar::new(0.0, 180.0, 180.0, 0.0),
                1,
                imgproc::LINE_AA,
                0,
            )?;
            // Center crosshair dot
            imgproc::circle(
                frame,
                Point::new(b.px, b.py),
                4,
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;

            // ── Crosshair lines ───────────────────────────────────────
            imgproc::line(
                frame,
                Point::new(b.px - b.radius - 15, b.py),
                Point::new(b.px - b.radius - 5, b.py),
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::line(
                frame,
                Point::new(b.px + b.radius + 5, b.py),
                Point::new(b.px + b.radius + 15, b.py),
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::line(
                frame,
                Point::new(b.px, b.py - b.radius - 15),
                Point::new(b.px, b.py - b.radius - 5),
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::line(
                frame,
                Point::new(b.px, b.py + b.radius + 5),
                Point::new(b.px, b.py + b.radius + 15),
                Scalar::new(0.0, 255.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;

            // ── Zone + distance label next to ball ────────────────────
            let zone_str = if b.is_three { "3PT" } else { "2PT" };
            let zone_color = if b.is_three {
                Scalar::new(0.0, 200.0, 255.0, 0.0) // orange = 3PT
            } else {
                Scalar::new(0.0, 255.0, 140.0, 0.0) // green = 2PT
            };
            // Shadow for readability
            imgproc::put_text(
                frame,
                zone_str,
                Point::new(b.px + b.radius + 9, b.py + 6),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.75,
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                4,
                imgproc::LINE_AA,
                false,
            )?;
            imgproc::put_text(
                frame,
                zone_str,
                Point::new(b.px + b.radius + 8, b.py + 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.75,
                zone_color,
                2,
                imgproc::LINE_AA,
                false,
            )?;

            let dist_str = format!("{:.1}ft", b.distance_ft);
            imgproc::put_text(
                frame,
                &dist_str,
                Point::new(b.px + b.radius + 9, b.py + 27),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                3,
                imgproc::LINE_AA,
                false,
            )?;
            imgproc::put_text(
                frame,
                &dist_str,
                Point::new(b.px + b.radius + 8, b.py + 26),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(200.0, 200.0, 200.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;

            // ── SHOT DETECTED — big bold banner above ball ────────────
            if b.smooth_vel_y < SHOT_RELEASE_VEL_Y {
                // Dark background rectangle for readability
                let text = "SHOT DETECTED";
                let tx = (b.px - 95).max(5);
                let ty = (b.py - b.radius - 35).max(40);
                imgproc::rectangle(
                    frame,
                    opencv::core::Rect::new(tx - 5, ty - 25, 210, 35),
                    Scalar::new(0.0, 0.0, 180.0, 0.0),
                    -1,
                    imgproc::LINE_AA,
                    0,
                )?;
                imgproc::rectangle(
                    frame,
                    opencv::core::Rect::new(tx - 5, ty - 25, 210, 35),
                    Scalar::new(0.0, 80.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;
                imgproc::put_text(
                    frame,
                    text,
                    Point::new(tx, ty),
                    imgproc::FONT_HERSHEY_DUPLEX,
                    0.65,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    false,
                )?;
            }

            // ── Status bar top-left ───────────────────────────────────
            // Green filled background when locked
            imgproc::rectangle(
                frame,
                opencv::core::Rect::new(5, 5, 220, 36),
                Scalar::new(0.0, 140.0, 0.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::rectangle(
                frame,
                opencv::core::Rect::new(5, 5, 220, 36),
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::put_text(
                frame,
                "BALL LOCKED",
                Point::new(12, 31),
                imgproc::FONT_HERSHEY_DUPLEX,
                0.7,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;

            // ── Velocity debug line below status ──────────────────────
            let vel_str = format!("vy:{:.0} svy:{:.0}", b.vel_y, b.smooth_vel_y);
            imgproc::put_text(
                frame,
                &vel_str,
                Point::new(8, 55),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.38,
                Scalar::new(0.0, 220.0, 220.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
        None => {
            let (bg_color, border_color, msg) = match tracker_state {
                TrackerState::Recovering => (
                    Scalar::new(0.0, 80.0, 180.0, 0.0), // dark blue = recovering
                    Scalar::new(0.0, 140.0, 255.0, 0.0),
                    "WAITING FOR BALL...",
                ),
                _ => (
                    Scalar::new(0.0, 0.0, 140.0, 0.0), // red = searching
                    Scalar::new(0.0, 0.0, 220.0, 0.0),
                    "SCANNING FOR BALL...",
                ),
            };

            // ── Red status bar when scanning ──────────────────────────
            imgproc::rectangle(
                frame,
                opencv::core::Rect::new(5, 5, 260, 36),
                Scalar::new(0.0, 0.0, 140.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::rectangle(
                frame,
                opencv::core::Rect::new(5, 5, 260, 36),
                Scalar::new(0.0, 0.0, 220.0, 0.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
            imgproc::put_text(
                frame,
                "SCANNING FOR BALL...",
                Point::new(12, 31),
                imgproc::FONT_HERSHEY_DUPLEX,
                0.6,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
    }
    Ok(())
}

fn draw_shot_dots(frame: &mut Mat, shot_dots: &[ShotDot]) -> Result<(), opencv::Error> {
    const DOT_COLOR_2PT_MAKE: (f64, f64, f64) = (0.0, 255.0, 140.0);
    const DOT_COLOR_2PT_MISS: (f64, f64, f64) = (0.0, 80.0, 255.0);
    const DOT_COLOR_3PT_MAKE: (f64, f64, f64) = (0.0, 220.0, 255.0);
    const DOT_COLOR_3PT_MISS: (f64, f64, f64) = (60.0, 0.0, 255.0);
    const DOT_RADIUS: i32 = 12;

    for dot in shot_dots {
        let (r, g, b) = match (dot.is_three, dot.made) {
            (false, true) => DOT_COLOR_2PT_MAKE,
            (false, false) => DOT_COLOR_2PT_MISS,
            (true, true) => DOT_COLOR_3PT_MAKE,
            (true, false) => DOT_COLOR_3PT_MISS,
        };
        imgproc::circle(
            frame,
            Point::new(dot.px, dot.py),
            DOT_RADIUS,
            Scalar::new(b, g, r, 0.0),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
        imgproc::circle(
            frame,
            Point::new(dot.px, dot.py),
            DOT_RADIUS,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            0,
        )?;
        let label = format!("{:.0}ft", dot.dist_ft);
        imgproc::put_text(
            frame,
            &label,
            Point::new(dot.px + DOT_RADIUS + 3, dot.py + 4),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

fn draw_legend(frame: &mut Mat) -> Result<(), opencv::Error> {
    let items: &[(&str, (f64, f64, f64))] = &[
        ("2PT Make", (0.0, 255.0, 140.0)),
        ("2PT Miss", (0.0, 80.0, 255.0)),
        ("3PT Make", (0.0, 220.0, 255.0)),
        ("3PT Miss", (60.0, 0.0, 255.0)),
    ];
    for (i, (label, (r, g, b))) in items.iter().enumerate() {
        let y = 55 + i as i32 * 22;
        imgproc::circle(
            frame,
            Point::new(15, y),
            6,
            Scalar::new(*b, *g, *r, 0.0),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
        imgproc::put_text(
            frame,
            label,
            Point::new(26, y + 5),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.42,
            Scalar::new(220.0, 220.0, 220.0, 0.0),
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

// ── Camera loop ───────────────────────────────────────────────────────

fn process_heatmap_camera(
    mut camera: videoio::VideoCapture,
    state: Arc<Mutex<GameState>>,
    frames: Arc<FrameStore>,
) -> Result<(), opencv::Error> {
    let mut frame = Mat::default();
    let mut frame_times: VecDeque<Instant> = VecDeque::new();
    let mut local_count: u64 = 0;
    let mut ball_candidate = BallCandidate::default();

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
            "3M",
            "-maxrate",
            "5M",
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
    println!(
        "Focal length: {:.1} (CALIB_PIXEL_DIAMETER={:.0})",
        focal_length(),
        CALIB_PIXEL_DIAMETER
    );

    loop {
        camera.read(&mut frame)?;
        if frame.empty() {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        local_count += 1;

        // Push raw frame to loopback BEFORE overlays
        let mut raw_buf = Vector::<u8>::new();
        let raw_params = Vector::<i32>::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, 85]);
        if imgcodecs::imencode(".jpg", &frame, &mut raw_buf, &raw_params).is_ok() {
            let _ = ffmpeg_loop_stdin.write_all(&raw_buf.to_vec());
        }

        // Ball detection every N frames
        let ball = if local_count % HEATMAP_DETECT_EVERY_N_FRAMES == 0 {
            let last_ball = state.lock().unwrap().ball_position.clone();
            detect_ball(&frame, &last_ball, &mut ball_candidate).unwrap_or(None)
        } else {
            state.lock().unwrap().ball_position.clone()
        };

        // Update state
        {
            let mut s = state.lock().unwrap();
            s.heatmap_frame_count = local_count;
            s.ball_position = ball.clone();

            if let Some(ref b) = ball {
                s.pending_zone = Some(b.is_three);
                s.total_players_detected += 1;
                s.current_players = vec![PlayerDetection {
                    id: get_timestamp(),
                    x: b.px,
                    y: b.py,
                    width: b.radius * 2,
                    height: b.radius * 2,
                    timestamp: b.last_seen_ms,
                }];
            } else {
                s.current_players.clear();
            }

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
        }

        // Draw overlays
        let shot_dots_snap = state.lock().unwrap().shot_dots.clone();
        if let Err(e) = draw_shot_dots(&mut frame, &shot_dots_snap) {
            eprintln!("Shot dots error: {}", e);
        }
        if let Err(e) = draw_ball_overlay(&mut frame, &ball, &ball_candidate.state) {
            eprintln!("Ball overlay error: {}", e);
        }
        if let Err(e) = draw_legend(&mut frame) {
            eprintln!("Legend error: {}", e);
        }

        // Encode and stream
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

// ── ESP32 listener ────────────────────────────────────────────────────

fn listen_to_esp32(state: Arc<Mutex<GameState>>) {
    thread::spawn(move || {
        let ports = ["/dev/ttyUSB0", "/dev/ttyACM0"];
        loop {
            for port in &ports {
                println!("Attempting to connect to ESP32 on {}", port);
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(port)
                {
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
                            let now_ms = get_timestamp();

                            let _hits: u32 = data
                                .split(',')
                                .find(|p| p.starts_with("HITS:"))
                                .and_then(|p| p[5..].parse().ok())
                                .unwrap_or(1);
                            let _delta: u32 = data
                                .split(',')
                                .find(|p| p.starts_with("DELTA:"))
                                .and_then(|p| p[6..].parse().ok())
                                .unwrap_or(0);

                            if data.starts_with("MAKE:") {
                                s.make_count += 1;
                                s.backboard_make_count += 1;
                                s.last_serial_event_ms = now_ms;
                                let zone = s.pending_zone;
                                s.record_shot_with_zone(true, "Make", 1280, 720, zone);
                                println!("Make! total {}", s.make_count);
                            } else if data.starts_with("SWISH:") {
                                s.make_count += 1;
                                s.swish_count += 1;
                                s.last_serial_event_ms = now_ms;
                                let zone = s.pending_zone;
                                s.record_shot_with_zone(true, "Swish", 1280, 720, zone);
                                println!("Swish! total {}", s.make_count);
                            } else if data.starts_with("BACK:") {
                                s.backboard_count += 1;
                                s.backboard_miss_count += 1;
                                s.last_serial_event_ms = now_ms;
                                let zone = s.pending_zone;
                                s.record_shot_with_zone(false, "Backboard Miss", 1280, 720, zone);
                                println!("Backboard miss: total {}", s.backboard_count);
                            } else if data.starts_with("RIM:") {
                                s.rim_count += 1;
                                s.last_serial_event_ms = now_ms;
                                println!("Rim contact: total {}", s.rim_count);
                            } else if data.starts_with("AIRBALL:") {
                                s.airball_count += 1;
                                s.last_serial_event_ms = now_ms;
                                s.record_shot(false, "Airball", 1280, 720);
                                println!("Airball (ESP32)! total: {}", s.airball_count);
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

// ── Airball watcher ───────────────────────────────────────────────────
// Watches ball velocity for a shot release signature (ball moving upward fast).
// Starts a 5-second window waiting for ESP32 confirmation.
// If no MAKE/SWISH/BACK/RIM fires within 5s → classified as AIRBALL.

fn start_airball_watcher(state: Arc<Mutex<GameState>>) {
    thread::spawn(move || {
        #[derive(PartialEq)]
        enum WatchState {
            Idle,
            WaitingForEsp32,
        }

        let mut watch_state = WatchState::Idle;
        let mut release_ms = 0u64;
        let mut serial_at_release = 0u64;
        let mut last_airball_ms = 0u64;
        let mut release_zone: Option<bool> = None;

        loop {
            thread::sleep(Duration::from_millis(50)); // 20hz check
            let now_ms = get_timestamp();

            let (ball, last_serial) = {
                let s = state.lock().unwrap();
                (s.ball_position.clone(), s.last_serial_event_ms)
            };

            match watch_state {
                WatchState::Idle => {
                    if let Some(ref b) = ball {
                        // ── Shot release conditions (ALL must be true) ────────────────
                        //
                        // 1. Ball moving upward fast enough to be a shot arc
                        let moving_up = b.smooth_vel_y < SHOT_RELEASE_VEL_Y;

                        // 2. Not a lateral pass — must be mostly vertical movement
                        let not_a_pass = b.vel_x.abs() < SHOT_RELEASE_VEL_X_MAX;

                        // 3. Ball must be in shooting zone — not right under the basket.
                        //    Under the basket = lower portion of frame (high py value).
                        //    A shot is released from mid-to-far distance, so ball should
                        //    be in the middle or far half of the frame vertically.
                        let not_under_basket = b.py < 480;

                        // 4. Ball must be at a shootable distance from camera.
                        //    Anything under 4ft is right under/at the basket — not a shot.
                        let shootable_distance = b.distance_ft > 4.0;

                        // 5. Ball must have been moving consistently upward — not just
                        //    a single noisy frame. Check velocity magnitude is significant.
                        let strong_upward = b.smooth_vel_y < -220.0;

                        // 6. Cooldown between shot detections — ignore re-triggers
                        let cooldown_ok =
                            now_ms.saturating_sub(last_airball_ms) > AIRBALL_COOLDOWN_MS;

                        if moving_up
                            && strong_upward
                            && not_a_pass
                            && not_under_basket
                            && shootable_distance
                            && cooldown_ok
                        {
                            watch_state = WatchState::WaitingForEsp32;
                            release_ms = now_ms;
                            serial_at_release = last_serial;
                            release_zone = Some(b.is_three);
                            println!(
                                "Shot release detected — vel_y:{:.0}px/s dist:{:.1}ft zone:{} — waiting 5s",
                                b.vel_y, b.distance_ft,
                                if b.is_three { "3PT" } else { "2PT" }
                            );
                        }
                    }
                }

                WatchState::WaitingForEsp32 => {
                    let elapsed = now_ms.saturating_sub(release_ms);

                    // ESP32 fired after release → normal shot result, cancel airball
                    if last_serial > serial_at_release {
                        println!("ESP32 confirmed shot — cancelling airball timer");
                        watch_state = WatchState::Idle;
                        continue;
                    }

                    // 5 second window expired with no ESP32 event → AIRBALL
                    if elapsed >= AIRBALL_WAIT_MS {
                        let cooldown_ok =
                            now_ms.saturating_sub(last_airball_ms) > AIRBALL_COOLDOWN_MS;

                        if cooldown_ok {
                            let mut s = state.lock().unwrap();
                            s.airball_count += 1;
                            s.record_shot_with_zone(false, "Airball", 1280, 720, release_zone);
                            println!(
                                "AIRBALL! zone:{} total:{}",
                                if release_zone == Some(true) {
                                    "3PT"
                                } else {
                                    "2PT"
                                },
                                s.airball_count
                            );
                            last_airball_ms = now_ms;
                        }

                        watch_state = WatchState::Idle;
                    }
                }
            }
        }
    });
}

// ── Cloud API ─────────────────────────────────────────────────────────

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
                let ball_zone = s
                    .ball_position
                    .as_ref()
                    .map(|b| if b.is_three { "3PT" } else { "2PT" })
                    .unwrap_or("--");
                let ball_dist_ft = s
                    .ball_position
                    .as_ref()
                    .map(|b| format!("{:.1}", b.distance_ft))
                    .unwrap_or_else(|| "--".to_string());

                json!({
                    "basketball_fps": 0,
                    "makes": s.make_count,
                    "attempts": s.attempts(),
                    "swishes": s.swish_count,
                    "backboard_makes": s.backboard_make_count,
                    "backboard_misses": s.backboard_miss_count,
                    "backboard_hits": s.backboard_count,
                    "rim_hits": s.rim_count,
                    "airballs": s.airball_count,
                    "fg_percent": fg,
                    "two_pt_makes": s.two_pt_makes,
                    "two_pt_attempts": s.two_pt_attempts,
                    "three_pt_makes": s.three_pt_makes,
                    "three_pt_attempts": s.three_pt_attempts,
                    "shot_dots": s.shot_dots,
                    "last_shot_type": s.last_shot_type,
                    "ball_zone": ball_zone,
                    "ball_dist_ft": ball_dist_ft,
                    "trajectories": 0,
                    "heatmap_fps": s.heatmap_fps,
                    "current_players": s.current_players.len() as u64,
                    "total_players_detected": s.total_players_detected,
                    "heatmap_points": s.shot_dots.len() as u64,
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

// ── Main ──────────────────────────────────────────────────────────────

fn main() -> Result<(), opencv::Error> {
    println!("========================================");
    println!("HOOP IQ - C922x Dual Stream 720p@60fps");
    println!("CECS490 Senior Project - Team 2");
    println!("========================================\n");
    println!("Ball distance estimation active.");
    println!(
        "Focal length: {:.1} px (calib: {:.0}px @ {}in)",
        focal_length(),
        CALIB_PIXEL_DIAMETER,
        CALIB_DISTANCE_INCHES as i32
    );
    println!(
        "3PT threshold: {:.0}ft from camera\n",
        THREE_PT_DISTANCE_INCHES / 12.0
    );

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
    start_airball_watcher(Arc::clone(&game_state));

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
