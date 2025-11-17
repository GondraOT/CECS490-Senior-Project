// CECS490 Senior Project
// Hoop IQ
// Christopher Hong, Gondra Kelly, Matthew Marguiles, Alfonso Mejia Vasquez, Carloz Orozco

use opencv::{
    core::{self, Mat, Point, Scalar, Size, Vector, BORDER_DEFAULT},
    highgui, imgproc,
    prelude::*,
    videoio,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Basketball Detector - Windows Version");
    println!("======================================");

    // Try to open camera with better error handling
    println!("Attempting to open camera...");
    let mut cam = match videoio::VideoCapture::new(0, videoio::CAP_V4L2) {
        // changed CAP_DSHOW to CAP_V4L2[Video4Linux2] for Raspberry Pi compatibility?
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error opening camera with DirectShow: {}", e);
            println!("Trying with default backend...");
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?
        }
    };

    if !videoio::VideoCapture::is_opened(&cam)? {
        eprintln!("Error: Could not open camera");
        eprintln!("\nTroubleshooting tips:");
        eprintln!("1. Check if camera is connected");
        eprintln!("2. Close other apps using the camera (Skype, Teams, etc.)");
        eprintln!("3. Check Windows camera permissions");
        eprintln!("4. Try running as administrator");
        return Ok(());
    }

    println!("Camera opened successfully!");

    // Set camera properties with error checking
    match cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0) {
        Ok(_) => println!("Resolution set to 640x480"),
        Err(e) => println!("Warning: Could not set width: {}", e),
    }

    match cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0) {
        Ok(_) => {}
        Err(e) => println!("Warning: Could not set height: {}", e),
    }

    // Get actual camera properties
    let actual_width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let actual_height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    println!("Actual resolution: {}x{}", actual_width, actual_height);

    highgui::named_window("Basketball Detector", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Mask", highgui::WINDOW_AUTOSIZE)?;

    println!("\n=== Controls ===");
    println!("Press 'q' or ESC to quit");
    println!("Press 's' to save current frame");
    println!("\n=== Detection Info ===");
    println!("Looking for orange/black basketballs...\n");

    let mut frame = Mat::default();
    let mut hsv = Mat::default();
    let mut mask_orange = Mat::default();
    let mut mask_black = Mat::default();
    let mut combined_mask = Mat::default();
    let mut temp_mask = Mat::default();
    let mut blurred = Mat::default();
    let mut result = Mat::default();

    let mut frame_count = 0;
    let mut detection_count = 0;

    loop {
        // Read frame with error handling
        if let Err(e) = cam.read(&mut frame) {
            eprintln!("Error reading frame: {}", e);
            continue;
        }

        if frame.empty() {
            eprintln!("Warning: Empty frame captured (frame {})", frame_count);
            continue;
        }

        frame_count += 1;

        // Convert to HSV color space
        imgproc::cvt_color(
            &frame,
            &mut hsv,
            imgproc::COLOR_BGR2HSV,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Define color ranges for basketball detection
        // Orange range (basketball body)
        let lower_orange = Scalar::new(5.0, 100.0, 100.0, 0.0);
        let upper_orange = Scalar::new(25.0, 255.0, 255.0, 0.0);

        // Black range (basketball lines) - more lenient for Windows cameras
        let lower_black = Scalar::new(0.0, 0.0, 0.0, 0.0);
        let upper_black = Scalar::new(180.0, 255.0, 60.0, 0.0);

        // Create masks for orange and black
        core::in_range(&hsv, &lower_orange, &upper_orange, &mut mask_orange)?;
        core::in_range(&hsv, &lower_black, &upper_black, &mut mask_black)?;

        // Combine masks
        core::bitwise_or(
            &mask_orange,
            &mask_black,
            &mut combined_mask,
            &core::no_array(),
        )?;

        // Morphological operations to reduce noise
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            Size::new(5, 5),
            Point::new(-1, -1),
        )?;

        // Close operation (remove small holes)
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

        // Open operation (remove small noise)
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

        // Blur the mask for better circle detection
        imgproc::gaussian_blur(
            &combined_mask,
            &mut blurred,
            Size::new(9, 9),
            2.0,
            2.0,
            BORDER_DEFAULT,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Detect circles using Hough Circle Transform
        let mut circles = Vector::<core::Vec3f>::new();
        imgproc::hough_circles(
            &blurred,
            &mut circles,
            imgproc::HOUGH_GRADIENT,
            1.0,   // dp: inverse ratio of accumulator resolution
            50.0,  // minDist: minimum distance between circle centers
            100.0, // param1: Canny edge threshold
            30.0,  // param2: accumulator threshold (lower = more circles)
            20,    // minRadius: minimum circle radius
            200,   // maxRadius: maximum circle radius
        )?;

        // Copy frame for drawing
        frame.copy_to(&mut result)?;

        // Draw detected circles
        let num_detections = circles.len();
        if num_detections > 0 {
            detection_count += 1;
        }

        for i in 0..num_detections {
            let circle = circles.get(i)?;
            let center = Point::new(circle[0] as i32, circle[1] as i32);
            let radius = circle[2] as i32;

            // Draw circle outline (green)
            imgproc::circle(
                &mut result,
                center,
                radius,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                3,
                imgproc::LINE_AA,
                0,
            )?;

            // Draw center point (red)
            imgproc::circle(
                &mut result,
                center,
                5,
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_AA,
                0,
            )?;

            // Add text label
            let text = format!("Basketball r={}", radius);
            imgproc::put_text(
                &mut result,
                &text,
                Point::new(center.x - 50, center.y - radius - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;

            // Print detection info (only every 30 frames to avoid spam)
            if frame_count % 30 == 0 {
                println!(
                    "Frame {}: Basketball detected at ({}, {}) with radius {}",
                    frame_count, center.x, center.y, radius
                );
            }
        }

        // Add frame counter to display
        let info_text = format!("Frame: {} | Detections: {}", frame_count, num_detections);
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

        // Show results
        highgui::imshow("Basketball Detector", &result)?;
        highgui::imshow("Mask", &combined_mask)?;

        // Handle keyboard input
        let key = highgui::wait_key(1)?;

        if key == 'q' as i32 || key == 27 {
            // 'q' or ESC
            println!("\nExiting...");
            println!("Total frames processed: {}", frame_count);
            println!("Frames with detections: {}", detection_count);
            break;
        } else if key == 's' as i32 {
            // 's' to save
            let filename = format!("basketball_frame_{}.jpg", frame_count);
            opencv::imgcodecs::imwrite(&filename, &result, &Vector::new())?;
            println!("Saved frame to: {}", filename);
        }
    }

    // Cleanup
    highgui::destroy_all_windows()?;
    println!("Windows closed successfully!");

    Ok(())
}
