use log::debug;
use opencv::core::{KeyPoint, Mat, Scalar, Vector};
use opencv::prelude::*;
#[cfg(feature = "opencvx")]
use opencv::xfeatures2d::StarDetector;

#[cfg(feature = "opencvx")]
static MAX_STARS_IN_IMAGE: i32 = 750;

#[cfg(feature = "opencvx")]
pub fn probe_response_threshold(image: &Mat, target: i32) -> u8 {
    if target > MAX_STARS_IN_IMAGE {
        panic!("Target to big > {}", MAX_STARS_IN_IMAGE);
    }

    let mut response_threshold: u8 = 100;

    let mut star_count = 0;
    while star_count < target || star_count > MAX_STARS_IN_IMAGE {
        let keypoints = detect_keypoints(&image, response_threshold);

        star_count = keypoints.len() as i32;
        debug!(
            "Got {} hits with response threshold {}",
            star_count, response_threshold
        );

        if star_count > MAX_STARS_IN_IMAGE {
            response_threshold += 2;
            continue;
        }
        response_threshold -= 2;
        if response_threshold < 1 {
            return 1;
        }
    }

    response_threshold
}


#[cfg(feature = "opencvx")]
pub fn detect_keypoints(image: &Mat, threshold: u8) -> Vector<KeyPoint> {
    let mut keypoints = Vector::<KeyPoint>::default();
    let mut star_detector = StarDetector::create(5, threshold as i32, 10, 8, 5).unwrap();
    star_detector
        .detect(&image, &mut keypoints, &opencv::core::Mat::default())
        .expect("Unable to detect stars from image.");

    debug!("Found {} keypoints", keypoints.len());
    keypoints
}


#[cfg(not(feature = "opencvx"))]
pub fn detect_keypoints(image: &Mat, threshold: f32) -> Vector<KeyPoint> {
    let mut detector = opencv::features2d::SimpleBlobDetector::create(
        opencv::features2d::SimpleBlobDetector_Params {
            threshold_step: threshold,
            min_threshold: 150.0,
            max_threshold: 255.0,
            min_repeatability: 3,
            min_dist_between_blobs: 5.0,
            filter_by_color: false,
            blob_color: 0,
            filter_by_area: true,
            min_area: 5.0,
            max_area: 30.0,
            filter_by_circularity: false,
            min_circularity: 0.0,
            max_circularity: 0.0,
            filter_by_inertia: false,
            min_inertia_ratio: 0.0,
            max_inertia_ratio: 0.0,
            filter_by_convexity: false,
            min_convexity: 0.0,
            max_convexity: 0.0,
        }
    ).unwrap();

    let mut keypoints = Vector::<KeyPoint>::default();
    detector
        .detect(&image, &mut keypoints, &opencv::core::Mat::default())
        .expect("Unable to detect stars from image.");

    debug!("Found {} keypoints", keypoints.len());
    keypoints
}


#[allow(dead_code)]
pub fn draw_keypoints(image: &Mat, keypoints: &Vector<KeyPoint>) -> Mat {
    let overlay_color = Scalar::new(0.0, 0.0, 255.0, 0.0);
    let mut overlay_image = image.clone();

    opencv::features2d::draw_keypoints(
        &image,
        keypoints,
        &mut overlay_image,
        overlay_color,
        opencv::features2d::DrawMatchesFlags::DRAW_RICH_KEYPOINTS,
    )
    .unwrap();
    overlay_image
}
