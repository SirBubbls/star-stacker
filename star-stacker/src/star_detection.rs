use opencv::prelude::*;
use opencv::core::{Scalar, Mat, Vector, KeyPoint};
use log::debug;

static MAX_STARS_IN_IMAGE: i32 = 750;

pub fn probe_response_threshold(image: &Mat, target: i32) -> u8 {
    if target > MAX_STARS_IN_IMAGE { panic!("Target to big > {}", MAX_STARS_IN_IMAGE); }
    let mut keypoints = Vector::<KeyPoint>::default();

    let mut response_threshold: u8 = 100;

    let mut star_count = 0;
    while star_count < target || star_count > MAX_STARS_IN_IMAGE {

        let mut star_detector = opencv::xfeatures2d::StarDetector::create(
                5,
                response_threshold.into(),
                10,
                8,
                5
        ).unwrap();

        star_detector.detect(&image, &mut keypoints, &opencv::core::Mat::default())
                    .expect("Unable to detect stars from image.");

        star_count = keypoints.len() as i32;
        debug!("Got {} hits with response threshold {}", star_count, response_threshold);

        if star_count > MAX_STARS_IN_IMAGE {
            response_threshold += 2;
            continue;
        }
        response_threshold -= 2;
        if response_threshold < 1 {
            return 1
        }
    }

    response_threshold
}


pub fn detect_keypoints(image: &Mat, threshold: u8) -> Vector<KeyPoint> {
    let mut keypoints = Vector::<KeyPoint>::default();

    let mut star_detector = opencv::xfeatures2d::StarDetector::create(
            5,
            threshold as i32,
            10,
            8,
            5
    ).unwrap();

    star_detector.detect(&image, &mut keypoints, &opencv::core::Mat::default())
                 .expect("Unable to detect stars from image.");

    debug!("Found {} keypoints", keypoints.len());
    keypoints
}


#[allow(dead_code)]
pub fn draw_keypoints(image: &Mat, keypoints: &Vector<KeyPoint>) -> Mat {
    let overlay_color = Scalar::new(0.0, 0.0, 255.0, 0.0);
    let mut overlay_image = image.clone();

    opencv::features2d::draw_keypoints(&image,
                                       keypoints,
                                       &mut overlay_image,
                                       overlay_color,
                                       opencv::features2d::DrawMatchesFlags::DRAW_RICH_KEYPOINTS).unwrap();
    overlay_image
}
