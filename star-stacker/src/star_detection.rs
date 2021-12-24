use opencv::prelude::*;
use opencv::core::{Scalar, Mat, Vector, KeyPoint};
use log::debug;


pub fn detect_keypoints(image: &Mat) -> Vector<KeyPoint> {
    let mut keypoints = Vector::<KeyPoint>::default();

    let mut star_detector = opencv::xfeatures2d::StarDetector::create(
            5,
            5,
            10,
            8,
            5
    ).unwrap();

    star_detector.detect(&image, &mut keypoints, &opencv::core::Mat::default())
                 .expect("Unable to detect stars from image.");

    debug!("Found {} keypoints", keypoints.len());
    keypoints
}

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
