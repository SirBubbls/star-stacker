use alignment::find_alignment_homography;
use clap::{App, Arg};
use opencv::{
    core::DecompTypes,
    features2d::draw_matches,
    highgui::{imshow, wait_key},
    imgproc::{get_affine_transform, warp_perspective},
};
// use rayon::prelude::*;
use log::info;
use opencv::core::{invert, KeyPoint, Mat, Scalar, Vector};
use opencv::imgcodecs::imwrite;
use stacking::stack_image;

mod alignment;
mod image_loading;
mod stacking;
mod star_detection;

fn main() {
    env_logger::init();
    let cli_matches = App::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .value_name("GLOB")
                .required(true)
                .help("")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("OUTPUT_FILE")
                .takes_value(true),
        )
        .get_matches();

    // load images
    let images = image_loading::load_image_series(cli_matches.value_of("input").unwrap());
    let stacked = stack_image(&images);
    imshow("stacked without alignment", &stacked).unwrap();
    wait_key(0).unwrap();

    // extract stars
    let keypoints = images
        .iter()
        .map(star_detection::detect_keypoints)
        .collect::<Vec<Vector<KeyPoint>>>();

    let aligned = alignment::align_series(&images, &keypoints);
    let stacked = stack_image(&aligned);
    imshow("Stacked and aligned", &stacked).unwrap();

    // genrate matches
    let matches = alignment::get_matches(&keypoints[1], &keypoints[0]);
    let mut visualized = Mat::default();
    draw_matches(
        &images[1],
        &keypoints[1],
        &images[0],
        &keypoints[0],
        &matches,
        &mut visualized,
        Scalar::all(-1.0),
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        &Vector::default(),
        opencv::features2d::DrawMatchesFlags::DEFAULT,
    )
    .unwrap();
    imshow("MATCHES", &visualized).unwrap();
    wait_key(0).unwrap();

    imwrite(
        cli_matches.value_of("output").unwrap(),
        &stacked,
        &Vector::from_slice(&[]),
    )
    .expect("Unable to write output file");
}
