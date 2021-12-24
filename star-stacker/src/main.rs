use alignment::find_alignment_homography;
use clap::{App, Arg};
use opencv::{highgui::{imshow, wait_key}, imgproc::{warp_perspective, get_affine_transform}, core::DecompTypes, features2d::draw_matches};
// use rayon::prelude::*;
use opencv::core::{Scalar, Mat, Vector, KeyPoint, invert};
use stacking::stack_image;
use log::info;

mod image_loading;
mod star_detection;
mod alignment;
mod stacking;


fn main() {
        env_logger::init();
        let matches = App::new(env!("CARGO_PKG_NAME"))
                .version(env!("CARGO_PKG_VERSION"))
                .author(env!("CARGO_PKG_AUTHORS"))
                .about(env!("CARGO_PKG_DESCRIPTION"))
                .arg(Arg::with_name("input")
                     .short("i")
                     .long("input")
                     .value_name("GLOB")
                     .required(true)
                     .help("")
                     .takes_value(true))
                .get_matches();

        // load images
        let images = image_loading::load_image_series(matches.value_of("input").unwrap());

        // extract stars
        let keypoints = images.iter()
                                  .map(star_detection::detect_keypoints)
                                  .collect::<Vec<Vector<KeyPoint>>>();

        // genrate matches
        let matches = alignment::get_matches(
                &keypoints[1],
                &keypoints[0]
        );

        info!("Matched {} stars", matches.len());
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
        ).unwrap();
        // imshow("MATCHES", &visualized).unwrap();


        // calculate homography of matches
        // alignment::find_affine_trans(&matches, &keypoints[0], &keypoints[1]);
        let homography = alignment::find_alignment_homography(
                &matches,
                &keypoints[1],
                &keypoints[0]
        );

        let mut inverted = Mat::default();
        invert(&homography, &mut inverted, DecompTypes::DECOMP_CHOLESKY as i32).unwrap();

        // warp image
        let mut warped = Mat::default();
        warp_perspective(&images[1],
                         &mut warped,
                         &homography,
                         opencv::core::Size_ { width: 1867, height: 2800 },
                         0,
                         0,
                         Scalar::default()
        ).unwrap();
        let stacked_warped = stack_image(&[images[0].clone(), warped]);
        imshow("Stacked with warp", &stacked_warped).unwrap();

        let stacked = stack_image(&images[0..2]);
        imshow("Stacked without warp", &stacked).unwrap();
        wait_key(0).unwrap();
}
