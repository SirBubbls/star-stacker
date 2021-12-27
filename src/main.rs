use clap::{App, Arg};
use log::info;
use opencv::core::{KeyPoint, Mat, Vector};
use opencv::imgcodecs::imwrite;
use stacking::stack_image;
use std::time;

mod alignment;
mod image_loading;
mod stacking;
mod star_detection;

#[cfg(feature = "opencvx")]
fn im_series_star_detectionx(images: &[Mat], target: i32) -> Vec<Vector<KeyPoint>> {
    // threshold value probing
    info!("Probing for a good threshold value...");
    let threshold = star_detection::probe_response_threshold(&images[0], target);

    images
        .iter()
        .map(|im| star_detection::detect_keypoints(&im, threshold))
        .collect::<Vec<Vector<KeyPoint>>>()
}

#[cfg(not(feature = "opencvx"))]
fn im_series_star_detection(images: &[Mat]) -> Vec<Vector<KeyPoint>> {
    images
        .iter()
        .map(star_detection::detect_keypoints)
        .collect::<Vec<Vector<KeyPoint>>>()
}

fn main() {
    env_logger::init();
    let cli_matches = App::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .args(&vec![
            Arg::with_name("input")
                .short("i")
                .long("input")
                .value_name("GLOB")
                .required(true)
                .help("path to input image files as a glob pattern (\"./images/*.png\")")
                .takes_value(true),
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("OUTPUT_FILE")
                .takes_value(true),
            #[cfg(feature = "opencvx")]
            Arg::with_name("target_landmarks")
                .value_name("N_TARGET_LANDMARKS")
                .long("target")
                .help(
                    "this value describes roughly how many stars should be detected in each image",
                )
                .takes_value(true)
                .default_value("20"),
            Arg::with_name("matching_precision")
                .value_name("MATCHIN_PRECISION")
                .long("precision")
                .help("max pixel distance between matched stars")
                .takes_value(true)
                .default_value("3.5"),
        ])
        .get_matches();

    // load images
    let images = image_loading::load_image_series(cli_matches.value_of("input").unwrap());

    // extract stars
    let t_start = time::Instant::now();
    let keypoints = {
        #[cfg(feature = "opencvx")]
        {
            im_series_star_detectionx(
                &images,
                cli_matches
                    .value_of("target_landmarks")
                    .unwrap()
                    .parse::<i32>()
                    .expect("Pass in a number as target."),
            )
        }
        #[cfg(not(feature = "opencvx"))]
        {
        im_series_star_detection(&images)
        }
    };

    let mut avg_stars: f32 = 0.0;
    keypoints.iter().for_each(|stars| {
        avg_stars += (stars.len() / images.len()) as f32;
    });
    info!(
        "Detected stars (avg. per image {}) in ~{}ms (total)",
        avg_stars,
        t_start.elapsed().as_millis()
    );

    // aligning frames
    info!("Aligning {} frames", images.len());
    let matching_precision = cli_matches
        .value_of("matching_precision")
        .unwrap()
        .parse::<f32>()
        .expect("You need to pass in a float as a precision parameter");
    let t_start = time::Instant::now();
    let aligned = alignment::align_series(&images, &keypoints, matching_precision);
    info!("Alignment took ~{}ms", t_start.elapsed().as_millis());

    // stacking frames
    info!("Stacking {} frames", aligned.len());
    let t_start = time::Instant::now();
    let stacked = stack_image(&aligned);
    info!("Stacking took ~{}ms", t_start.elapsed().as_millis());

    // writing output file
    info!(
        "Writing file to '{}'",
        cli_matches.value_of("output").unwrap()
    );
    imwrite(
        cli_matches.value_of("output").unwrap(),
        &stacked,
        &Vector::from_slice(&[]),
    )
    .expect("Unable to write output file");
}
