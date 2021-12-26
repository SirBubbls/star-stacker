use clap::{App, Arg};
use log::info;
use opencv::core::{KeyPoint, Vector};
use opencv::imgcodecs::imwrite;
use stacking::stack_image;
use std::time;

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

    // extract stars
    let t_start = time::Instant::now();
    let keypoints = images
        .iter()
        .map(star_detection::detect_keypoints)
        .collect::<Vec<Vector<KeyPoint>>>();
    let mut avg_stars: f32 = 0.0;
    keypoints.iter().for_each(|stars| {avg_stars += (stars.len() / images.len()) as f32;});
    info!("Detected stars (avg. per image {}) in ~{}ms (total)", avg_stars, t_start.elapsed().as_millis());

    // aligning frames
    info!("Aligning {} frames", images.len());
    let t_start = time::Instant::now();
    let aligned = alignment::align_series(&images, &keypoints);
    info!("Alignment took ~{}ms", t_start.elapsed().as_millis());

    // stacking frames
    info!("Stacking {} frames", aligned.len());
    let t_start = time::Instant::now();
    let stacked = stack_image(&aligned);
    info!("Stacking took ~{}ms", t_start.elapsed().as_millis());

    // writing output file
    info!("Writing file to '{}'", cli_matches.value_of("output").unwrap());
    imwrite(
        cli_matches.value_of("output").unwrap(),
        &stacked,
        &Vector::from_slice(&[]),
    ).expect("Unable to write output file");
}
