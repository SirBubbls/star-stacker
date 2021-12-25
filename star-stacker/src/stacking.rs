use opencv::core::{Mat, add_weighted};
use opencv::photo::fast_nl_means_denoising_colored;
use log::info;
use rayon::prelude::*;


pub fn stack_image(images: &[Mat]) -> Mat {
    if images.len() == 1 {
        return images.first().unwrap().clone();
    }

    // info!("Applying Denoising");
    // let images = images.iter().map(|im| {
    //     let mut dest = Mat::default();
    //     fast_nl_means_denoising_colored(im, &mut dest, 2.0, 2.0, 7, 7).unwrap();
    //     dest
    // }).collect::<Vec<Mat>>();

    let mut stacked = images[0].clone();

    info!("Stacking");
    images.iter().enumerate().skip(1).for_each(|(i, x)| {
        let alpha = 1.0 / i as f64;
        add_weighted(&stacked.clone(),
                     1.0 - alpha,
                     x,
                     alpha,
                     0.0,
                     &mut stacked,
                     -1).expect("Unable to stack frames.");
    });

    stacked
}
