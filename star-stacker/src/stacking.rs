use opencv::core::{Mat, add_weighted};


pub fn stack_image(images: &[Mat]) -> Mat {
    if images.len() == 1 {
        return images.first().unwrap().clone();
    }

    let mut stacked = Mat::default();
    add_weighted(
        &images[0],
        1.0 / images.len() as f64,
        &images[1],
        1.0 / images.len() as f64,
        0.0,
        &mut stacked,
        -1
    ).expect("Unable to stack frames.");

    images.iter().skip(2).for_each(|x| {
        add_weighted(&stacked.clone(),
                     1.0,
                     x,
                     1.0 / images.len() as f64,
                     0.0,
                     &mut stacked,
                     -1).expect("Unable to stack frames.");
    });

    stacked
}
