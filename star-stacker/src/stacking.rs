use opencv::core::{Mat, add_weighted};


pub fn stack_image(images: &[Mat]) -> Mat {
    if images.len() == 1 {
        return images.first().unwrap().clone();
    }

    let mut stacked = images[0].clone();

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
