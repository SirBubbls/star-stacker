use opencv::imgcodecs::{imread, ImreadModes};
use opencv::core::Mat;
use log::debug;
use glob::glob;
use rayon::prelude::*;


/// Load a series of images by a glob path.
pub fn load_image_series(glob_path: &str) -> Vec<Mat> {
    let files: Vec<String> = glob(glob_path).unwrap()
                          .map(|file| file.unwrap().as_os_str().to_str().unwrap().to_string())
                          .collect();
    // files.par_iter()
    files.par_iter()
        .map(|path| {
          debug!("Loading image: {}", path);
          imread(path, ImreadModes::IMREAD_COLOR as i32).unwrap()
        })
        .collect()
}

