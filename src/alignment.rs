use log::debug;
use opencv::calib3d::{find_homography, RANSAC};
use opencv::core::{DMatch, KeyPoint, Mat, Point2f, Vector};
use opencv::core::Scalar;
use opencv::prelude::*;
use opencv::imgproc::warp_perspective;


/// Takes to list of keypoints and tries to match each one to another.
/// This matching algorithm matches each point $P$ points from the `source` dataset,
/// to the closest point $P'$ in the `target` dataset.
///
/// # Complexity
/// The runtime complexity is $n*m$ with $n$ as the number of `KeyPoint`s in
/// `source` and $m$ the number of `KeyPoint`s in `target`.
///
/// # Arguments
/// * source
/// * target
///
/// # Returns
/// A `Vec` of `DMatch`. Each containing information about the index of `source` set (`query_idx`)
/// and the `target` set (`train_idx`) and the pixel distance between both points $P$ and $P'$.
fn min_pos_delta_match(source: &Vector<KeyPoint>, target: &Vector<KeyPoint>) -> Vec<DMatch> {
    fn find_nearest(point: &KeyPoint, point_idx: usize, target_set: &Vector<KeyPoint>) -> DMatch {

        let mut lowest_idx: usize = 0;
        let mut lowest_dist: f32 = f32::MAX;

        target_set.to_vec().iter().enumerate().for_each(|(i, p)| {
            let dist = p.pt - point.pt;
            let dist = dist.dot(dist);
            if lowest_dist > dist {
                lowest_idx = i;
                lowest_dist = dist;
            }
        });

        DMatch::new(point_idx as i32, lowest_idx as i32, lowest_dist.sqrt()).unwrap()
    }

    let mut src: Vec<i32> = vec![];
    let mut dest: Vec<i32> = vec![];
    source.iter()
          .enumerate()
          .map(|(i, point)| find_nearest(&point, i, target))
          .filter(|m| {
              if src.contains(&m.query_idx) || dest.contains(&m.train_idx) { return false }
              src.push(m.query_idx);
              dest.push(m.train_idx);
              true
          })
          .collect()
}

pub fn get_matches(
    source: &Vector<KeyPoint>,
    dest: &Vector<KeyPoint>,
    precision: f32
) -> Vector<DMatch> {
    let matches: Vector<DMatch> = Vector::from_iter(
        min_pos_delta_match(source, dest)
            .into_iter()
            .filter(|x| x.distance < precision),
    );
    matches.iter().for_each(|m| {
        debug!(
            "Match {} to {} with distance: {}",
            m.train_idx, m.query_idx, m.distance
        )
    });
    matches
}


/// Returns the homography of
///
/// # Arguments
/// * `matches`:
/// * `source_kp`:
/// * `dest_kp`:
///
/// # Returns
pub fn find_alignment_homography(
    matches: &Vector<DMatch>,
    source_kp: &Vector<KeyPoint>,
    dest_kp: &Vector<KeyPoint>,
) -> Mat {
    let src: Vector<Point2f> = matches
        .iter()
        .map(|mat| source_kp.get(mat.query_idx as usize).unwrap().pt)
        .collect();

    let dest: Vector<Point2f> = matches
        .iter()
        .map(|mat| dest_kp.get(mat.train_idx as usize).unwrap().pt)
        .collect();

    find_homography(&src,
                    &dest,
                    &mut Mat::default(),
                    RANSAC,
                    5.0
    ).expect("Unable to find alignment homography for an image pair.")
}


pub fn align_series(images: &[Mat], keypoints: &[Vector<KeyPoint>], precisison: f32) -> Vec<Mat> {
    let size = images[0].size().unwrap();

    let transformations = keypoints.iter()
                                   .enumerate()
                                   .skip(1)
                                   .rev()
                                   .map(|(i, source_kp)| {
                                       let target_kp = &keypoints[i - 1];
                                       let matches = get_matches(source_kp, target_kp, precisison);
                                       find_alignment_homography(&matches, source_kp, target_kp)
    }).collect::<Vec<Mat>>();
    let mut transformed = images.iter().enumerate().skip(1).map(|(i, image)| {
        let mut transformed = image.clone();

        transformations.iter().take(i).for_each(|trans| {
            warp_perspective(&transformed.clone(),
                             &mut transformed,
                             trans,
                             size,
                             0,
                             0,
                             Scalar::all(-1.0)
            ).expect("Unable to apply transformation.")
        });
        transformed
    }).collect::<Vec<Mat>>();

    transformed.push(images[0].clone());
    transformed
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_delta_match() {
        let kp1 = Vector::from_iter(vec![
            KeyPoint::new_coords(10.0, 10.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
            KeyPoint::new_coords(0.0, 10.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
        ]);
        let kp2 = Vector::from_iter(vec![
            KeyPoint::new_coords(0.0, 9.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
            KeyPoint::new_coords(10.0, 9.0, 1.0, 0.0, 0.0, 0, 0).unwrap(),
        ]);

        let result = min_pos_delta_match(&kp1, &kp2);

        assert_eq!(result[0].train_idx, 1);
        assert_eq!(result[1].train_idx, 0);
    }
}
